"""Train per-Saffir-Simpson damage ratios (mean + SD) from supplied PCS losses.

The impact panel prices a county as

    loss = TIV x sum_c(areaFraction_c x DR_c)        c = TS, Cat 1 .. Cat 5

with ``areaFraction_c`` from the same ``compute_impact`` wind bands the panel
gets (historical track, multiplier 2.5), plus the IBTrACS R34/R50
tropical-storm field the industry-loss engine uses. Inside hurricane-force
wind the two fields are the same; without the 34-63 kt field most
tropical-storm loss falls outside the footprint (Helene: $16B of $18B). That model
is linear in ``DR_c``, so it is fitted with the same objective, materiality
censoring, monotone projection and leave-one-event-out holdout as the 10 mph
curve calibration (``services/curve_fit.py``), with six category bins in
place of the 13 Form V-1 bands. The target is insured loss (PCS basis), on-
levelled to the industry TIV valuation date.

A category the data cannot see (in practice Cat 5) keeps the prior's shape,
scaled by the nearest identified category's fit / prior, and borrows that
category's sigma. ``--fix-ts`` holds TS at a value from elsewhere (the
tropical-storm-field fit) and gives it the pooled sigma.

Variance comes from the holdout, not the fit: every event is priced by the
table fitted without it. For each category the script reports
  - county sigma: log error of held-out material counties whose largest wind
    slice is that category,
  - event sigma: log error of the held-out event's summed loss over those
    counties (cells with at least EVENT_CELL_MIN of reported loss), weighted
    by sqrt(reported loss).
The panel shows mean +/- SD, so SD = mean x sqrt(exp(sigma_event^2) - 1),
the standard deviation of a lognormal with that mean.

Writes ``mockdata/damage_assumptions/sshws_calibrated.json``.

  python backend/scripts/train_category_damage.py
  python backend/scripts/train_category_damage.py --no-ts-field --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

from app.services.curve_fit import FitOptions, Observation, fit_with_holdout, predict
from app.services.damage_curves import load_curve_sets
from app.services.event_loss import resolve_storm
from app.services.hurricane_impact import DAMAGING_WIND_MULTIPLIER, compute_impact
from app.services.industry_losses import LOSS_SEGMENTS, data_dir, load_actuals, tiv_for_year
from app.services.loss_calibration import MATERIALITY_LOSS, below_materiality
from app.services.onlevel import onlevel_actuals

KT_TO_MPH = 1.15078
CATEGORIES = (0, 1, 2, 3, 4, 5)
LABEL = {0: "TS", 1: "Cat 1", 2: "Cat 2", 3: "Cat 3", 4: "Cat 4", 5: "Cat 5"}
# Marine 1-minute sustained kt. ASSUMPTION: Cat 5 is averaged up to 160 kt.
CAT_KT = {0: (34, 63), 1: (64, 82), 2: (83, 95), 3: (96, 112), 4: (113, 136), 5: (137, 160)}
ALL = "ALL"
# ASSUMPTION: an event x category cell with under $25M reported is too thin
# to say anything about that storm's error in that category.
EVENT_CELL_MIN = 25_000_000.0
MIN_EVENT_CELLS = 3
# ASSUMPTION: a category needs 3+ counties where it is the largest slice and
# 0.1% of the TIV x area to be identified by the data. Otherwise it keeps the
# prior's shape, scaled by the nearest identified category's fit / prior.
MIN_DOMINANT_COUNTIES = 3
MIN_TIV_AREA_SHARE = 0.001


def category_prior(prior_set, segment: str, cat: int) -> float:
    lo, hi = CAT_KT[cat]
    kts = [lo + (hi - lo) * k / 20 for k in range(21)]
    return sum(prior_set.dr(segment, kt * KT_TO_MPH) for kt in kts) / len(kts)


def category_bins(impact) -> list[tuple[int, float]]:
    return [(b.category, b.area_fraction) for b in impact.wind_bands if b.category >= 0 and b.area_fraction > 0]


def dominant(bins: list[tuple[int, float]]) -> int:
    return max(bins, key=lambda bf: bf[1])[0]


def _rms(values: list[float]) -> float | None:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else None


def _weighted_rms(cells: list[tuple[str, float, float]]) -> float | None:
    """RMS of (event, log error, reported loss) cells, weighted by sqrt(loss)
    like the fit, so a $50M storm does not count as much as Ian."""
    if not cells:
        return None
    w = [math.sqrt(a) for _ev, _e, a in cells]
    return math.sqrt(sum(wi * e * e for wi, (_ev, e, _a) in zip(w, cells)) / sum(w))


def lognormal_sd(mean: float, sigma: float) -> float:
    return mean * math.sqrt(math.exp(sigma * sigma) - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", nargs="*")
    parser.add_argument("--prior", default="kcc")
    parser.add_argument("--ridge", type=float, default=FitOptions().ridge)
    parser.add_argument("--alpha-event", type=float, default=FitOptions().alpha_event)
    parser.add_argument("--multiplier", type=float, default=DAMAGING_WIND_MULTIPLIER)
    parser.add_argument("--no-ts-field", action="store_true",
                        help="score on the hurricane-force field only (no IBTrACS R34/R50 field)")
    parser.add_argument("--no-onlevel", action="store_true")
    parser.add_argument("--fix-ts", type=float, default=None, metavar="PCT",
                        help="hold the TS ratio at PCT (all segments) instead of fitting it")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    actuals = load_actuals()
    wanted = [e.upper() for e in args.events] if args.events else sorted(actuals)
    prior_set = load_curve_sets()[args.prior]

    obs: dict[str, list[Observation]] = {seg: [] for seg in (*LOSS_SEGMENTS, ALL)}
    dom: dict[tuple[str, str], int] = {}
    book_tiv: dict[str, float] = {seg: 0.0 for seg in LOSS_SEGMENTS}
    events: list[dict] = []
    for storm_id in wanted:
        storm, _source = resolve_storm(storm_id)
        act = actuals[storm_id]
        tiv = tiv_for_year(storm.year)
        if not args.no_onlevel:
            act, _ = onlevel_actuals(act, tiv.as_of)
        impacts, *_ = compute_impact(storm, multiplier=args.multiplier, ts_field=not args.no_ts_field)
        by_gid = {imp.geography_id: imp for imp in impacts}
        counties: dict[str, dict[str, float]] = {}
        for (gid, seg), loss in act.county.items():
            counties.setdefault(gid, {})[seg] = loss
        in_field = miss = gap = 0.0
        for gid, losses in sorted(counties.items()):
            total = sum(losses.values())
            imp = by_gid.get(gid)
            bins = category_bins(imp) if imp else []
            segs = tiv.county.get(gid) or {}
            if not bins:
                miss += total
                continue
            if sum(segs.values()) <= 0:
                gap += total
                continue
            in_field += total
            dom[(storm_id, gid)] = dominant(bins)
            for seg in LOSS_SEGMENTS:
                value = float(segs.get(seg, 0.0))
                if value <= 0 or seg not in losses:
                    continue
                loss = losses[seg]
                small = below_materiality(loss, act.county_claims.get((gid, seg)))
                obs[seg].append(Observation(storm_id, gid, value, bins, loss,
                                            max(loss, MATERIALITY_LOSS) if small else None))
            claims = [act.county_claims.get((gid, s)) for s in losses]
            claims_total = sum(c for c in claims if c is not None) if any(c is not None for c in claims) else None
            small = below_materiality(total, claims_total)
            obs[ALL].append(Observation(storm_id, gid, float(sum(segs.values())), bins, total,
                                        max(total, MATERIALITY_LOSS) if small else None))
        events.append({
            "stormId": storm_id, "name": storm.name, "year": storm.year,
            "actualTotal": act.total(), "inField": in_field, "footprintMiss": miss,
            "exposureGap": gap,
        })
        print(f"{storm_id} {storm.name} {storm.year}: {act.total() / 1e9:6.1f}B on-level; "
              f"in field {in_field / 1e9:5.1f}B, outside {miss / 1e9:5.1f}B, "
              f"no TIV {gap / 1e9:4.1f}B")

    for segs in tiv_for_year(2026).county.values():
        for seg in LOSS_SEGMENTS:
            book_tiv[seg] += segs.get(seg, 0.0)
    book_total = sum(book_tiv.values())
    mix = {seg: v / book_total for seg, v in book_tiv.items()}

    fixed = ((0, args.fix_ts / 100.0),) if args.fix_ts is not None else ()
    opts = FitOptions(alpha_event=args.alpha_event, ridge=args.ridge, fixed=fixed)
    tables: dict[str, dict] = {}
    for seg in (ALL, *LOSS_SEGMENTS):
        if seg == ALL:
            prior = [sum(mix[s] * category_prior(prior_set, s, c) for s in LOSS_SEGMENTS) for c in CATEGORIES]
        else:
            prior = [category_prior(prior_set, seg, c) for c in CATEGORIES]
        prior = [max(1e-5, p) for p in prior]
        res = fit_with_holdout(obs[seg], prior, opts)

        county_err: dict[int, list[float]] = {c: [] for c in CATEGORIES}
        cells: dict[tuple[str, int], list[float]] = {}
        emp: dict[int, list[float]] = {c: [0.0, 0.0] for c in CATEGORIES}
        n_dom = {c: 0 for c in CATEGORIES}
        for o in obs[seg]:
            c = dom[(o.event, o.geography_id)]
            n_dom[c] += 1
            curve = res.oos_curves.get(o.event, res.curve)
            m = predict(o, curve)
            cell = cells.setdefault((o.event, c), [0.0, 0.0])
            cell[0] += m
            cell[1] += o.actual
            if o.upper_bound is None and m > 0 and o.actual > 0:
                county_err[c].append(math.log(m / o.actual))
                emp[c][0] += o.actual
                emp[c][1] += o.tiv * sum(f for _b, f in o.bins)
        event_err: dict[int, list[tuple[str, float, float]]] = {c: [] for c in CATEGORIES}
        for (ev, c), (m, a) in sorted(cells.items()):
            if a >= EVENT_CELL_MIN and m > 0:
                event_err[c].append((ev, math.log(m / a), a))
        every = [cell for c in CATEGORIES for cell in event_err[c]]
        pooled = _weighted_rms(every) or 1.0
        fitted = list(res.curve)
        identified = [
            c for c in CATEGORIES
            if n_dom[c] >= MIN_DOMINANT_COUNTIES and res.band_weight[c] >= MIN_TIV_AREA_SHARE
        ]
        for c in CATEGORIES:
            if c in dict(fixed):
                continue
            if c not in identified and identified:
                ref = min(identified, key=lambda k: abs(k - c))
                fitted[c] = prior[c] * res.curve[ref] / prior[ref]
        for c in CATEGORIES[1:]:
            fitted[c] = max(fitted[c], fitted[c - 1])

        sigma_ev_by = {
            c: (_weighted_rms(event_err[c]) if len(event_err[c]) >= MIN_EVENT_CELLS else None)
            for c in CATEGORIES
        }
        rows = []
        for c in CATEGORIES:
            sigma_ev = sigma_ev_by[c]
            if c in dict(fixed):
                sigma_used, source = pooled, "pooled event holdout (ratio fixed)"
            elif c not in identified and identified:
                ref = min(identified, key=lambda k: abs(k - c))
                sigma_used = sigma_ev_by[ref] if sigma_ev_by[ref] is not None else pooled
                source = f"{LABEL[ref]} event holdout (not identified)"
            elif sigma_ev is not None:
                sigma_used, source = sigma_ev, "event holdout"
            else:
                sigma_used, source = pooled, "pooled event holdout"
            mean = fitted[c]
            rows.append({
                "category": c,
                "label": LABEL[c],
                "meanPct": round(mean * 100, 3),
                "fitPct": round(res.curve[c] * 100, 3),
                "identified": c in identified,
                "fixed": c in dict(fixed),
                "sdPct": round(lognormal_sd(mean, sigma_used) * 100, 3),
                "priorPct": round(prior[c] * 100, 3),
                "tivAreaShare": round(res.band_weight[c], 4),
                "countiesDominant": n_dom[c],
                "empiricalPct": round(emp[c][0] / emp[c][1] * 100, 3) if emp[c][1] > 0 else None,
                "sigmaLogCountyHoldout": round(_rms(county_err[c]), 3) if county_err[c] else None,
                "countyBiasHoldout": round(sorted(county_err[c])[len(county_err[c]) // 2], 3) if county_err[c] else None,
                "sigmaLogEventHoldout": round(sigma_ev, 3) if sigma_ev is not None else None,
                "sigmaUsed": round(sigma_used, 3),
                "sigmaSource": source,
                "eventErrors": {ev: {"logError": round(e, 3), "actual": round(a)} for ev, e, a in event_err[c]},
            })
        tables[seg] = {
            "nObs": res.n_obs,
            "events": res.events,
            "pooledSigmaLogEvent": round(pooled, 3),
            "eventLogErrorHoldout": {k: round(v, 3) for k, v in res.oos_event_errors.items()},
            "categories": rows,
        }
        print(f"\n{seg}: {res.n_obs} county obs, pooled event sigma {pooled:.2f}")
        print("  cat     prior%   fit%  mean%    SD%   emp%   TIVxarea  n   sigC   biasC  sigE  cells")
        for r in rows:
            print(f"  {r['label']:<6} {r['priorPct']:7.2f} {r['fitPct']:6.2f} {r['meanPct']:6.2f} {r['sdPct']:6.2f} "
                  f"{(r['empiricalPct'] or 0):6.2f}  {r['tivAreaShare']:7.3f} {r['countiesDominant']:4d} "
                  f"{(r['sigmaLogCountyHoldout'] or 0):5.2f} {(r['countyBiasHoldout'] or 0):6.2f} "
                  f"{(r['sigmaLogEventHoldout'] or float('nan')):5.2f}  {len(r['eventErrors'])}")
        for ev, e in sorted(res.oos_event_errors.items()):
            print(f"    {ev} holdout event x{math.exp(e):.2f}")

    now = datetime.now(UTC)
    payload = {
        "id": f"sshws-calibrated-{now:%Y%m%d}",
        "createdAt": now.isoformat(timespec="seconds"),
        "basis": (
            "insured PCS loss (NFIP and LAE out; private/commercial flood or surge may be in), "
            "on-levelled to the TIV date"
        ),
        "windField": f"compute_impact, historical track, multiplier {args.multiplier}, "
                     + ("no" if args.no_ts_field else "with") + " tropical-storm field",
        "prior": args.prior,
        "ridge": args.ridge,
        "alphaEvent": args.alpha_event,
        "fixTsPct": args.fix_ts,
        "onLevel": not args.no_onlevel,
        "bookMix": {k: round(v, 4) for k, v in mix.items()},
        "events": events,
        "segments": tables,
    }
    if args.dry_run:
        print("Dry run, nothing written.")
        return
    out = data_dir() / "damage_assumptions" / "sshws_calibrated.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
