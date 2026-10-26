"""Match supplied industry losses to the modelled wind field.

Shared by the backtest endpoint and ``scripts/calibrate_damage_curves.py``
so the fit and the screen use the same county matching.

County rules:
  - reported loss, county in the wind field with TIV -> observation
  - reported loss, county outside the wind field      -> footprint miss
  - reported loss, no TIV for that segment            -> exposure gap
  - modelled loss, no reported row                    -> unreported
ASSUMPTION: a county missing from the loss file is unknown, not zero. It is
left out of county scoring but its modelled loss still counts in the event
total, because PCS event totals include every county.

Below-materiality rows (reported zeros included) are scored as "somewhere
between 0 and max(reported, MATERIALITY_LOSS)": the fit is only penalised
when the model goes above that ceiling. They stay in every event total.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from .curve_fit import Observation
from .damage_curves import BAND_MID, CurveSet, interp_weights
from .event_evidence import county_evidence, miss_peril
from .event_evidence import summary as evidence_summary
from .event_loss import EventLoss, county_bins, event_loss
from .hurricane_impact import CountyImpact
from .industry_losses import LOSS_SEGMENTS, EventActuals, IndustryTiv


LandFn = Callable[[float], float]


def _land_fn(wind: float | LandFn) -> LandFn:
    if callable(wind):
        return wind
    factor = float(wind)
    return lambda mph: mph * factor


def indexed_bins(impact: CountyImpact, wind: float | LandFn, multiplier: float = 1.0) -> list[tuple[int, float]]:
    """Per-band weights so ground-up DR = sum(w x DR_b). Linear in the curve,
    which is what the fit needs. ``wind`` is a flat marine-to-land factor or a
    marine-mph-to-curve-mph function (``CurveSet.land_mph``)."""
    land = _land_fn(wind)
    out: dict[int, float] = {}
    for mph, frac in county_bins(impact):
        for idx, w in interp_weights(land(mph) * multiplier):
            out[idx] = out.get(idx, 0.0) + frac * w
    return sorted(out.items())


# ASSUMPTION: county rows under $250k on-levelled loss, or under 25 claims
# when the source carries counts, are allocation noise rather than a measured
# damage ratio. Across the 15 PCS events about 50-75% of positive county rows
# sit under $250k and hold 0.1-5% of the dollars, mostly round amounts.
MATERIALITY_LOSS = 250_000.0
MATERIALITY_CLAIMS = 25.0


def below_materiality(loss: float, claims: float | None) -> bool:
    if loss < MATERIALITY_LOSS:
        return True
    return claims is not None and claims < MATERIALITY_CLAIMS


@dataclass(slots=True)
class Matched:
    observations: dict[str, list[Observation]] = field(default_factory=dict)
    footprint_miss: list[tuple[str, str, float]] = field(default_factory=list)
    exposure_gap: list[tuple[str, str, float]] = field(default_factory=list)


def match_event(
    actuals: EventActuals,
    impacts: list[CountyImpact],
    tiv: IndustryTiv,
    wind: float | LandFn,
    wind_multiplier: dict[str, float] | None = None,
) -> Matched:
    by_gid = {imp.geography_id: imp for imp in impacts}
    mult = wind_multiplier or {}
    out = Matched(observations={seg: [] for seg in LOSS_SEGMENTS})
    for (gid, seg), loss in sorted(actuals.county.items()):
        segs = tiv.county.get(gid) or {}
        value = float(segs.get(seg, 0.0))
        imp = by_gid.get(gid)
        bins = indexed_bins(imp, wind, mult.get(gid, 1.0)) if imp is not None else []
        if not bins:
            if loss > 0:
                out.footprint_miss.append((gid, seg, loss))
            continue
        if value <= 0:
            if loss > 0:
                out.exposure_gap.append((gid, seg, loss))
            continue
        small = below_materiality(loss, actuals.county_claims.get((gid, seg)))
        out.observations[seg].append(
            Observation(
                event=actuals.storm_id,
                geography_id=gid,
                tiv=value,
                bins=bins,
                actual=loss,
                upper_bound=max(loss, MATERIALITY_LOSS) if small else None,
            )
        )
    return out


def _log_ratio(modelled: float, actual: float) -> float | None:
    if modelled > 0 and actual > 0:
        return math.log(modelled / actual)
    return None


def _rms(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(v * v for v in values) / len(values))


def backtest(
    actuals: EventActuals,
    impacts: list[CountyImpact],
    tiv: IndustryTiv,
    curve: CurveSet,
    evidence: dict | None = None,
    wind_multiplier: dict[str, float] | None = None,
) -> dict:
    """Modelled vs reported for one event, by segment and county.

    ``evidence`` (see ``event_evidence``) adds observed wind and reported
    perils per county, and splits out-of-field dollars by the peril that
    was reported there."""
    modelled: EventLoss = event_loss(impacts, tiv, curve, wind_multiplier=wind_multiplier)
    by_gid = {c.impact.geography_id: c for c in modelled.counties}
    matched = match_event(actuals, impacts, tiv, curve.land_mph, wind_multiplier)
    actual_seg = actuals.segment_totals()

    segments: dict[str, dict] = {}
    for seg in LOSS_SEGMENTS:
        act = actual_seg.get(seg, 0.0)
        mod = modelled.by_segment.get(seg, 0.0)
        material = [o for o in matched.observations[seg] if o.upper_bound is None]
        county_err = [
            e for o in material
            if (e := _log_ratio(by_gid[o.geography_id].loss[seg], o.actual)) is not None
        ]
        over_ceiling = sum(
            1 for o in matched.observations[seg]
            if o.upper_bound is not None and by_gid[o.geography_id].loss[seg] > o.upper_bound
        )
        reported_in_field = sum(o.actual for o in matched.observations[seg])
        segments[seg] = {
            "actual": act,
            "modelled": mod,
            "ratio": (mod / act) if act > 0 else None,
            "logError": _log_ratio(mod, act),
            "countiesScored": len(matched.observations[seg]),
            "countiesMaterial": len(material),
            "countiesBelowMateriality": len(matched.observations[seg]) - len(material),
            # Below-materiality counties where the model is above the ceiling:
            # the only way those rows can disagree with the model.
            "countiesAboveMaterialityCeiling": over_ceiling,
            "countyLogRms": _rms(county_err),
            "countyLogMedian": sorted(county_err)[len(county_err) // 2] if county_err else None,
            "shareOfActualInWindField": (reported_in_field / act) if act > 0 else None,
        }

    reported = {gid for gid, _seg in actuals.county}
    rows: list[dict] = []
    for (gid, seg), act in actuals.county.items():
        cl = by_gid.get(gid)
        mod = cl.loss[seg] if cl else 0.0
        seg_tiv = cl.tiv[seg] if cl else float((tiv.county.get(gid) or {}).get(seg, 0.0))
        imp = cl.impact if cl else None
        rows.append({
            "geographyId": gid,
            "name": imp.name if imp else None,
            "state": gid.split("-")[1],
            "segment": seg,
            "actual": act,
            "modelled": mod,
            "tiv": seg_tiv,
            "modelledDr": cl.dr[seg] if cl else 0.0,
            "impliedDr": (act / seg_tiv) if seg_tiv > 0 else None,
            "maxWindKt": imp.max_wind_kt if imp else None,
            "logError": _log_ratio(mod, act),
            "inWindField": imp is not None and bool(county_bins(imp)),
            "observedWindMultiplier": (wind_multiplier or {}).get(gid),
            "claims": actuals.county_claims.get((gid, seg)),
            "belowMateriality": below_materiality(act, actuals.county_claims.get((gid, seg))),
            "evidence": county_evidence(evidence, gid),
        })
    rows.sort(key=lambda r: -abs(r["modelled"] - r["actual"]))

    unreported = [
        {
            "geographyId": c.impact.geography_id,
            "name": c.impact.name,
            "state": c.impact.state_usps,
            "modelled": c.total,
            "maxWindKt": c.impact.max_wind_kt,
        }
        for c in modelled.counties
        if c.impact.geography_id not in reported and c.total > 0
    ]

    total_act = actuals.total()
    return {
        "total": {
            "actual": total_act,
            "modelled": modelled.total,
            "ratio": (modelled.total / total_act) if total_act > 0 else None,
            "logError": _log_ratio(modelled.total, total_act),
        },
        "segments": segments,
        "counties": rows,
        "unreportedCounties": unreported[:200],
        "footprintMisses": [
            {
                "geographyId": g,
                "segment": s,
                "actual": a,
                "belowMateriality": below_materiality(a, actuals.county_claims.get((g, s))),
                "evidencePeril": miss_peril(county_evidence(evidence, g)) if evidence else None,
            }
            for g, s, a in matched.footprint_miss
        ],
        "footprintMissByPeril": _miss_by_peril(matched.footprint_miss, evidence),
        "evidence": evidence_summary(evidence),
        "materiality": {
            "lossBelow": MATERIALITY_LOSS,
            "claimsBelow": MATERIALITY_CLAIMS,
            "currency": actuals.currency,
        },
        "exposureGaps": [
            {"geographyId": g, "segment": s, "actual": a} for g, s, a in matched.exposure_gap
        ],
        "missingTivCounties": len(modelled.missing_tiv),
    }


def _miss_by_peril(misses: list[tuple[str, str, float]], evidence: dict | None) -> dict | None:
    if not evidence:
        return None
    out: dict[str, float] = {}
    for gid, _seg, loss in misses:
        peril = miss_peril(county_evidence(evidence, gid))
        out[peril] = out.get(peril, 0.0) + loss
    return out


def implied_dr_by_band(observations: list[Observation]) -> list[dict]:
    """Diagnostic: actual / TIV for material counties whose land wind sits
    mostly on one band midpoint."""
    out: dict[int, list[float]] = {}
    for o in observations:
        if not o.bins or o.upper_bound is not None:
            continue
        b, frac = max(o.bins, key=lambda bf: bf[1])
        if frac >= 0.7 and o.tiv > 0:
            out.setdefault(b, []).append(o.actual / o.tiv / frac)
    rows = []
    for b in sorted(out):
        vals = sorted(out[b])
        rows.append({
            "curveMph": BAND_MID[b],
            "n": len(vals),
            "median": vals[len(vals) // 2],
            "min": vals[0],
            "max": vals[-1],
        })
    return rows
