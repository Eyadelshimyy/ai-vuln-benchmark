"""Leave-one-event-out test of shared vs per-segment damage curves.

Compares three insured-loss curve structures on the supplied PCS events, using
the shipped calibration's matching, observed-wind shape, on-levelling, KCC
insured priors, smoothing, tail rule and materiality censoring:

  shared      one curve for every segment (13 parameters), prior = book-TIV
              weighted mean of the three insured KCC priors
  level       shared curve x one level per segment fitted on training event
              totals only (16 parameters)
  separate    one curve per segment (39 parameters)

Each fold holds out a whole event and refits everything on the rest. Scores are
log RMS of model/actual for event totals (unweighted and sqrt(loss)-weighted),
event x segment totals, and material counties. A paired event bootstrap gives
95% intervals for each alternative minus shared. Analysis only: nothing in
mockdata is written.

Run from the repo root (about five minutes):
  python backend/scripts/evaluate_segment_curves.py --out /tmp/segment-eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.curve_fit import FitOptions, Observation, fit_curve, predict
from app.services.damage_curves import insured_prior, land_mph, load_curve_sets
from app.services.event_evidence import load_evidence
from app.services.event_loss import resolve_storm, storm_impacts
from app.services.industry_losses import LOSS_SEGMENTS, load_actuals, tiv_for_year
from app.services.loss_calibration import match_event
from app.services.onlevel import onlevel_actuals
from app.services.wind_calibration import fit_land_shape

MODELS = ("shared", "level", "separate")
# Same options as calibrate_damage_curves.py defaults.
OPTS = FitOptions(alpha_event=1.0, ridge=0.05, smooth=2.0, tail_min_weight=5e-4)
FOOTPRINT_MULTIPLIER = 2.5


def observations() -> dict[str, list[Observation]]:
    actuals = load_actuals()
    stations = [
        station
        for storm_id in sorted(actuals)
        for station in ((load_evidence(storm_id) or {}).get("stations") or [])
    ]
    shape = fit_land_shape(stations)
    wind_factor = load_curve_sets()["kcc"].wind_factor

    def land(mph: float) -> float:
        return land_mph(mph, wind_factor, shape.exponent, shape.floor)

    out: dict[str, list[Observation]] = {segment: [] for segment in LOSS_SEGMENTS}
    for storm_id, supplied in sorted(actuals.items()):
        found = resolve_storm(storm_id)
        if found is None:
            raise SystemExit(f"{storm_id}: storm not found")
        storm, source = found
        tiv = tiv_for_year(storm.year)
        actual, _ = onlevel_actuals(supplied, tiv.as_of)
        matched = match_event(actual, storm_impacts(storm, source, FOOTPRINT_MULTIPLIER), tiv, land)
        for segment in LOSS_SEGMENTS:
            out[segment].extend(matched.observations[segment])
    return out


def shared_prior() -> list[float]:
    kcc = load_curve_sets()["kcc"]
    book = tiv_for_year(2026)
    weight = {s: sum(v.get(s, 0.0) for v in book.county.values()) for s in LOSS_SEGMENTS}
    total = sum(weight.values())
    priors = {s: insured_prior(kcc, s) for s in LOSS_SEGMENTS}
    bands = len(next(iter(priors.values())))
    return [sum(weight[s] * priors[s][b] for s in LOSS_SEGMENTS) / total for b in range(bands)]


def segment_levels(train: dict[str, list[Observation]], curve: list[float]) -> dict[str, float]:
    levels: dict[str, float] = {}
    for segment, rows in train.items():
        cells: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for row in rows:
            cells[row.event][0] += row.actual
            cells[row.event][1] += predict(row, curve)
        pairs = [(math.sqrt(a), math.log(a / m)) for a, m in cells.values() if a > 0 and m > 0]
        levels[segment] = math.exp(sum(w * e for w, e in pairs) / sum(w for w, _ in pairs))
    return levels


def holdout_cells(obs: dict[str, list[Observation]]) -> dict:
    """Per held-out event: segment [actual, model] cells and county log errors."""
    kcc = load_curve_sets()["kcc"]
    priors = {s: insured_prior(kcc, s) for s in LOSS_SEGMENTS}
    prior = shared_prior()
    events = sorted({row.event for rows in obs.values() for row in rows})
    out: dict[str, dict] = {}
    for held in events:
        train = {s: [r for r in rows if r.event != held] for s, rows in obs.items()}
        shared = fit_curve([r for rows in train.values() for r in rows], prior, OPTS).curve
        levels = segment_levels(train, shared)
        separate = {s: fit_curve(train[s], priors[s], OPTS).curve for s in LOSS_SEGMENTS}
        cell = {m: {"segments": {s: [0.0, 0.0] for s in LOSS_SEGMENTS}, "county": []} for m in MODELS}
        for segment in LOSS_SEGMENTS:
            for row in obs[segment]:
                if row.event != held:
                    continue
                base = predict(row, shared)
                model = {"shared": base, "level": base * levels[segment],
                         "separate": predict(row, separate[segment])}
                for name, value in model.items():
                    cell[name]["segments"][segment][0] += row.actual
                    cell[name]["segments"][segment][1] += value
                    if row.upper_bound is None and row.actual > 0 and value > 0:
                        cell[name]["county"].append(math.log(value / row.actual))
        out[held] = cell
        print(f"  fold {held} done", file=sys.stderr)
    return out


def scores(cells: dict, events: list[str], model: str) -> dict[str, float]:
    event_err, weighted, seg_err, county = [], [], [], []
    for event in events:
        c = cells[event][model]
        actual = sum(a for a, _ in c["segments"].values())
        modelled = sum(m for _, m in c["segments"].values())
        if actual > 0 and modelled > 0:
            e = math.log(modelled / actual)
            event_err.append(e)
            weighted.append((math.sqrt(actual), e))
        seg_err.extend(math.log(m / a) for a, m in c["segments"].values() if a > 0 and m > 0)
        county.extend(c["county"])

    def rms(xs: list[float]) -> float:
        return math.sqrt(sum(x * x for x in xs) / len(xs))

    return {
        "eventLogRms": rms(event_err),
        "eventLogRmsSqrtLossWeighted": math.sqrt(
            sum(w * e * e for w, e in weighted) / sum(w for w, _ in weighted)
        ),
        "eventSegmentLogRms": rms(seg_err),
        "materialCountyLogRms": rms(county),
    }


def summarise(cells: dict, draws: int, seed: int) -> dict:
    events = sorted(cells)
    point = {m: scores(cells, events, m) for m in MODELS}
    rng = random.Random(seed)
    diffs = {m: defaultdict(list) for m in MODELS if m != "shared"}
    for _ in range(draws):
        sample = [rng.choice(events) for _ in events]
        base = scores(cells, sample, "shared")
        for m in diffs:
            alt = scores(cells, sample, m)
            for key, value in alt.items():
                diffs[m][key].append(value - base[key])

    def interval(xs: list[float]) -> list[float]:
        xs = sorted(xs)
        return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)]

    def abs_event_error(event: str, model: str) -> float:
        segs = cells[event][model]["segments"].values()
        return abs(math.log(sum(m for _, m in segs) / sum(a for a, _ in segs)))

    return {
        "events": events,
        "scores": {m: {k: round(v, 4) for k, v in s.items()} for m, s in point.items()},
        "segmentEventLogRms": {
            m: {
                s: round(math.sqrt(sum(
                    math.log(cells[e][m]["segments"][s][1] / cells[e][m]["segments"][s][0]) ** 2
                    for e in events
                    if cells[e][m]["segments"][s][0] > 0 and cells[e][m]["segments"][s][1] > 0
                ) / len(events)), 4)
                for s in LOSS_SEGMENTS
            }
            for m in MODELS
        },
        "eventWinsVsShared": {
            m: sum(abs_event_error(e, m) < abs_event_error(e, "shared") for e in events)
            for m in diffs
        },
        "bootstrapDiffVsShared95": {
            m: {k: interval(v) for k, v in by_key.items()} for m, by_key in diffs.items()
        },
        "bootstrap": {"draws": draws, "seed": seed, "unit": "event"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--cells", type=Path, help="reuse fold cells written by an earlier --out run")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.cells:
        cells = json.loads(args.cells.read_text(encoding="utf-8"))["cells"]
    else:
        cells = holdout_cells(observations())
    report = summarise(cells, args.draws, args.seed)
    if args.out:
        args.out.write_text(json.dumps({**report, "cells": cells}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
