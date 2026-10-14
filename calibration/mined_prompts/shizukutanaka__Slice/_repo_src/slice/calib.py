"""Confidence calibration — turn heuristic scores into measured ones.

Every joint carries `confidence` between 0 and 1, but the value is a
hand-set heuristic (0.8 for a widest-row shoulder, 0.55 for a
fallback hip) — *how often is a 0.8 joint actually right?* Nothing
in the document answers that, so downstream consumers can't rank a
0.75 wrist against a 0.75 shoulder honestly.

`reliability_table` answers it empirically: run the estimator over
ground-truth fixture pairs, bin joints by their reported
confidence, and measure the hit rate inside each bin — the standard
reliability diagram from calibration theory. `lookup` then maps an
arbitrary confidence to the empirical accuracy its bin showed, and
`apply` annotates a skeleton with those calibrated values — as a
side record, never overwriting the raw score.

Report-only by design: the raw estimator output stays untouched;
the calibration is disclosed next to it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .pose import HeuristicPoseEstimator
from .skeleton import OBSERVED, Skeleton

BINS = 10
# a joint counts as "hit" when its estimate is this close to truth
HIT_RADIUS_PX = 8.0


def reliability_table(pairs, estimator: Optional[HeuristicPoseEstimator] = None,
                      *, bins: int = BINS,
                      hit_radius: float = HIT_RADIUS_PX) -> List[Dict]:
    """Bin observed joints by confidence; measure hit rate per bin.

    `pairs` is an iterable of (bitmap, truth) as produced by
    `evaluate.draw_case`. Returns a list of
    {lo, hi, n, mean_conf, accuracy, mean_error} — bins with no
    samples are kept with n=0 so the table stays indexable.
    """
    est = estimator or HeuristicPoseEstimator()
    width = 1.0 / bins
    cells = [[] for _ in range(bins)]
    errs = [[] for _ in range(bins)]
    for bmp, truth in pairs:
        skel = est.estimate(bmp)
        for name, (tx, ty) in truth.items():
            j = skel.get(name)
            if j is None or j.state != OBSERVED:
                continue
            # multiply like `lookup` does — `conf / 0.1` loses boundary
            # scores to float error (0.6 / 0.1 == 5.999...)
            i = min(bins - 1, int(j.confidence * bins))
            cells[i].append(j.confidence)
            errs[i].append(math.hypot(j.x - tx, j.y - ty))
    table = []
    for i in range(bins):
        n = len(cells[i])
        hits = sum(1 for e in errs[i] if e <= hit_radius)
        table.append({
            "lo": round(i * width, 2), "hi": round((i + 1) * width, 2),
            "n": n,
            "mean_conf": round(sum(cells[i]) / n, 3) if n else None,
            "accuracy": round(hits / n, 3) if n else None,
            "mean_error": round(sum(errs[i]) / n, 2) if n else None,
        })
    return table


def report(pairs, estimator=None, **kw) -> Dict:
    """Table + headline calibration stats."""
    table = reliability_table(pairs, estimator, **kw)
    filled = [t for t in table if t["n"]]
    overconf = underconf = 0
    for t in filled:
        gap = t["mean_conf"] - t["accuracy"]
        if gap > 0.10:
            overconf += 1
        elif gap < -0.10:
            underconf += 1
    return {
        # an all-empty table measured nothing — do not claim estimated
        "state": "estimated" if filled else "unmeasured",
        "basis": "ground-truth fixture error rates per confidence bin",
        "hit_radius_px": kw.get("hit_radius", HIT_RADIUS_PX),
        "bins": table,
        "overconfident_bins": overconf,
        "underconfident_bins": underconf,
        "note": "accuracy = fraction of joints within hit_radius_px "
                "of ground truth; empty bins stay n=0 for indexing",
    }


def lookup(confidence: float, table: List[Dict]) -> Optional[float]:
    """Empirical accuracy for a confidence value, or None if its bin
    had no samples (no evidence either way)."""
    i = min(len(table) - 1, int(confidence * len(table)))
    return table[i]["accuracy"]


def apply(skel: Skeleton, table: List[Dict]) -> Dict[str, float]:
    """Annotate each observed joint with its bin's empirical accuracy.

    Returns {joint_name: calibrated_accuracy}. The joint's own
    confidence field is NOT rewritten — the raw heuristic stays
    inspectable next to the calibrated value.
    """
    out: Dict[str, float] = {}
    for name, j in skel.joints.items():
        if j.state != OBSERVED:
            continue
        acc = lookup(j.confidence, table)
        if acc is not None:
            out[name] = acc
    return out
