"""Perturbation stability — how much would this joint move?

`calib` asks "how accurate is a given confidence?" — this module
asks the dual question: "how sensitive is this joint to the
settings that produced it?" A shoulder that lands at the same
pixel under `bg_threshold` 30, 40, and 50 is genuinely supported
by the image; a wrist that jumps 40px when the gate wiggles was
barely detectable and should not be trusted.

`probe` re-runs the estimator with its *effective* threshold
shifted by `±DELTA` (the shift reaches the Otsu split in adaptive
mode, where moving `bg_threshold` alone would perturb nothing) and
reports, per joint, the maximum displacement across the perturbed
runs — a measured sensitivity, not a heuristic.
Joints absent in the baseline estimate are skipped (predicted
positions inherit their parents' stability, which this module does
not try to model).

Per-joint verdict: `stable` (< STABLE_PX), `sensitive`
(< UNSTABLE_PX), `unstable` otherwise. `unmeasurable` when the
joint was not observed at baseline.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .anatomy import DEFAULT_MODEL
from .bitmap import Bitmap
from .pose import HeuristicPoseEstimator
from .skeleton import OBSERVED

DELTA = 10            # probe at threshold ±DELTA
STABLE_PX = 3.0
UNSTABLE_PX = 15.0


def _label(d: float) -> str:
    return ("stable" if d < STABLE_PX else
            "sensitive" if d < UNSTABLE_PX else "unstable")


def probe(bmp: Bitmap,
          estimator: Optional[HeuristicPoseEstimator] = None,
          model: str = DEFAULT_MODEL,
          delta: int = DELTA) -> Dict:
    """Re-estimate at threshold±delta; report per-joint displacement.

    Returns {joints: {name: {max_displacement_px, verdict, runs}},
    stable_fraction, summary}. A joint measured in only one run gets
    `verdict: "single_run"` — its stability is unknown, not assumed.
    """
    est = estimator or HeuristicPoseEstimator()
    base = est.estimate(bmp, model)
    variants = []
    for shift in (-delta, delta):
        v = HeuristicPoseEstimator(
            max_dim=est.max_dim, bg_threshold=est.bg_threshold,
            adaptive=est.adaptive, reject_shadow=est.reject_shadow,
            clean=est.clean,
            threshold_offset=est.threshold_offset + shift)
        variants.append(v.estimate(bmp, model))

    joints: Dict[str, Dict] = {}
    for name, j in base.joints.items():
        if j.state != OBSERVED:
            continue
        positions = [(j.x, j.y)]
        for v in variants:
            vj = v.get(name)
            if vj and vj.state == OBSERVED:
                positions.append((vj.x, vj.y))
        if len(positions) == 1:
            joints[name] = {"verdict": "single_run",
                            "max_displacement_px": None,
                            "runs": 1}
            continue
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        dmax = max(math.hypot(x - cx, y - cy) for x, y in positions)
        joints[name] = {"verdict": _label(dmax),
                        "max_displacement_px": round(dmax, 2),
                        "runs": len(positions)}

    measured = [v for v in joints.values() if v["verdict"] != "single_run"]
    stable = sum(1 for v in measured if v["verdict"] == "stable")
    return {
        "joints": joints,
        "stable_fraction": round(stable / len(measured), 3)
        if measured else None,
        "state": "measured" if measured else "unmeasurable",
        "basis": "joint displacement across bg_threshold ±%d probes"
                 % delta,
        "probe_delta": delta,
    }


def unstable(result: Dict) -> list:
    """Names of joints that moved more than UNSTABLE_PX under probe."""
    return sorted(n for n, v in result["joints"].items()
                  if v["verdict"] == "unstable")
