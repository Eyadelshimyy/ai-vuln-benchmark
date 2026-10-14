"""Reproducibility check — does the same input still give this output?

A knowledge document claims "this image, this engine, these joints".
Nothing verifies the claim is stable: library changes, threshold
tweaks, or silent nondeterminism could make a re-run produce a
different skeleton without anyone noticing. `verify` re-estimates
from the recorded source image and diffs the result against the
stored skeleton, joint by joint.

What it catches:
- `position_drift` — recorded vs re-estimated position beyond
  `TOLERANCE_PX` (engine changed)
- `state_flip` — observed↔predicted flip (evidence reclassified)
- `missing` — joint present in doc, absent on re-run
- `added` — joint absent in doc, produced on re-run

verdict: `reproducible` (all joints within tolerance),
`drifted` (positions moved but no structural change),
`changed` (flips/misses/adds). Positions are compared in the
*recorded* frame — re-estimated joints are rescaled when the
downscale differs, so resolution changes alone don't count as
drift.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .anatomy import DEFAULT_MODEL
from .bitmap import Bitmap
from .pose import HeuristicPoseEstimator

TOLERANCE_PX = 2.0


def verify(doc: dict, bmp: Bitmap,
           estimator: Optional[HeuristicPoseEstimator] = None,
           tolerance: float = TOLERANCE_PX) -> Dict:
    """Re-estimate `bmp` and diff against `doc`'s recorded skeleton.

    The document's frame size and joint list come from
    `doc["skeleton"]`; the model from `doc["engine"]`-adjacent fields
    is not part of schema v1, so `model` falls back to the estimator
    default — disclosed in `basis`. The engine *profile* the doc
    recorded (`engine.profile`, e.g. "robust") is replayed too —
    re-estimating a robust doc under the default profile reports
    configuration drift, not engine drift.
    """
    rec = (doc.get("skeleton") or {}).get("joints") or {}
    frame = (doc.get("skeleton") or {}).get("frame") or {}
    model = ((doc.get("skeleton") or {}).get("body_model") or {}) \
        .get("name") or DEFAULT_MODEL

    profile = (doc.get("engine") or {}).get("profile") or "default"
    est = estimator or _estimator_for(profile)
    sk = est.estimate(bmp, model)
    sx = frame.get("width") and sk.image_width / frame["width"] or 1.0
    sy = frame.get("height") and sk.image_height / frame["height"] or 1.0

    drifts, flips, missing, added = [], [], [], []
    compared = 0
    for name, rj in rec.items():
        ej = sk.get(name)
        if ej is None:
            missing.append(name)
            continue
        ex, ey = ej.x / (sx or 1.0), ej.y / (sy or 1.0)
        d = math.hypot(ex - rj["x"], ey - rj["y"])
        compared += 1
        if ej.state != rj.get("state"):
            flips.append({"joint": name,
                          "was": rj.get("state"), "now": ej.state})
        if d > tolerance:
            drifts.append({"joint": name,
                           "drift_px": round(d, 2)})
    for name in sk.joints:
        if name not in rec:
            added.append(name)

    verdict = ("changed" if flips or missing or added else
               "drifted" if drifts else
               "reproducible")
    return {
        "verdict": verdict,
        "joints_compared": compared,
        "drifts": drifts,
        "state_flips": flips,
        "missing": sorted(missing),
        "added": sorted(added),
        "tolerance_px": tolerance,
        "engine_profile": profile,
        "state": "measured",
        "basis": "re-estimate on source image vs recorded joints "
                 "(frame-rescaled, profile=%s)" % profile,
    }


def _estimator_for(profile: str) -> HeuristicPoseEstimator:
    """Estimator matching a doc's recorded engine profile."""
    if profile == "robust":
        return HeuristicPoseEstimator(adaptive=True, reject_shadow=True,
                                      clean=True)
    return HeuristicPoseEstimator()


def reproducible(result: Dict) -> bool:
    return result["verdict"] == "reproducible"
