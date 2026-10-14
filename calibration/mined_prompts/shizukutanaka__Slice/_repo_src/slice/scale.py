"""Real-world scale — pixels to centimetres via the head-size prior.

Pixel measurements are resolution-relative; anatomical knowledge
wants real units. An adult head is ~22–25 cm tall regardless of
image size, so measuring the head in pixels yields a px→cm factor
and lets every length (height, limbs, shoulder width) be reported
in centimetres. Every value is tagged `estimated` — the scale is a
prior-based inference, not a measurement of the world.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .anatomy import BODY_MODELS, DEFAULT_MODEL
from .skeleton import NECK_CLAVICLE_BASIS, Skeleton, head_length_px

# Mean adult head length (vertex→chin), anthropometric standard.
HEAD_CM = {"adult": 23.0, "child": 19.0, "deformed": 20.0}

_BONES_CM = [
    ("height", "head", "foot_l"),
    ("torso", "neck", "pelvis"),
    ("upper_arm_l", "shoulder_l", "elbow_l"),
    ("thigh_l", "hip_l", "knee_l"),
    ("shin_l", "knee_l", "ankle_l"),
    ("shoulder_width", "shoulder_l", "shoulder_r"),
    ("hip_width", "hip_l", "hip_r"),
]


def _obs_point(skel: Skeleton, name: str):
    """Observed-only lookup: a predicted joint is prior fill — a
    scale factor or cm length derived from it would measure the
    prior, not the person, so it is excluded like a missing joint."""
    j = skel.joints.get(name)
    if j is None or j.state != "observed":
        return None
    return (j.x, j.y)


def _head_px(skel: Skeleton) -> Optional[float]:
    """Head length in px — convention-aware (clavicle vs chin neck),
    observed joints only: a predicted head/neck is prior fill, and
    measuring it would measure the prior, not the person."""
    head, neck = _obs_point(skel, "head"), _obs_point(skel, "neck")
    if not head or not neck or neck[1] <= head[1]:
        return None
    neck_j = skel.joints.get("neck")
    mult = 1.0 if neck_j.basis == NECK_CLAVICLE_BASIS else 2.0
    return (neck[1] - head[1]) * mult


def calibrate(skel: Skeleton, model: Optional[str] = None) -> dict:
    """px→cm scale factor, or {scale: None} when it can't be estimated."""
    name = model or (skel.body_model or {}).get("name") or DEFAULT_MODEL
    prior = BODY_MODELS.get(name, BODY_MODELS[DEFAULT_MODEL])
    head_cm = HEAD_CM.get(name, HEAD_CM["adult"])
    head_px = _head_px(skel)
    if not head_px or head_px < 4:
        return {"px_per_cm": None, "basis": "head_prior",
                "state": "estimated", "head_cm_assumed": head_cm}
    return {"px_per_cm": head_px / head_cm, "basis": "head_prior",
            "state": "estimated", "head_cm_assumed": head_cm}


def measure_cm(skel: Skeleton, model: Optional[str] = None) -> Dict:
    """Bone/torso/height lengths in centimetres (estimated)."""
    cal = calibrate(skel, model)
    ppcm = cal.get("px_per_cm")
    out: Dict = {"scale": cal}
    if not ppcm:
        out["lengths"] = {}
        return out
    lengths = {}
    for label, a, b in _BONES_CM:
        pa, pb = _obs_point(skel, a), _obs_point(skel, b)
        if pa and pb:
            px = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
            lengths[label] = round(px / ppcm, 1)
    out["lengths"] = lengths
    return out
