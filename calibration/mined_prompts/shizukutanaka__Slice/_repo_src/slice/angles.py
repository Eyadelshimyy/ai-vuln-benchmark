"""Joint angles — anatomically meaningful measurements, in degrees.

Flexion is the interior angle at the middle joint of a bone pair
(180° = perfectly straight). Elevation measures a limb's direction
relative to the torso vertical (0° = hanging down, 90° = horizontal,
180° = overhead). Missing joints omit the entry rather than guess —
and predicted joints count as missing: a prior-placed limb reads
~180° by construction, which would fabricate a measurement.
"""

from __future__ import annotations

import math
from typing import Optional

from .skeleton import Skeleton

_FLEX = [
    ("elbow_l_flex", "shoulder_l", "elbow_l", "wrist_l"),
    ("elbow_r_flex", "shoulder_r", "elbow_r", "wrist_r"),
    ("knee_l_flex", "hip_l", "knee_l", "ankle_l"),
    ("knee_r_flex", "hip_r", "knee_r", "ankle_r"),
    ("shoulder_l_flex", "neck", "shoulder_l", "elbow_l"),
    ("shoulder_r_flex", "neck", "shoulder_r", "elbow_r"),
]


def _angle(a, b, c) -> Optional[float]:
    """Interior angle at vertex b, degrees."""
    if not (a and b and c):
        return None
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1])
                        / (n1 * n2)))
    return math.degrees(math.acos(cos))


def _obs_point(skel: Skeleton, name: str):
    j = skel.joints.get(name)
    return (j.x, j.y) if j and j.state == "observed" else None


def _elevation(skel: Skeleton, a: str, b: str) -> Optional[float]:
    """Direction of a→b measured from straight-down, degrees."""
    pa, pb = _obs_point(skel, a), _obs_point(skel, b)
    if not pa or not pb:
        return None
    # y grows downward in image space: down is (0, +1)
    v = (pb[0] - pa[0], pb[1] - pa[1])
    n = math.hypot(*v)
    if n < 1e-6:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, v[1] / n))))


def analyze(skel: Skeleton) -> dict:
    out: dict = {}
    for name, a, b, c in _FLEX:
        v = _angle(_obs_point(skel, a), _obs_point(skel, b),
                   _obs_point(skel, c))
        if v is not None:
            out[name] = round(v, 1)
    for side in ("l", "r"):
        v = _elevation(skel, f"shoulder_{side}", f"elbow_{side}")
        if v is not None:
            out[f"arm_{side}_elevation"] = round(v, 1)
        v = _elevation(skel, f"hip_{side}", f"knee_{side}")
        if v is not None:
            out[f"leg_{side}_elevation"] = round(v, 1)
    # Torso lean: angle of neck→pelvis from straight-down.
    v = _elevation(skel, "neck", "pelvis")
    if v is not None:
        out["torso_lean"] = round(v, 1)
    return out
