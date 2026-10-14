"""Bilateral pose symmetry — mirrored bone-direction agreement.

For each left/right bone pair, mirror the left bone's direction
across the vertical axis and compare with the right bone's actual
direction. Perfect mirror = 1.0, opposite = -1.0 (clamped to 0 for
the score). A standing T-pose scores ~1; a person waving one arm
scores low on the arm chain but still symmetric in the legs. Pairs
missing a side are reported, not guessed.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .skeleton import Skeleton

_PAIRS = (
    (("neck", "shoulder_l"), ("neck", "shoulder_r")),
    (("shoulder_l", "elbow_l"), ("shoulder_r", "elbow_r")),
    (("elbow_l", "wrist_l"), ("elbow_r", "wrist_r")),
    (("pelvis", "hip_l"), ("pelvis", "hip_r")),
    (("hip_l", "knee_l"), ("hip_r", "knee_r")),
    (("knee_l", "ankle_l"), ("knee_r", "ankle_r")),
    (("ankle_l", "foot_l"), ("ankle_r", "foot_r")),
)


def _dir(skel: Skeleton, a: str, b: str) -> Optional[Tuple[float, float]]:
    """Bone direction — only when BOTH endpoints are observed.

    Predicted joints are filled by mirroring the observed side, so a
    pair containing them would always report perfect symmetry: a
    fabricated measurement, not a comparison."""
    ja, jb = skel.joints.get(a), skel.joints.get(b)
    if not ja or not jb:
        return None
    if ja.state != "observed" or jb.state != "observed":
        return None
    dx, dy = jb.x - ja.x, jb.y - ja.y
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else None


def score(skel: Skeleton) -> dict:
    """{score 0-1, pairs {name: mirrored-dot}, compared, missing}"""
    pairs: Dict[str, float] = {}
    missing = []
    dots = []
    for lb, rb in _PAIRS:
        name = f"{lb[1][:-2]}_pair"
        dl, dr = _dir(skel, *lb), _dir(skel, *rb)
        if dl is None or dr is None:
            missing.append(name)
            continue
        # mirror left across vertical axis, then dot with right
        dot = -dl[0] * dr[0] + dl[1] * dr[1]
        pairs[name] = round(max(0.0, dot), 3)
        dots.append(pairs[name])
    mean = round(sum(dots) / len(dots), 3) if dots else None
    return {"score": mean, "pairs": pairs,
            "compared": len(dots), "missing": missing}


def asymmetric_side(skel: Skeleton) -> Optional[dict]:
    """Which side pair diverges most from the mirror, if any."""
    r = score(skel)
    if not r["pairs"]:
        return None
    worst = min(r["pairs"].items(), key=lambda kv: kv[1])
    return {"pair": worst[0], "score": worst[1],
            "symmetric": worst[1] >= 0.8}
