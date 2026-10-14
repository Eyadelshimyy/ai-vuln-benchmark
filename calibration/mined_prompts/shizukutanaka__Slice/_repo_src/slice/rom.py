"""Range-of-motion check — joints vs anatomical limits.

Every joint has a physiological range: elbow 0–160°, knee 0–150°,
shoulder flexion to ~180°. A measured angle past the limit isn't
"flexible", it's evidence the skeleton itself is wrong — this
audit flags overextension as a consistency problem, and borderline
angles as hypermobile-but-possible.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .skeleton import Skeleton

Point = Tuple[float, float]

# (proximal, vertex, distal, min_deg, max_deg) — interior angle
_JOINTS = (
    ("shoulder_l", "elbow_l", "wrist_l", 10.0, 180.0),
    ("shoulder_r", "elbow_r", "wrist_r", 10.0, 180.0),
    ("hip_l", "knee_l", "ankle_l", 15.0, 180.0),
    ("hip_r", "knee_r", "ankle_r", 15.0, 180.0),
    ("knee_l", "ankle_l", "foot_l", 30.0, 175.0),
    ("knee_r", "ankle_r", "foot_r", 30.0, 175.0),
    ("neck", "shoulder_l", "elbow_l", 20.0, 175.0),
    ("neck", "shoulder_r", "elbow_r", 20.0, 175.0),
)


def _angle(a: Point, b: Point, c: Point) -> Optional[float]:
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1])
                        / (n1 * n2)))
    return math.degrees(math.acos(cos))


def check(skel: Skeleton) -> Dict[str, dict]:
    """{vertex: {angle, limits, verdict, measured}} — verdict:
    compliant|hypermobile|overextended."""
    out: Dict[str, dict] = {}
    for a, b, c, lo, hi in _JOINTS:
        pa, pb, pc = skel.point(a), skel.point(b), skel.point(c)
        if not (pa and pb and pc):
            continue
        # A ~1px segment can't define a direction (foot placed one
        # pixel below the ankle tip); flagging an angle measured
        # off pixel noise would be fabricating a finding.
        if math.hypot(pc[0] - pb[0], pc[1] - pb[1]) < 3.0:
            continue
        ang = _angle(pa, pb, pc)
        if ang is None:
            continue
        measured = all(skel.joints[n].state == "observed"
                       for n in (a, b, c))
        if ang < lo:
            verdict = "overextended" if measured else "implausible"
        elif ang > hi:
            verdict = "overextended" if measured else "implausible"
        elif ang < lo + 8 or ang > hi - 8:
            verdict = "hypermobile"
        else:
            verdict = "compliant"
        out[b] = {"angle": round(ang, 1), "limits": [lo, hi],
                  "verdict": verdict, "measured": measured}
    return out


def violations(skel: Skeleton) -> list:
    """Only the implausible/overextended ones."""
    return [{"joint": k, **v} for k, v in check(skel).items()
            if v["verdict"] in ("overextended", "implausible")]
