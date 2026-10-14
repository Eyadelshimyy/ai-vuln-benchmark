"""Functional reach — what each arm can physically touch.

An arm's workspace is a disc around the shoulder with radius
upper_arm + forearm (+ a hand-length margin). A target inside 85%
of that radius is comfortably reachable, inside 100% is at full
extension, beyond is out of reach without stepping. Missing arm
bones fall back to the body-height prior (~0.4 × height) — marked
estimated, not measured.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from .skeleton import Skeleton, observed_body_span, observed_point

Point = Tuple[float, float]

_HAND_FRAC = 0.08   # hand ≈ 8% of arm reach beyond the wrist
_PRIOR_FRAC = 0.40  # arm span ≈ 40% of body height


def _d(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def arm_reach(skel: Skeleton, side: str) -> Optional[dict]:
    """{shoulder, radius, measured} — radius in px."""
    s = observed_point(skel, f"shoulder_{side}")
    e = observed_point(skel, f"elbow_{side}")
    w = observed_point(skel, f"wrist_{side}")
    if s is None:
        return None
    if e and w:
        r = (_d(s, e) + _d(e, w)) * (1.0 + _HAND_FRAC)
        return {"shoulder": s, "radius": round(r, 1),
                "measured": True}
    # no arm bones: fall back to the body-height prior — but only
    # when a body scale exists; a degenerate (inverted/empty)
    # skeleton has no honest reach radius
    span = observed_body_span(skel)
    if span > 0:
        r = span * _PRIOR_FRAC
        return {"shoulder": s, "radius": round(r, 1),
                "measured": False}
    return None


def reachable(skel: Skeleton, side: str,
              target: Point) -> Optional[dict]:
    """{within, margin, state} — margin>0 means the wrist can reach
    the target with slack."""
    env = arm_reach(skel, side)
    if env is None:
        return None
    d = _d(env["shoulder"], target)
    margin = env["radius"] - d
    within = "inside" if d <= 0.85 * env["radius"] else (
        "edge" if d <= env["radius"] else "outside")
    return {"within": within, "margin": round(margin, 1),
            "distance": round(d, 1),
            "state": "measured" if env["measured"] else "estimated"}


def workspace(skel: Skeleton) -> dict:
    """{side: {radius, measured}} — both arms."""
    out = {}
    for side in ("l", "r"):
        env = arm_reach(skel, side)
        if env:
            out[side] = {"radius": env["radius"],
                         "measured": env["measured"],
                         "shoulder": env["shoulder"]}
    return out
