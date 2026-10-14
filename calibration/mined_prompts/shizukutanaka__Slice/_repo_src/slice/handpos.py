"""Hand position semantics — where each hand sits on the body map.

Above the head, at the face, by the chest, at the waist, near the
knee, hanging below, or hidden behind the body — a hand's zone
says more about intent than its raw coordinates. Zones are defined
relative to observed landmarks (head top, chest, pelvis, knee) so
they scale with the body, and predicted wrists are never zoned —
nor is a wrist zoned against predicted landmarks, which would
fabricate the zone's boundaries from a prior guess.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .skeleton import Skeleton


def _ref(skel: Skeleton, name: str) -> Optional[Tuple[float, float]]:
    j = skel.joints.get(name)
    return (j.x, j.y) if j and j.state == "observed" else None


def _zone(skel: Skeleton, side: str) -> Optional[str]:
    j = skel.joints.get(f"wrist_{side}")
    if not j or j.state != "observed":
        return None
    w = (j.x, j.y)
    head = _ref(skel, "head")
    chest = _ref(skel, "chest")
    pelvis = _ref(skel, "pelvis")
    knee = _ref(skel, f"knee_{side}")
    hip = _ref(skel, f"hip_{side}")
    if not any((head, chest, pelvis, knee, hip)):
        return None

    def near(p, frac_of: float, ref: float) -> bool:
        return abs(w[1] - p[1]) < frac_of * ref

    span = 0.0
    if head and pelvis:
        span = abs(pelvis[1] - head[1])
    if head and w[1] < head[1] - 0.15 * max(span, 1.0):
        return "above_head"
    if head and near(head, 0.30, max(span, 1.0)):
        return "at_head"
    if chest and near(chest, 0.35, max(span, 1.0)):
        return "at_chest"
    if pelvis and near(pelvis, 0.30, max(span, 1.0)):
        return "at_waist"
    if hip and near(hip, 0.25, max(span, 1.0)):
        return "at_hip"
    if knee and abs(w[1] - knee[1]) < 0.2 * max(span, 1.0):
        return "at_knee"
    return "hanging"


def positions(skel: Skeleton) -> Dict[str, Optional[str]]:
    """{l: zone|None, r: zone|None} — None = wrist or the landmarks it would be zoned against are not observed."""
    return {side: _zone(skel, side) for side in ("l", "r")}


def summary(skel: Skeleton) -> dict:
    """{zones, symmetric, observed_hands} for knowledge output."""
    p = positions(skel)
    zones = {s: z for s, z in p.items() if z}
    obs = len(zones)
    sym = obs == 2 and len(set(zones.values())) == 1
    return {"zones": zones, "observed_hands": obs,
            "symmetric": sym}
