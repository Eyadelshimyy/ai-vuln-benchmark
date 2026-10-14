"""Limb length profile — per-limb chain totals.

Anthropometry reports limb length as the sum of its segments
(shoulder→wrist = upper arm + forearm + hand reach). Here each
limb reports px length, body-height fraction, and whether it was
fully measured or partially predicted — a left-vs-right delta
flags pose foreshortening, not body asymmetry.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .skeleton import Skeleton

Point = Tuple[float, float]

_CHAINS = {
    "arm": ("shoulder", "elbow", "wrist"),
    "leg": ("hip", "knee", "ankle", "foot"),
}


def _d(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def limb(skel: Skeleton, kind: str, side: str) -> Optional[dict]:
    """{length_px, of_body_h, segments, observed, partial}."""
    chain = _CHAINS[kind]
    names = [f"{n}_{side}" for n in chain]
    pts, states = [], []
    for n in names:
        j = skel.joints.get(n)
        if j is None:
            continue
        pts.append((j.x, j.y))
        states.append(j.state)
    if len(pts) < 2:
        return None

    # the normalising body span must come from observed joints only:
    # a predicted head or foot sits at a prior guess, so a "measured"
    # fraction divided by it inherits the guess
    obs_pts = [j.y for j in skel.joints.values() if j.state == "observed"]
    body_h = (max(obs_pts) - min(obs_pts)) if obs_pts else 0.0
    length = sum(_d(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    obs = sum(1 for s in states if s == "observed")
    # a missing chain joint joins its neighbours into a chord — the
    # summed length then under-measures the limb, so the profile is
    # partial even when every present joint is observed
    missing = len(pts) < len(names)
    return {"length_px": round(length, 1),
            "of_body_h": round(length / body_h, 3)
            if body_h > 0 else None,
            "segments": len(pts) - 1,
            "observed": obs,
            "partial": missing or obs < len(states)}


def profile(skel: Skeleton) -> Dict[str, Optional[dict]]:
    """{arm_l, arm_r, leg_l, leg_r, delta} — delta = |L-R| px
    asymmetry per kind, only when both sides are fully measured."""
    out: Dict[str, Optional[dict]] = {}
    for kind in _CHAINS:
        for side in ("l", "r"):
            out[f"{kind}_{side}"] = limb(skel, kind, side)
    delta = {}
    for kind in _CHAINS:
        l, r = out[f"{kind}_l"], out[f"{kind}_r"]
        # compare only fully measured limbs — a chord across a
        # missing or predicted joint under-measures its limb, so
        # |partial - measured| would report a fabricated asymmetry
        if l and r and not (l["partial"] or r["partial"]):
            delta[kind] = round(abs(l["length_px"] - r["length_px"]), 1)
    out["delta"] = delta
    return out
