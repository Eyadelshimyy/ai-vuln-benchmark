"""Static balance — center-of-mass estimate vs support polygon.

Anthropometric priors (Winter's segment masses) put ~2/3 of body mass
in head+torso. A person is statically stable only while the vertical
projection of their center of mass falls inside the support polygon —
the convex hull of the feet. One observed foot → point polygon;
two → line segment; missing feet → the honest answer is
"cannot determine", not a guess.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .skeleton import Skeleton, observed_point

# segment name -> (mass fraction, joint(s) locating the segment's
# approximate center). Winter 2009 proportions, both sides pooled.
_SEGMENTS = (
    ("head_torso", 0.60, ("chest",)),
    ("arm_l", 0.05, ("shoulder_l", "elbow_l", "wrist_l")),
    ("arm_r", 0.05, ("shoulder_r", "elbow_r", "wrist_r")),
    ("leg_l", 0.15, ("hip_l", "knee_l", "ankle_l")),
    ("leg_r", 0.15, ("hip_r", "knee_r", "ankle_r")),
)


def _centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def center_of_mass(skel: Skeleton) -> Optional[dict]:
    """{x, y, mass_covered, state} — None if fewer than 30% of the
    mass can be located. Estimated, not measured."""
    mx = my = covered = 0.0
    for _, frac, names in _SEGMENTS:
        pts = [observed_point(skel, n) for n in names]
        pts = [p for p in pts if p]
        if not pts:
            continue
        cx, cy = _centroid(pts)
        mx += cx * frac
        my += cy * frac
        covered += frac
    if covered < 0.30:
        return None
    return {"x": round(mx / covered, 1), "y": round(my / covered, 1),
            "mass_covered": round(covered, 2), "state": "estimated"}



def assess(skel: Skeleton) -> dict:
    """{com, support, projected, state} — projected inside the
    support region = stable. Margins are horizontal distance to the
    support boundary."""
    com = center_of_mass(skel)
    feet = [observed_point(skel, "foot_l"), observed_point(skel, "foot_r")]
    feet = [f for f in feet if f]
    out = {"com": com, "support_joints": len(feet),
           "state": "estimated"}

    if com is None:
        out.update(projected="unknown", margin=None,
                   reason="insufficient_mass_covered")
        return out
    if not feet:
        out.update(projected="unknown", margin=None,
                   reason="no_feet")
        return out

    # horizontal margin: how far the COM x sits inside/outside the
    # lateral extent of the feet (sagittal balance, 2D projection)
    xs = sorted(f[0] for f in feet)
    lo, hi = xs[0], xs[-1]
    if len(feet) == 1:
        margin = com["x"] - lo  # point support: distance from foot
        projected = "marginal" if abs(margin) <= com_span(skel) else "outside"
    else:
        inside = lo <= com["x"] <= hi
        margin = min(com["x"] - lo, hi - com["x"])
        projected = "inside" if inside else "outside"
    out.update(projected=projected,
               margin=round(margin, 1),
               reason=None)
    return out


def com_span(skel: Skeleton) -> float:
    """Typical tolerable offset: half the hip width when known."""
    hl, hr = observed_point(skel, "hip_l"), observed_point(skel, "hip_r")
    if hl and hr:
        return abs(hr[0] - hl[0]) / 2.0
    return 20.0
