"""Body principal axis — PCA on the joint cloud.

Pose labels bin the body into categories (standing, lying); the
principal axis is the continuous version: the direction of maximum
spread across all joints, plus anisotropy (long-axis variance vs
short-axis). Vertical axis + high anisotropy = upright elongated
figure; horizontal = lying; round blob = compressed/crouched.
2x2 eigensolve is closed-form — no numpy needed.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from .skeleton import Skeleton


def _eig2x2(sxx: float, syy: float, sxy: float) -> Tuple[float, float, float]:
    """Return (lambda1, lambda2, angle_of_major_axis_radians)."""
    tr = sxx + syy
    d = math.hypot(sxx - syy, 2.0 * sxy)
    l1 = (tr + d) / 2.0
    l2 = (tr - d) / 2.0
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return l1, l2, angle


def principal(skel: Skeleton) -> Optional[dict]:
    """{angle_deg, anisotropy, spread, centroid} — angle 90°=vertical
    image axis (upright), 0°=horizontal (lying).

    PCA runs on observed joints only: predicted positions are
    fabricated geometry and would pull the "measured" axis toward
    the prior's shape."""
    pts = [(j.x, j.y) for j in skel.joints.values()
           if j.state == "observed"]
    n = len(pts)
    if n < 3:
        return None
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - cx) ** 2 for p in pts) / n
    syy = sum((p[1] - cy) ** 2 for p in pts) / n
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in pts) / n
    l1, l2, ang = _eig2x2(sxx, syy, sxy)
    # angle of major axis; 90 = vertical
    deg = math.degrees(ang)
    if deg < 0:
        deg += 180
    aniso = (l1 - l2) / l1 if l1 > 1e-9 else 0.0
    return {"angle_deg": round(deg, 1),
            "anisotropy": round(aniso, 3),
            "spread": round(math.sqrt(l1), 1),
            "centroid": [round(cx, 1), round(cy, 1)]}


def tilt(skel: Skeleton) -> Optional[float]:
    """Angle away from vertical in degrees (0 = upright)."""
    r = principal(skel)
    if not r:
        return None
    return round(abs(90.0 - r["angle_deg"]), 1)
