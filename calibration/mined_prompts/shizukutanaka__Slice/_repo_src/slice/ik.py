"""Two-bone inverse kinematics — analytic elbow/knee placement.

Given a chain root (shoulder/hip), a target (wrist/ankle position),
and the two segment lengths, the mid joint sits at the intersection
of two circles. There are 0, 1, or 2 solutions; `bend` picks which
side of the chain the mid joint lands on. Unreachable targets clamp
to full extension — the honest answer is "bent as far as anatomy
allows", not "anywhere you asked".
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

Point = Tuple[float, float]


def solve_ik(root: Point, target: Point, L1: float, L2: float,
             bend: float = 1.0) -> Optional[Tuple[Point, Point]]:
    """Return (mid, end). mid = elbow/knee position, end = reached
    position (clamped to L1+L2 if the target is unreachable, or to
    |L1-L2| if it is too close). None if segments are degenerate."""
    if L1 <= 0 or L2 <= 0:
        # a zero/negative-length bone cannot bend — no chain exists
        # to solve, so there is no honest mid point to return
        return None
    dx, dy = target[0] - root[0], target[1] - root[1]
    d = math.hypot(dx, dy)
    if d < 1e-9:
        d = 1e-9
        dx, dy = L1 + L2, 0.0  # aim along +x when target sits on root
    d_max, d_min = L1 + L2, abs(L1 - L2)
    d = max(d_min + 1e-6, min(d_max - 1e-6, d))
    ux, uy = dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)
    end = (root[0] + ux * d, root[1] + uy * d)

    # circle intersection: mid is `a` along the axis, `h` perpendicular
    a = (L1 * L1 - L2 * L2 + d * d) / (2 * d)
    h2 = L1 * L1 - a * a
    h = math.sqrt(h2) if h2 > 0 else 0.0
    px, py = -uy * bend, ux * bend  # perpendicular unit
    mid = (root[0] + ux * a + px * h, root[1] + uy * a + py * h)
    return mid, end


def bend_angle(root: Point, mid: Point, end: Point) -> Optional[float]:
    """Interior angle at the mid joint, degrees (180 = straight)."""
    v1 = (root[0] - mid[0], root[1] - mid[1])
    v2 = (end[0] - mid[0], end[1] - mid[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1])
                        / (n1 * n2)))
    return round(math.degrees(math.acos(cos)), 2)
