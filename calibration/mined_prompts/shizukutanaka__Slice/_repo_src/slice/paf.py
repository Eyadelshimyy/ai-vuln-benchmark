"""Part Affinity Fields — association fields along bones.

OpenPose pairs joint heatmaps with PAFs: for every bone, a vector
field in a thin band around the segment pointing from the parent
joint to the child. It encodes *which joints belong together* — the
association channel, not the detection channel. Rendering our bones
in this form makes Slice's output consumable by the same tooling and
lets us see where limb connectivity is asserted vs guessed
(predicted bones emit a weak field).
"""

from __future__ import annotations

import math
from typing import List

from .landmarks import BONES
from .skeleton import PREDICTED, Skeleton


def _band_score(px: float, py: float, a, b, width: float) -> float:
    """1.0 on the segment, falling to 0 at `width` px away."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    if not (0.0 <= t <= 1.0):
        return 0.0
    dist = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
    return max(0.0, 1.0 - dist / width)


def field(skel: Skeleton, width: float = 6.0,
          stride: int = 4,
          predicted_gain: float = 0.3) -> List[dict]:
    """One PAF per bone: unit direction × confidence × coverage mask.

    Returns a list of {bone, dir, strength, cells:[(x,y)]} where cells
    are the stride-spaced grid points inside the bone's band.
    """
    out: List[dict] = []
    w, h = skel.image_width, skel.image_height
    for ja, jb in BONES:
        pa, pb = skel.point(ja), skel.point(jb)
        if not pa or not pb:
            continue
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        unit = (dx / L, dy / L)
        conf = min(skel.joints[ja].confidence,
                   skel.joints[jb].confidence)
        if PREDICTED in (skel.joints[ja].state,
                         skel.joints[jb].state):
            conf *= predicted_gain
        x0 = max(0, int(min(pa[0], pb[0]) - width))
        x1 = min(w, int(max(pa[0], pb[0]) + width) + 1)
        y0 = max(0, int(min(pa[1], pb[1]) - width))
        y1 = min(h, int(max(pa[1], pb[1]) + width) + 1)
        cells = []
        for y in range(y0, y1, stride):
            for x in range(x0, x1, stride):
                if _band_score(x + 0.5, y + 0.5, pa, pb, width) > 0:
                    cells.append((x, y))
        out.append({"bone": (ja, jb), "dir": (round(unit[0], 3),
                                              round(unit[1], 3)),
                    "strength": round(conf, 3), "cells": cells})
    return out


