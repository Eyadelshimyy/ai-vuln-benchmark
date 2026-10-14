"""Silhouette contour tracing and shape descriptors.

Extracts the outer boundary of a foreground mask (Moore-neighbor
tracing) and derives pose-independent shape features: perimeter,
area, bounding-box aspect, compactness (perimeter²/area), centroid.
These are classic silhouette descriptors — a tall thin standing figure
and a wide crouching one differ measurably in this space.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

# 8-neighborhood, clockwise starting at W
_N8 = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1),
       (-1, 1)]

Point = Tuple[int, int]


def _border(mask: List[bytearray], x: int, y: int) -> bool:
    """Foreground pixel with at least one non-fg 4-neighbor."""
    h, w = len(mask), len(mask[0])
    if not mask[y][x]:
        return False
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx >= w or ny >= h or not mask[ny][nx]:
            return True
    return False


def _trace(mask: List[bytearray]) -> Tuple[List[Point], bool]:
    """Moore trace + completion flag: (path, closed)."""
    h, w = len(mask), len(mask[0])
    start: Optional[Point] = None
    for y in range(h):
        for x in range(w):
            if mask[y][x]:
                start = (x, y)
                break
        if start:
            break
    if start is None:
        return [], True

    path = [start]
    cur = start
    back = 0  # direction index we arrived from; start heading W
    for _ in range(4 * w * h):
        found = None
        for k in range(8):
            i = (back + k) % 8
            nx, ny = cur[0] + _N8[i][0], cur[1] + _N8[i][1]
            if 0 <= nx < w and 0 <= ny < h and mask[ny][nx]:
                found = ((nx, ny), i)
                break
        if found is None:
            return path, True  # isolated pixel
        nxt, i = found
        back = (i + 5) % 8  # resume searching just past the backtrack
        if nxt == start:
            return path, True
        path.append(nxt)
        cur = nxt
    # cap reached without returning to the start pixel: the boundary
    # walked is longer than the bound allows (pathological masks)
    return path, False


def trace(mask: List[bytearray]) -> List[Point]:
    """Moore-neighbor boundary trace, clockwise, from the topmost fg."""
    return _trace(mask)[0]


def features(mask: List[bytearray]) -> dict:
    """Shape descriptors of the foreground component."""
    h, w = len(mask), len(mask[0])
    pts = [(x, y) for y in range(h) for x in range(w) if mask[y][x]]
    if not pts:
        return {"area": 0, "perimeter": 0, "bbox": None, "aspect": 0.0,
                "compactness": 0.0, "centroid": None}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
    contour, closed = _trace(mask)
    area = len(pts)
    # perimeter as polyline arc length: orthogonal steps cost 1,
    # diagonal steps cost sqrt(2) — counting traced pixels as 1px
    # each under-reports diagonal-heavy boundaries by up to ~29%.
    perim = 0.0
    for i in range(len(contour)):
        (ax, ay), (bx, by) = contour[i], contour[(i + 1) % len(contour)]
        perim += math.hypot(ax - bx, ay - by)
    perim = round(perim, 2)
    return {
        "area": area,
        "perimeter": perim,
        # False = the trace hit its iteration bound without closing;
        # perimeter and compactness are then lower bounds, not a
        # finished measurement — a truncated number must not pose
        # as a complete one
        "contour_closed": closed,
        "bbox": bbox,
        "aspect": round(bw / bh, 4),
        "compactness": round(perim * perim / (4 * math.pi * area), 3),
        "centroid": (round(sum(xs) / area, 1), round(sum(ys) / area, 1)),
    }
