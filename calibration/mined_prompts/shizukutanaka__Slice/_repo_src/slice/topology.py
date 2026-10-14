"""Silhouette topology — components and holes.

Euler number = connected foreground components − enclosed holes.
A plain standing figure is 1 and 0. A hand on a hip closes a
triangular hole; a donut pose (hands together overhead) closes a
bigger one. Holes are pose semantics a skeleton can't express —
this measures them directly on the mask.

Background regions are flood-filled from the border; any
background region the border can't reach is a hole. Small holes
(< `min_hole` px) are likely raster noise and are ignored.
"""

from __future__ import annotations

from typing import List, Optional

from .bitmap import Bitmap
from .pose import HeuristicPoseEstimator


def _regions(mask: List[bytearray], w: int, h: int,
             want_fg: bool):
    """Connected components of fg (want_fg) or bg."""
    lab = [[-1] * w for _ in range(h)]
    out = []
    for y in range(h):
        for x in range(w):
            if lab[y][x] >= 0 or bool(mask[y][x]) != want_fg:
                continue
            cid = len(out)
            stack, cells = [(x, y)], []
            lab[y][x] = cid
            while stack:
                px, py = stack.pop()
                cells.append((px, py))
                for nx, ny in ((px + 1, py), (px - 1, py),
                               (px, py + 1), (px, py - 1)):
                    if (0 <= nx < w and 0 <= ny < h
                            and lab[ny][nx] < 0
                            and bool(mask[ny][nx]) == want_fg):
                        lab[ny][nx] = cid
                        stack.append((nx, ny))
            out.append(cells)
    return out


def analyze(bmp: Bitmap, min_hole: int = 12) -> Optional[dict]:
    """{components, holes, hole_areas, euler, fg_px} or None."""
    small = bmp.downscale(256)
    est = HeuristicPoseEstimator()
    mask = est._mask(small)
    w, h = small.width, small.height
    comp, size = est._largest_component(mask, w, h)
    if size == 0:
        return None
    fg = _regions(mask, w, h, want_fg=True)
    holes = []
    for region in _regions(mask, w, h, want_fg=False):
        if any(x == 0 or y == 0 or x == w - 1 or y == h - 1
               for x, y in region):
            continue  # reaches the border → outside, not a hole
        if len(region) >= min_hole:
            holes.append(len(region))
    return {"components": len(fg),
            "holes": len(holes),
            "hole_areas": sorted(holes, reverse=True),
            "euler": len(fg) - len(holes),
            "fg_px": sum(len(r) for r in fg),
            "state": "observed",
            "basis": "foreground mask connected components"}
