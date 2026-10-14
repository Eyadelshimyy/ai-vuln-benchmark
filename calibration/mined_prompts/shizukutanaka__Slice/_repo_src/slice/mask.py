"""Foreground mask utilities — the estimator's mask, made inspectable.

`HeuristicPoseEstimator._mask` is where every analysis begins, but it
never leaves the estimator: contour and topology consumers rebuild it
privately, and nothing can show it. This module exposes it as data:

- `foreground` runs the same background-subtraction on any bitmap
- `to_bitmap` renders a mask for debugging (white fg / transparent bg)
- `cutout` keeps only foreground pixels (alpha-0 elsewhere)
- `coverage` reports the foreground fraction

Note the mask is computed on whatever bitmap you pass — if you feed a
`bmp.downscale()` result, mask rows match the *downscaled* skeleton
coordinate frame, not the original.
"""

from __future__ import annotations

from typing import List

from .bitmap import Bitmap
from .pose import HeuristicPoseEstimator

_FG = (255, 255, 255, 255)
_BG = (0, 0, 0, 0)


def foreground(bmp: Bitmap,
               est: HeuristicPoseEstimator = None
               ) -> List[bytearray]:
    """Foreground mask: list of h rows of w bytes, 1 = person."""
    return (est or HeuristicPoseEstimator())._mask(bmp)


def coverage(mask: List[bytearray]) -> float:
    fg = sum(sum(row) for row in mask)
    total = sum(len(row) for row in mask)
    return fg / total if total else 0.0


def to_bitmap(mask: List[bytearray], width: int, height: int,
              fg=_FG, bg=_BG) -> Bitmap:
    out = Bitmap.new(width, height, bg)
    for y in range(min(height, len(mask))):
        row = mask[y]
        for x in range(min(width, len(row))):
            if row[x]:
                out.set(x, y, fg)
    return out


def cutout(bmp: Bitmap,
           est: HeuristicPoseEstimator = None) -> Bitmap:
    """Same-size bitmap with background alpha-zeroed."""
    mask = foreground(bmp, est)
    # Copy original RGB everywhere; zero only background alpha.
    # Transparent pixels keep their colour so a later
    # `_background` estimate still reads the true background.
    out = Bitmap(bmp.width, bmp.height, bytearray(bmp.data))
    for y in range(bmp.height):
        row = mask[y]
        for x in range(bmp.width):
            if not row[x]:
                i = (y * bmp.width + x) * 4
                out.data[i + 3] = 0
    return out
