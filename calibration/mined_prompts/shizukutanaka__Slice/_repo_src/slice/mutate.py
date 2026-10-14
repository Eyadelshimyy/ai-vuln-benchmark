"""Image mutations for robustness testing.

Real inputs are noisy, partially occluded, and cropped. These
transforms degrade a bitmap in controlled ways so tests can verify the
estimator degrades *honestly* — missing evidence should yield missing
or predicted joints, never fabricated observed ones.
"""

from __future__ import annotations

import random
from typing import Tuple

from .bitmap import Bitmap

RGBA = Tuple[int, int, int, int]


def add_noise(bmp: Bitmap, amount: int = 30, *, seed: int = 0) -> Bitmap:
    """Uniform per-pixel channel jitter."""
    rng = random.Random(seed)
    out = Bitmap.new(bmp.width, bmp.height, (0, 0, 0, 255))
    for y in range(bmp.height):
        for x in range(bmp.width):
            r, g, b, a = bmp.get(x, y)
            j = rng.randrange(-amount, amount + 1)
            out.set(x, y, (max(0, min(255, r + j)),
                           max(0, min(255, g + j)),
                           max(0, min(255, b + j)), a))
    return out


def occlude(bmp: Bitmap, x0: int, y0: int, x1: int, y1: int,
            color: RGBA = (235, 235, 235, 255)) -> Bitmap:
    """Blank a rectangle — simulates a barrier hiding body parts."""
    out = Bitmap.new(bmp.width, bmp.height, (0, 0, 0, 255))
    out.data[:] = bmp.data
    for y in range(max(0, y0), min(bmp.height, y1)):
        for x in range(max(0, x0), min(bmp.width, x1)):
            out.set(x, y, color)
    return out


def crop(bmp: Bitmap, x0: int, y0: int, x1: int, y1: int) -> Bitmap:
    """Trim to a sub-rectangle — simulates framing that cuts the body.

    Sample positions outside the frame leave the background fill:
    `Bitmap.get` wraps negative indices to the image tail, so reading
    there would fabricate content where the camera saw nothing.
    """
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    out = Bitmap.new(w, h, (0, 0, 0, 255))
    for y in range(h):
        for x in range(w):
            sx, sy = x0 + x, y0 + y
            if 0 <= sx < bmp.width and 0 <= sy < bmp.height:
                out.set(x, y, bmp.get(sx, sy))
    return out
