"""Near-realistic synthetic fixtures — procedural stand-ins for real
photos. The estimator was built and tested against clean flat
backgrounds; these fixtures exercise the failure modes AUDIT P0
flags: gradient walls, sensor noise, occluders, and cast shadows.

Each helper starts from tests.synthetic_person (dark figure, uniform
bg) and repaints only the background pixels, so the foreground shape
stays identical across variants.
"""
from __future__ import annotations

import random

from tests import synthetic_person

BG = (235, 235, 235, 255)


def _recolor_bg(bmp, fn):
    """Replace every pixel still equal to the flat fixture bg via
    fn(x, y) -> (r, g, b, a)."""
    for y in range(bmp.height):
        for x in range(bmp.width):
            if bmp.get(x, y) == BG:
                bmp.set(x, y, fn(x, y))
    return bmp


def gradient_person(width=160, height=300, depth=30):
    """Figure on a vertical gradient wall (mild: <40 total drift so
    the threshold detector should still isolate the figure)."""
    bmp = synthetic_person(width, height, bg=BG)
    def wall(x, y):
        t = y / max(1, height - 1)
        v = int(235 - depth * t)
        return (v, v, min(255, v + 4), 255)
    return _recolor_bg(bmp, wall)


def noisy_person(width=160, height=300, amp=12, seed=7):
    """Figure on sensor-noise textured background (uniform noise ±amp)."""
    rng = random.Random(seed)
    bmp = synthetic_person(width, height, bg=BG)
    def wall(x, y):
        n = rng.randint(-amp, amp)
        v = max(0, min(255, 235 + n))
        return (v, v, v, 255)
    return _recolor_bg(bmp, wall)


def occluded_person(width=160, height=300):
    """Figure with the lower-right quadrant hidden behind a box
    the colour of the background — partial-body input."""
    bmp = synthetic_person(width, height, bg=BG)
    for y in range(int(height * 0.6), height):
        for x in range(int(width * 0.55), width):
            bmp.set(x, y, BG)
    return bmp


def shadow_person(width=160, height=300):
    """Figure casting a soft elliptical shadow at its feet — the
    classic reason silhouette extractors over-segment."""
    bmp = synthetic_person(width, height, bg=BG)
    cx, cy = width // 2, height - 8
    rx, ry = width * 0.30, 6.0
    def wall(x, y):
        d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
        if d < 1.0:
            shade = int(235 - 60 * (1.0 - d))
            return (shade, shade, shade, 255)
        return BG
    return _recolor_bg(bmp, wall)
