"""Distance transform of the silhouette — local limb thickness.

Chamfer (3-4) two-pass distance transform: every foreground pixel gets
its distance to the nearest background pixel. The value at a joint is
roughly half the limb width there — arms read thin, the torso reads
thick. This is the standard way to measure local body-part thickness
from a silhouette, and future work (limb-width priors, muscle/skin
estimation) builds on it.
"""

from __future__ import annotations

from typing import List

_D1, _D2 = 3.0, 4.0  # chamfer weights (orthogonal / diagonal)


def distance_transform(mask: List[bytearray]) -> List[float]:
    """Distance in pixels from each foreground pixel to the background."""
    h, w = len(mask), len(mask[0])
    inf = float("inf")
    d = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if not mask[y][x]:
                continue
            # image outside counts as background → border fg = 1 step
            best = _D1 if (x == 0 or x == w - 1 or
                           y == 0 or y == h - 1) else inf
            if y:
                best = min(best, d[y - 1][x] + _D1)
                if x:
                    best = min(best, d[y - 1][x - 1] + _D2)
                if x < w - 1:
                    best = min(best, d[y - 1][x + 1] + _D2)
            if x:
                best = min(best, d[y][x - 1] + _D1)
            d[y][x] = best
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if not mask[y][x]:
                continue
            best = d[y][x]
            if y < h - 1:
                best = min(best, d[y + 1][x] + _D1)
                if x:
                    best = min(best, d[y + 1][x - 1] + _D2)
                if x < w - 1:
                    best = min(best, d[y + 1][x + 1] + _D2)
            if x < w - 1:
                best = min(best, d[y][x + 1] + _D1)
            d[y][x] = best
    return [v / _D1 for row in d for v in row]


def thickness_at(dist: List[float], w: int, x: float, y: float) -> float:
    """Local thickness estimate = 2 × distance at the pixel."""
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < len(dist) // w):
        return 0.0
    return 2.0 * dist[yi * w + xi]


def thickness_profile(dist: List[float], w: int, h: int,
                      mask: List[bytearray]) -> dict:
    """Global thickness stats: max (torso core) and median of the body."""
    vals = [dist[y * w + x]
            for y in range(h) for x in range(w) if mask[y][x]]
    if not vals:
        return {"max": 0.0, "median": 0.0}
    vals.sort()
    return {"max": round(2.0 * vals[-1], 1),
            "median": round(2.0 * vals[len(vals) // 2], 1)}
