"""Adaptive foreground threshold — Otsu on the bg-distance field.

`_mask` decides foreground by "any channel differs from the border
mode color by > 40". That constant assumes real photos behave like
the fixtures; under different lighting, exposure, or a closer-toned
background the same 40 silently over- or under-segments.

Otsu's method replaces the constant with evidence: build the
per-pixel distance-to-background histogram, then choose the split
that maximizes between-class variance — the classic 1979
discriminant criterion, so the threshold adapts to what the image
actually contains. When the field is not bimodal (flat background,
no measurable separation) the caller falls back to the constant —
a weak separation claim is itself reported, never guessed.
"""

from __future__ import annotations

from typing import List, Tuple

BINS = 64
# below this Otsu threshold the split sits inside quantization
# noise (bg bin centers are ~8 units from their pixels); keep the
# caller's fixed default instead of believing a degenerate split
MIN_MEANINGFUL = 4.0


def _dist_hist(dist: List[float], max_dist: float) -> List[int]:
    hist = [0] * BINS
    scale = BINS / max_dist if max_dist > 0 else 0
    for d in dist:
        i = int(d * scale)
        hist[i if i < BINS else BINS - 1] += 1
    return hist


def otsu_threshold(dist: List[float], max_dist: float = 441.7
                   ) -> float:
    """Between-class-variance optimum on the distance field.

    `max_dist` is the largest possible bg distance — sqrt(3*255²)
    ≈ 441.7 for RGB. Returns the threshold in distance units, or
    0.0 when there is no bimodal structure to exploit (empty input
    or a single occupied bin).
    """
    if not dist:
        return 0.0
    hist = _dist_hist(dist, max_dist)
    total = len(dist)
    sum_all = sum(i * hist[i] for i in range(BINS))
    best_var, best_i = -1.0, -1
    w0 = s0 = 0
    for i in range(BINS):
        w0 += hist[i]
        s0 += i * hist[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        mean0, mean1 = s0 / w0, (sum_all - s0) / w1
        var = w0 * w1 * (mean0 - mean1) ** 2
        if var > best_var:
            best_var, best_i = var, i
    if best_i < 0:
        return 0.0
    return (best_i + 0.5) * max_dist / BINS


def threshold(dist: List[float], *, fallback: float = 40.0,
              max_dist: float = 441.7) -> Tuple[float, str]:
    """Pick the fg/bg distance threshold.

    Returns (value, method): "otsu" when a meaningful split exists
    (>= MIN_MEANINGFUL), otherwise ("fixed", fallback) — the honest
    thing to do when the histogram shows no separation at all.
    """
    t = otsu_threshold(dist, max_dist)
    if t >= MIN_MEANINGFUL:
        return t, "otsu"
    return fallback, "fixed"


def distances(bmp, bg_rgb: Tuple[int, int, int]) -> List[float]:
    """Per-pixel RGB distance to the background color (opaque only)."""
    br, bg, bb = bg_rgb
    d = bmp.data
    out: List[float] = []
    n = bmp.width * bmp.height
    for i in range(n):
        o = i * 4
        if d[o + 3] < 128:
            out.append(0.0)
            continue
        dr, dg, db = d[o] - br, d[o + 1] - bg, d[o + 2] - bb
        out.append((dr * dr + dg * dg + db * db) ** 0.5)
    return out
