"""Joint confidence heatmaps — the OpenPose-style output format.

Each joint becomes a Gaussian blob whose peak intensity is its
confidence: observed joints burn bright, predicted ones burn dim or
not at all. This is the standard representation deep pose estimators
both consume and produce (Hourglass/PAF/BlazePose), so rendering our
result in the same form makes Slice output directly comparable —
and shows *where* the engine is sure vs guessing.
"""

from __future__ import annotations

import math
from typing import Optional

from .bitmap import Bitmap
from .skeleton import PREDICTED, Skeleton


def render(skel: Skeleton, width: Optional[int] = None,
           height: Optional[int] = None, sigma: float = 5.0,
           predicted_gain: float = 0.3) -> Bitmap:
    """Grayscale heatmap; intensity ≈ confidence × gaussian(x, y)."""
    w = width or skel.image_width
    h = height or skel.image_height
    out = Bitmap.new(w, h, (0, 0, 0, 255))
    for j in skel.joints.values():
        gain = 1.0 if j.state != PREDICTED else predicted_gain
        peak = j.confidence * gain
        if peak <= 0:
            continue
        r = max(1, int(sigma * 3))
        cx, cy = int(round(j.x)), int(round(j.y))
        for y in range(max(0, cy - r), min(h, cy + r + 1)):
            for x in range(max(0, cx - r), min(w, cx + r + 1)):
                g = math.exp(-((x - j.x) ** 2 + (y - j.y) ** 2)
                             / (2 * sigma * sigma))
                v = int(255 * peak * g)
                if v <= 0:
                    continue
                cur = out.get(x, y)
                if v > cur[0]:
                    out.set(x, y, (v, v, v, 255))
    return out


def max_at(bmp: Bitmap, x: int, y: int, radius: int = 2) -> int:
    """Peak heatmap value within `radius` of (x, y)."""
    best = 0
    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):
            if 0 <= xx < bmp.width and 0 <= yy < bmp.height:
                best = max(best, bmp.get(xx, yy)[0])
    return best
