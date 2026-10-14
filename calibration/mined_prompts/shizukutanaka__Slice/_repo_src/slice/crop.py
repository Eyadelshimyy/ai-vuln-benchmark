"""Rectangle crop — actually cut the rect autocrop suggests.

`autocrop` answers *where* to crop; `crop` performs it. The rect is
clamped to the frame (a suggestion that overflows the right/bottom
edge yields what's actually inside, not a black border invented from
nothing). A rect with no overlap at all raises — cropping nothing
and returning a frame is lying.
"""

from __future__ import annotations

from typing import Tuple, Union

from .bitmap import Bitmap

Rect = Union[Tuple[int, int, int, int], dict]


def crop(bmp: Bitmap, rect: Rect) -> Bitmap:
    """Return the pixels inside `rect` (x, y, w, h or a dict), clamped."""
    if isinstance(rect, dict):
        x, y = int(rect.get("x", 0)), int(rect.get("y", 0))
        w = int(rect.get("width", rect.get("w", 0)))
        h = int(rect.get("height", rect.get("h", 0)))
    else:
        x, y, w, h = (int(v) for v in rect)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(bmp.width, x + w)
    y1 = min(bmp.height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("crop rect has no overlap with frame")
    out = Bitmap.new(x1 - x0, y1 - y0, (0, 0, 0, 0))
    for yy in range(y1 - y0):
        for xx in range(x1 - x0):
            out.set(xx, yy, bmp.get(x0 + xx, y0 + yy))
    return out
