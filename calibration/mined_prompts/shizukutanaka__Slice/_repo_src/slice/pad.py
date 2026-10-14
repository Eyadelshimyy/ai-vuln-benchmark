"""Letterbox padding — grow the frame, never invent pixels inside it.

`crop` removes; `pad` adds background around the image so a batch of
odd-sized frames shares one canvas. Functions return the (ox, oy)
offset the original image landed at, because a padded bitmap without
its offset is coordinates waiting to be lied about — skeleton and
mask consumers must add it back.
"""

from __future__ import annotations

from typing import Tuple

from .bitmap import Bitmap

_BG = (0, 0, 0, 0)


def pad(bmp: Bitmap, width: int, height: int,
        bg=_BG) -> Tuple[Bitmap, int, int]:
    """Center `bmp` on a `width`x`height` canvas.

    Returns (canvas, offset_x, offset_y). Refuses to shrink — that is
    `crop`'s job, and silently picking one is how offsets get lost.
    """
    if width < bmp.width or height < bmp.height:
        raise ValueError(
            "pad target smaller than source; use crop instead")
    out = Bitmap.new(width, height, bg)
    ox = (width - bmp.width) // 2
    oy = (height - bmp.height) // 2
    for y in range(bmp.height):
        for x in range(bmp.width):
            out.set(ox + x, oy + y, bmp.get(x, y))
    return out, ox, oy


def to_aspect(bmp: Bitmap, aspect: float,
              bg=_BG) -> Tuple[Bitmap, int, int]:
    """Grow the short side until width/height == aspect (>=0 required)."""
    if aspect <= 0:
        raise ValueError("aspect must be positive")
    w, h = bmp.width, bmp.height
    if w / h < aspect:
        w = int(h * aspect + 0.5)
    else:
        h = int(w / aspect + 0.5)
    return pad(bmp, w, h, bg)


def to_square(bmp: Bitmap, bg=_BG) -> Tuple[Bitmap, int, int]:
    """The commonest case: pad to max(w,h) on both axes."""
    return to_aspect(bmp, 1.0, bg)
