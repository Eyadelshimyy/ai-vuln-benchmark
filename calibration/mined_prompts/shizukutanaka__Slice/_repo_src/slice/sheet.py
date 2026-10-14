"""Contact sheet — many skeleton renders in one grid image.

Dataset review shouldn't mean opening 200 PNGs. `sheet` tiles the
render overlay of each skeleton into a grid: every tile is scaled to
preserve that skeleton's frame aspect (letterboxed, never stretched),
so a glance compares poses across a whole batch.

Lying by distortion is still lying: a tile whose frame aspect differs
from the cell's keeps its proportions and pads the rest with the
background colour.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .bitmap import Bitmap, encode_png
from .render import overlay

_BG = (18, 18, 22, 255)


def _paste(dst: Bitmap, src: Bitmap, ox: int, oy: int) -> None:
    for y in range(src.height):
        for x in range(src.width):
            dst.set(ox + x, oy + y, src.get(x, y))


def sheet(skels: Sequence, cols: Optional[int] = None,
          cell: int = 128, bg=_BG) -> Bitmap:
    """Tile skeleton overlays into a grid bitmap.

    Each skeleton is letterboxed into a `cell`x`cell` square via the
    aspect-preserving scale already inside `render.overlay`.
    """
    n = len(skels)
    if n == 0:
        return Bitmap.new(cell, cell, bg)
    cols = cols or max(1, round(n ** 0.5 + 0.5))
    rows = (n + cols - 1) // cols
    out = Bitmap.new(cols * cell, rows * cell, bg)
    for i, skel in enumerate(skels):
        scale = min(cell / max(1, skel.image_width),
                    cell / max(1, skel.image_height))
        tw = max(1, int(skel.image_width * scale))
        th = max(1, int(skel.image_height * scale))
        tile = overlay(Bitmap.new(tw, th, bg), skel)
        ox = (i % cols) * cell + (cell - tw) // 2
        oy = (i // cols) * cell + (cell - th) // 2
        _paste(out, tile, ox, oy)
    return out


def sheet_png(skels: Sequence, cols: Optional[int] = None,
              cell: int = 128, bg=_BG) -> bytes:
    return encode_png(sheet(skels, cols=cols, cell=cell, bg=bg))
