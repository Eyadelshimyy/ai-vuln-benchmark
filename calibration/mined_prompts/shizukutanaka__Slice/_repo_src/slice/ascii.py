"""ASCII rendering — the skeleton on a terminal.

`render.overlay` draws PNG, `svg.render` draws vectors; `ascii`
draws the same skeleton into a character grid for CLIs, logs and
tests where a picture isn't a file. The honesty code survives in
glyph form: observed joints `@`, predicted `o`; observed bones `#`,
predicted bones `:`. Anything the skeleton doesn't contain is
simply absent — blanks are never drawn as evidence.
"""

from __future__ import annotations

from .landmarks import BONES
from .skeleton import OBSERVED, Skeleton

_J_OBS, _J_PRED = "@", "o"
_B_OBS, _B_PRED = "#", ":"


def render(skel: Skeleton, *, width: int = 40) -> str:
    """ASCII art of the skeleton at roughly `width` columns.

    Rows follow the frame's aspect ratio corrected for character
    cells (~1:2), so a 2:3 portrait frame renders proportionally.
    """
    w = max(8, int(width))
    rows = max(4, int(round(w * skel.image_height
                            / max(1, skel.image_width) * 0.5)))
    grid = [[" "] * w for _ in range(rows)]

    def cell(x: float, y: float):
        cx = min(w - 1, max(0, int(x / skel.image_width * w)))
        cy = min(rows - 1, max(0, int(y / skel.image_height * rows)))
        return cx, cy

    # Bones first; joints overdraw them so endpoints stay visible.
    for a, b in BONES:
        ja, jb = skel.get(a), skel.get(b)
        if not ja or not jb:
            continue
        ch = _B_OBS if (ja.state == OBSERVED
                        and jb.state == OBSERVED) else _B_PRED
        for x, y in _line_cells(cell(ja.x, ja.y), cell(jb.x, jb.y)):
            grid[y][x] = ch
    for j in skel.joints.values():
        x, y = cell(j.x, j.y)
        grid[y][x] = _J_OBS if j.state == OBSERVED else _J_PRED
    return "\n".join("".join(r).rstrip() for r in grid)


def _line_cells(a, b):
    """Bresenham cells from a to b (exclusive of endpoints)."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    out = []
    while not (x0 == x1 and y0 == y1):
        out.append((x0, y0))
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return out
