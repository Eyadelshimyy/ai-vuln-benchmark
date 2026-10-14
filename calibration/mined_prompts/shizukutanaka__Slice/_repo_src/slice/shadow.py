"""Cast-shadow rejection — darker copies of the background.

A real photograph's cast shadow fools `_mask` the same way a body
does: its pixels differ from the border mode color. But a shadow is
a *uniformly darkened* version of the local background — it darkens
all channels by roughly the same factor (luminance scale, no hue
shift). Clothing differs: a dark jacket shifts hue as well as
brightness.

`is_shadow` therefore checks two things: the pixel is darker than
background (scale < 1), and all channel ratios agree within
tolerance — a pure dimming, not a color change. Foreground pixels
failing that test are just... pixels that differ; the mask keeps
them. The distinction is claimed per pixel, never guessed in bulk.
Very dark pixels (< 0.35 × bg) are deliberately *not* shadows:
black clothing and deep shadows share that range, so those stay
foreground on the same-surface doubt.
"""

from __future__ import annotations

from typing import List, Tuple

RATIO_TOL = 0.22      # channel ratios must agree within this band
MAX_SCALE = 0.92      # above this it isn't "darker" enough to be shadow
MIN_SCALE = 0.35      # below this it reads as clothing/object, not
                      # darkened background — the same-surface claim
                      # is too weak to claim

Mask = List[bytearray]


def _scale(a: int, b: int) -> float:
    return a / b if b else (1.0 if a == 0 else 0.0)


def is_shadow(r: int, g: int, b: int,
              bg: Tuple[int, int, int]) -> bool:
    """True when the pixel looks like bg darkened by one factor."""
    rs = [_scale(r, bg[0]), _scale(g, bg[1]), _scale(b, bg[2])]
    mean = sum(rs) / 3.0
    if not MIN_SCALE <= mean <= MAX_SCALE:
        return False
    return all(abs(s - mean) <= RATIO_TOL * mean for s in rs)


def shadow_pixels(bmp, bg_rgb: Tuple[int, int, int],
                  mask: Mask) -> Mask:
    """Foreground pixels classified as cast shadow (1 = shadow)."""
    w, h = bmp.width, bmp.height
    out = [bytearray(w) for _ in range(h)]
    d = bmp.data
    for y in range(h):
        for x in range(w):
            if not mask[y][x]:
                continue
            i = (y * w + x) * 4
            if is_shadow(d[i], d[i + 1], d[i + 2], bg_rgb):
                out[y][x] = 1
    return out


def remove(mask: Mask, shadows: Mask, *, flatness: float = 0.30
           ) -> Tuple[Mask, int]:
    """Drop flat shadow components from a foreground mask.

    Color alone can't tell a dark gray jacket from a cast shadow —
    both are uniformly dimmed background. The shape disambiguates:
    a real cast shadow spreads along the ground, so its connected
    component is flat (height < `flatness` × width); a body stands
    tall and is never removed, even when its pixels look dim.

    Returns (cleaned mask, removed count)."""
    h, w = len(mask), len(mask[0])
    labels = [[0] * w for _ in range(h)]
    boxes = {}
    label = 0
    for y in range(h):
        for x in range(w):
            if not shadows[y][x] or labels[y][x]:
                continue
            label += 1
            cells = [(x, y)]
            labels[y][x] = label
            for cx, cy in cells:
                for nx, ny in ((cx - 1, cy), (cx + 1, cy),
                               (cx, cy - 1), (cx, cy + 1)):
                    if (0 <= nx < w and 0 <= ny < h
                            and shadows[ny][nx] and not labels[ny][nx]):
                        labels[ny][nx] = label
                        cells.append((nx, ny))
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            boxes[label] = (max(xs) - min(xs) + 1,
                            max(ys) - min(ys) + 1, cells)
    removed = 0
    for bw, bh, cells in boxes.values():
        if bh >= flatness * bw:
            continue  # tall structure — a body part, not a shadow
        for x, y in cells:
            if mask[y][x]:
                mask[y][x] = 0
                removed += 1
    return mask, removed
