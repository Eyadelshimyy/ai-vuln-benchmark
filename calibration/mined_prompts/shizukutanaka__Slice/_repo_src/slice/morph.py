"""Binary morphology — speckle and pinhole cleanup for masks.

Compression artifacts and antialiased edges leave noise in the
foreground mask: isolated specks that look like fragments, and
one-pixel holes punched inside the silhouette. Neither is
evidence of a body part; both distort the component analysis
downstream (a speck becomes a fake "person candidate", a hole
becomes a phantom gap).

The classical fix is morphological opening and closing with a
small structuring element — defined here on 4-connectivity to
match the component labelling in `pose`. Every op is a pure mask
transform: what it removes or fills is measurable, never hidden.
"""

from __future__ import annotations

from typing import List

Mask = List[bytearray]


def _empty(w: int, h: int) -> Mask:
    return [bytearray(w) for _ in range(h)]


def erode(mask: Mask) -> Mask:
    """Keep only pixels whose 4-neighbours are all foreground."""
    if not mask or not mask[0]:
        return mask
    h, w = len(mask), len(mask[0])
    out = _empty(w, h)
    for y in range(h):
        for x in range(w):
            if not mask[y][x]:
                continue
            if (x == 0 or x == w - 1 or y == 0 or y == h - 1):
                continue  # frame edge can't prove its surround
            if (mask[y][x - 1] and mask[y][x + 1]
                    and mask[y - 1][x] and mask[y + 1][x]):
                out[y][x] = 1
    return out


def dilate(mask: Mask) -> Mask:
    """Grow foreground into any 4-neighbour of a foreground pixel."""
    if not mask or not mask[0]:
        return mask
    h, w = len(mask), len(mask[0])
    out = _empty(w, h)
    for y in range(h):
        for x in range(w):
            if not mask[y][x]:
                continue
            out[y][x] = 1
            if x:
                out[y][x - 1] = 1
            if x < w - 1:
                out[y][x + 1] = 1
            if y:
                out[y - 1][x] = 1
            if y < h - 1:
                out[y + 1][x] = 1
    return out


def open(mask: Mask, iterations: int = 1) -> Mask:
    """Erode then dilate — removes isolated specks < ~2px."""
    out = mask
    for _ in range(iterations):
        out = erode(out)
    for _ in range(iterations):
        out = dilate(out)
    return out


def close(mask: Mask, iterations: int = 1) -> Mask:
    """Dilate then erode — fills one-pixel holes inside regions."""
    out = mask
    for _ in range(iterations):
        out = dilate(out)
    for _ in range(iterations):
        out = erode(out)
    return out


def clean(mask: Mask) -> Mask:
    """open-then-close: speckle out, pinholes filled."""
    return close(open(mask))
