"""Horizontal flipping for bitmaps and skeletons.

A pose estimator built from silhouette features should be nearly
symmetric: `estimate(flip(img))` should equal `flip(estimate(img))`.
These utilities provide both halves of that equation so tests and
augmentation can measure the estimator's left/right consistency.
"""

from __future__ import annotations

from .bitmap import Bitmap
from .landmarks import MIRROR
from .skeleton import Joint, Skeleton


def flip_bitmap(bmp: Bitmap) -> Bitmap:
    """Left/right mirror of an image."""
    w, h = bmp.width, bmp.height
    out = Bitmap.new(w, h, (0, 0, 0, 0))
    for y in range(h):
        for x in range(w):
            p = bmp.get(x, y)
            out.set(w - 1 - x, y, p)
    return out


def flip_skeleton(skel: Skeleton) -> Skeleton:
    """Left/right mirror of a skeleton: x flips, _l/_r joints swap."""
    out = Skeleton(skel.image_width, skel.image_height)
    for name, j in skel.joints.items():
        out.set(Joint(MIRROR.get(name, name),
                      skel.image_width - 1 - j.x, j.y,
                      j.confidence, j.state, j.basis))
    orient = dict(skel.orientation)
    if orient.get("facing") == "left":
        orient["facing"] = "right"
    elif orient.get("facing") == "right":
        orient["facing"] = "left"
    if "head_shift" in orient:
        orient["head_shift"] = -orient["head_shift"]
    out.orientation = orient
    out.body_model = dict(skel.body_model)
    if skel.centroid is not None:
        out.centroid = (skel.image_width - 1 - skel.centroid[0],
                        skel.centroid[1])
    return out
