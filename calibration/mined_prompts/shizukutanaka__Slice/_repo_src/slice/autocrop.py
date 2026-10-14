"""Auto-crop — a tight frame around the person.

Pre-processing utility: find the foreground bounding box and emit a
suggested crop rectangle, optionally padded by a margin or expanded
to a target aspect ratio. It answers "where in this image is the
person" using only silhouette evidence — the same mask the pose
estimator trusts.

Everything returned is `state: "observed"` when pixels back it, or
`"unknown"` (with crop=None) when the frame is empty — a missing
subject is reported, never guessed.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .bitmap import Bitmap
from .pose import HeuristicPoseEstimator

Rect = Tuple[int, int, int, int]


def person_bbox(bmp: Bitmap) -> Optional[Rect]:
    """Tight bounding box of the largest foreground component."""
    est = HeuristicPoseEstimator()
    mask = est._mask(bmp)
    comp, size = est._largest_component(mask, bmp.width, bmp.height)
    if size == 0:
        return None
    x0, y0, x1, y1 = bmp.width, bmp.height, -1, -1
    for y in range(bmp.height):
        row = comp[y]
        for x in range(bmp.width):
            if row[x]:
                if x < x0:
                    x0 = x
                if x > x1:
                    x1 = x
                if y < y0:
                    y0 = y
                if y > y1:
                    y1 = y
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _clamp(v: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(v))))


def suggest(bmp: Bitmap, *, margin: float = 0.1,
            aspect: Optional[float] = None) -> dict:
    """Crop suggestion around the person.

    `margin` pads the bbox by that fraction of its own size on each
    side. `aspect` (width/height, e.g. 3/4 for portrait) grows the
    shorter side, centered, until the rect matches; the result is
    always clamped to the frame.
    """
    box = person_bbox(bmp)
    if box is None:
        return {
            "bbox": None,
            "crop": None,
            "coverage": 0.0,
            "state": "unknown",
            "basis": "no foreground component found",
        }
    x, y, w, h = box
    mx, my = w * margin, h * margin
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w + 2 * mx, h + 2 * my
    if aspect and aspect > 0:
        cur = nw / nh
        if cur < aspect:
            nw = nh * aspect
        else:
            nh = nw / aspect
    # Clamp size to the frame first, then center the window.
    nw = min(nw, bmp.width)
    nh = min(nh, bmp.height)
    rx = _clamp(cx - nw / 2, 0, bmp.width - int(nw))
    ry = _clamp(cy - nh / 2, 0, bmp.height - int(nh))
    coverage = (w * h) / (bmp.width * bmp.height)
    return {
        "bbox": [x, y, w, h],
        "crop": [rx, ry, int(nw), int(nh)],
        "coverage": round(coverage, 4),
        "state": "observed",
        "basis": ("largest-component bbox + %.0f%% margin%s" % (
            margin * 100,
            ", aspect %.3g" % aspect if aspect else "")),
    }
