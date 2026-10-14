"""Empty-skeleton diagnosis — why did estimate() return nothing?

The estimator answers a blank frame and a blank pose the same way:
an empty Skeleton. `diagnose` replays the pipeline stages and reports
the first gate that failed, as reason codes, so "no result" becomes
"no result, because ...".

Codes
-----
no_foreground         mask is empty — person ≈ background colour,
                      or frame is uniform / fully transparent
too_small             foreground exists but < 0.5% of the frame
too_short             body bbox under the 24px minimum height
foreground_at_edge    fg touches all 4 borders — background colour
                      estimate is probably polluted by the subject
low_contrast          fg↔bg mean colour distance is under ~1.5×thr
ok                    every gate passed — skeleton should be non-empty
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .bitmap import Bitmap
from .pose import HeuristicPoseEstimator

_MIN_FG_FRAC = 0.005
_MIN_BODY_H = 24


def diagnose(bmp: Bitmap,
             est: HeuristicPoseEstimator = None) -> dict:
    est = est or HeuristicPoseEstimator()
    small = bmp.downscale(est.max_dim)
    w, h = small.width, small.height
    mask = est._mask(small)
    fg_px = sum(sum(r) for r in mask)
    frac = fg_px / (w * h) if w * h else 0.0

    reasons: List[str] = []
    if fg_px == 0:
        reasons.append("no_foreground")
    elif frac < _MIN_FG_FRAC:
        reasons.append("too_small")

    comp, comp_size = est._largest_component(mask, w, h)
    touches = _border_touch(comp, w, h)
    if fg_px and touches == 4:
        reasons.append("foreground_at_edge")

    dist = _fg_distance(small, mask, est._background(small))
    if fg_px and dist is not None and dist < est.bg_threshold * 1.5:
        reasons.append("low_contrast")

    body_h = _body_height(comp, w, h)
    if comp_size and body_h is not None and body_h < _MIN_BODY_H:
        reasons.append("too_short")

    return {
        "ok": not reasons,
        "reasons": reasons or ["ok"],
        "fg_coverage": round(frac, 5),
        "fg_pixels": fg_px,
        "largest_component": comp_size,
        "fg_bg_distance": None if dist is None else round(dist, 1),
        "borders_touched": touches,
        "basis": "replays _mask/_largest_component gates of %s %s"
                 % (est.name, est.version),
    }


def _border_touch(comp: List[bytearray], w: int, h: int) -> int:
    n = 0
    if any(comp[0][x] for x in range(w)):
        n += 1
    if any(comp[h - 1][x] for x in range(w)):
        n += 1
    if any(comp[y][0] for y in range(h)):
        n += 1
    if any(comp[y][w - 1] for y in range(h)):
        n += 1
    return n


def _fg_distance(bmp: Bitmap, mask: List[bytearray],
                 bg: Tuple[int, int, int]) -> Optional[float]:
    w, h = bmp.width, bmp.height
    d = bmp.data
    tot = [0.0, 0.0]
    n = 0
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if row[x]:
                i = (y * w + x) * 4
                tot[0] += abs(d[i] - bg[0])
                tot[1] += max(abs(d[i + 1] - bg[1]),
                              abs(d[i + 2] - bg[2]))
                n += 1
    return (tot[0] + tot[1]) / (2 * n) if n else None


def _body_height(comp: List[bytearray], w: int, h: int) -> Optional[int]:
    rows = [y for y in range(h) if any(comp[y][x] for x in range(w))]
    return (rows[-1] - rows[0] + 1) if rows else None
