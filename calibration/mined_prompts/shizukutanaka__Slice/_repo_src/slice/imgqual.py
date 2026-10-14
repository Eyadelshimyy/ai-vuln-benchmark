"""Input image adequacy — is this picture evidence at all?

Every downstream layer assumes the input carries usable signal, yet
nothing checks it: a 64-pixel thumbnail, a pitch-black frame, or a
blur so heavy the silhouette has no edge all flow through the same
pipeline and produce a `Skeleton` whose `observed` joints look no
different from a clean input's. `imgqual` is the precondition
layer: four cheap measurements over the (downscaled) frame, each
yielding a named flag, folded into one adequacy verdict.

- `size`:   min dimension under `MIN_DIM` px — too little evidence
- `dynamic`:global channel stddev under `MIN_STD` — flat/dead frame
- `blur`:   mean |Laplacian| under `MIN_SHARP` — edges too soft to
            locate a silhouette boundary
- `contrast`: foreground-vs-border distance under `MIN_FG_BG` —
            no measurable separation (the estimator's own gate will
            starve; better to say so up front)

Pure-stdlib, byte-level math on `bmp.get` — no numpy. Each flag
carries its measured value so a caller can see *how far* off the
input is, not just that it failed.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .bitmap import Bitmap

MIN_DIM = 96          # px on the short edge
MIN_STD = 8.0         # global channel stddev
MIN_SHARP = 2.0       # mean |laplacian| on luminance
MIN_FG_BG = 15.0      # fg-vs-border RGB distance


def _lum(bmp: Bitmap, x: int, y: int) -> float:
    r, g, b, _ = bmp.get(x, y)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _opaque(bmp: Bitmap, x: int, y: int) -> bool:
    # alpha<128 = absent content, not a black pixel — a cutout's
    # transparent region must not vote its zeroed RGB into any
    # statistic (same rule style.analyze and pose._background use)
    return bmp.get(x, y)[3] >= 128


def _sharpness(bmp: Bitmap) -> float:
    """Mean absolute 4-neighbour Laplacian on luminance."""
    total = n = 0
    for y in range(1, bmp.height - 1, 2):
        for x in range(1, bmp.width - 1, 2):
            # a cutout outline is not content blur/sharpness — only a
            # fully-opaque cross measures real edge softness
            if not (_opaque(bmp, x, y) and _opaque(bmp, x - 1, y)
                    and _opaque(bmp, x + 1, y)
                    and _opaque(bmp, x, y - 1)
                    and _opaque(bmp, x, y + 1)):
                continue
            c = _lum(bmp, x, y)
            lap = abs(4 * c - _lum(bmp, x - 1, y) - _lum(bmp, x + 1, y)
                      - _lum(bmp, x, y - 1) - _lum(bmp, x, y + 1))
            total += lap
            n += 1
    return total / n if n else 0.0


def _stddev(bmp: Bitmap) -> float:
    vals: List[float] = []
    for y in range(0, bmp.height, 4):
        for x in range(0, bmp.width, 4):
            if _opaque(bmp, x, y):
                vals.append(_lum(bmp, x, y))
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _border_rgb(bmp: Bitmap):
    """Mean opaque border colour, or None when no border pixel is
    opaque — a fully transparent border is *absent* background, not
    a black one, and averaging its zeroed RGB would fabricate the
    reference the contrast flag is measured against."""
    pts = []
    for x in (0, bmp.width - 1):
        for y in range(0, bmp.height, max(1, bmp.height // 16)):
            p = bmp.get(x, y)
            if p[3] >= 128:
                pts.append(p[:3])
    for y in (0, bmp.height - 1):
        for x in range(0, bmp.width, max(1, bmp.width // 16)):
            p = bmp.get(x, y)
            if p[3] >= 128:
                pts.append(p[:3])
    if not pts:
        return None
    return [sum(p[i] for p in pts) / len(pts) for i in range(3)]


def _fg_bg_dist(bmp: Bitmap, bg: List[int]) -> float:
    """Upper-quartile distance of sampled opaque pixels from border colour.

    "Median-ish" per the module contract — a robust order
    statistic, not the max(): a lone outlier pixel (a lamp, a
    specular highlight, a compression artifact) must not pass the
    contrast gate on its own. The 75th percentile still requires a
    substantial region (~a quarter of the centre) to measurably
    differ from the border — a true median would flag small centred
    subjects that the estimator can in fact separate.
    """
    ds = []
    for y in range(bmp.height // 4, 3 * bmp.height // 4, 8):
        for x in range(bmp.width // 4, 3 * bmp.width // 4, 8):
            p = bmp.get(x, y)
            if p[3] < 128:
                continue  # transparent isn't foreground evidence
            ds.append(math.sqrt(sum(
                (p[i] - bg[i]) ** 2 for i in range(3))))
    if not ds:
        return 0.0
    ds.sort()
    return ds[int(len(ds) * 0.75)]


def assess(bmp: Bitmap) -> Dict:
    """Measure the frame; return flags + verdict + every number."""
    small = bmp.downscale(256)
    bg = _border_rgb(small)
    sharp = _sharpness(small)
    sd = _stddev(small)
    fg_bg = _fg_bg_dist(small, bg) if bg is not None else None
    flags: List[Dict] = []

    def flag(code: str, ok: bool, value: float, limit: float) -> None:
        flags.append({"code": code, "ok": ok,
                      "value": round(value, 2), "limit": limit})

    flag("size", min(small.width, small.height) >= MIN_DIM or
         min(bmp.width, bmp.height) >= MIN_DIM,
         min(bmp.width, bmp.height), MIN_DIM)
    flag("dynamic", sd >= MIN_STD, sd, MIN_STD)
    flag("blur", sharp >= MIN_SHARP, sharp, MIN_SHARP)
    if bg is None:
        # no opaque border — the bg reference itself is unmeasurable,
        # so contrast is reported unmeasurable, not failed or passed
        flags.append({"code": "contrast", "ok": False,
                      "value": None, "limit": MIN_FG_BG,
                      "unmeasurable": True})
    else:
        flag("contrast", fg_bg >= MIN_FG_BG, fg_bg, MIN_FG_BG)

    failed = [f["code"] for f in flags
              if not f["ok"] and not f.get("unmeasurable")]
    verdict = ("adequate" if not failed else
               "marginal" if len(failed) == 1 else
               "inadequate")
    return {
        "verdict": verdict,
        "failed": failed,
        "flags": flags,
        "state": "measured",
        "basis": "frame statistics vs minimum-evidence limits",
    }


def adequate(result: Dict) -> bool:
    return result["verdict"] != "inadequate"
