"""Person candidates — every foreground component, ranked.

`estimate` looks at the single largest connected component and
assumes it is the person. `candidates` enumerates ALL components of
the foreground mask and ranks them, so a frame with two people (or a
person plus a prop) stops pretending there is only one blob.

Honesty note: a component is not a person — two overlapping people
merge into one blob, and a detached hat becomes a fake third. The
output says "candidate components", never "people detected".
"""

from __future__ import annotations

from typing import List

from .bitmap import Bitmap
from .pose import HeuristicPoseEstimator

_MIN_FRAC = 0.005


def _components(labels: List[list], sizes: dict,
                w: int, h: int) -> List[dict]:
    """Per-label size+bbox from a labelled mask (the estimator's own
    component labelling supplies `labels`/`sizes`; only the box pass
    is local)."""
    boxes: dict = {}
    for y in range(h):
        row = labels[y]
        for x in range(w):
            lab = row[x]
            if not lab:
                continue
            bb = boxes.setdefault(lab, [x, y, x, y])
            if x < bb[0]:
                bb[0] = x
            if y < bb[1]:
                bb[1] = y
            if x > bb[2]:
                bb[2] = x
            if y > bb[3]:
                bb[3] = y
    return [{"size": sizes[lab],
             "bbox": (bb[0], bb[1],
                      bb[2] - bb[0] + 1, bb[3] - bb[1] + 1)}
            for lab, bb in boxes.items()]


def candidates(bmp: Bitmap, est: HeuristicPoseEstimator = None,
               min_frac: float = _MIN_FRAC,
               top_k: int = 8) -> dict:
    """Rank foreground components as person candidates."""
    est = est or HeuristicPoseEstimator()
    small = bmp.downscale(est.max_dim)
    w, h = small.width, small.height
    mask = est._mask(small)
    frame = w * h
    labels, sizes = est._label_components(mask, w, h)
    comps = _components(labels, sizes, w, h)
    out = []
    for i, c in enumerate(sorted(comps, key=lambda c: -c["size"])):
        if c["size"] < frame * min_frac:
            continue
        lx, ty, bw, bh = c["bbox"]
        out.append({
            "rank": len(out),
            "size_px": c["size"],
            "fraction": round(c["size"] / frame, 4),
            "bbox": {"x": lx, "y": ty, "w": bw, "h": bh},
            "aspect": round(bw / max(1, bh), 3),
            "touches_edge": (lx == 0 or ty == 0
                             or lx + bw == w or ty + bh == h),
        })
        if len(out) >= top_k:
            break
    return {
        "n_components": len(comps),
        "n_candidates": len(out),
        "candidates": out,
        "dominant_share": (round(out[0]["size_px"] / frame, 4)
                           if out else 0.0),
        "note": "components are candidates, not confirmed people — "
                "overlap merges, detached parts inflate the count",
        "basis": "est._mask + 4-conn components @%dx%d" % (w, h),
    }
