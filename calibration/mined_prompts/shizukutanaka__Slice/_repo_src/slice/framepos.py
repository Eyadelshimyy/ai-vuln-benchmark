"""Frame position — where the figure sits in the image.

Composition is knowledge too: headroom above the figure's topmost
point says whether
the framing is portrait-tight or environment-wide; horizontal
offset says centered subject or candid edge framing; body fraction
of frame says how much scene is included. All derived from the
joint cloud bounds vs the image dimensions.
"""

from __future__ import annotations

from typing import Optional

from .skeleton import Skeleton


def bounds(skel: Skeleton) -> Optional[dict]:
    """Joint-cloud bounding box — OBSERVED joints only.

    Predicted joints are prior fill (a "foot below ankle" guess,
    a mirror copy): including them stretches the cloud to guessed
    geometry and fabricates headroom/footroom/side_gap."""
    pts = [(j.x, j.y) for j in skel.joints.values()
           if j.state == "observed"]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return {"x0": min(xs), "y0": min(ys),
            "x1": max(xs), "y1": max(ys)}


def analyze(skel: Skeleton, frame_w: int, frame_h: int) -> Optional[dict]:
    """{headroom, footroom, side_gap, center_offset, body_fraction,
    thirds_zone} — all fractions of frame dims."""
    b = bounds(skel)
    if not b or not frame_w or not frame_h:
        return None
    hj = skel.joints.get("head")
    head = (hj.x, hj.y) if hj and hj.state == "observed" else None
    bw = b["x1"] - b["x0"]
    # a 0-height cloud has zero area — `or 1.0` would fabricate a
    # 1px row and inflate body_fraction by a whole pixel column
    bh = b["y1"] - b["y0"]

    cx = (b["x0"] + b["x1"]) / 2.0
    cy = (b["y0"] + b["y1"]) / 2.0

    third_w, third_h = frame_w / 3.0, frame_h / 3.0
    col = "l" if cx < third_w else ("r" if cx > 2 * third_w else "c")
    row = "t" if cy < third_h else ("b" if cy > 2 * third_h else "m")

    return {"headroom": round(b["y0"] / frame_h, 3),
            "footroom": round((frame_h - b["y1"]) / frame_h, 3),
            "side_gap": round(min(b["x0"], frame_w - b["x1"]) / frame_w, 3),
            "center_offset": round((cx - frame_w / 2.0) / frame_w, 3),
            "body_fraction": round(bw * bh / (frame_w * frame_h), 3),
            "thirds_zone": f"{row}{col}"}


def framing(skel: Skeleton, frame_w: int, frame_h: int) -> Optional[str]:
    """tight|portrait|wide — headline label."""
    r = analyze(skel, frame_w, frame_h)
    if not r:
        return None
    if r["body_fraction"] > 0.55 or r["headroom"] < 0.05:
        return "tight"
    if r["headroom"] > 0.3:
        return "wide"
    return "portrait"
