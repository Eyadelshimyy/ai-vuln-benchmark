"""Frame-to-frame motion — per-joint displacement and summary.

Compares two skeletons (e.g. consecutive video frames or a walk
cycle) and reports where the figure moved: per-joint vectors, the
fastest-moving parts, and overall translation. Sequence analysis —
walk detection, gesture deltas, animation retarget checks — starts
here rather than at single-image pose.
"""

from __future__ import annotations

import math
from typing import Dict

from .skeleton import Skeleton


def _scale_into(a: Skeleton, b: Skeleton):
    """(sx, sy, changed) mapping b's coordinate space onto a's.

    Two skeletons measured in different frames (e.g. a re-encoded or
    re-cropped frame, or a doc produced on the downscaled working
    space) must not be differenced raw — a 1000px frame's joint minus
    a 300px frame's joint is not motion. Scale b's coordinates into
    a's frame and disclose that we did."""
    aw, ah = a.image_width or 0, a.image_height or 0
    bw, bh = b.image_width or 0, b.image_height or 0
    if not (aw and ah and bw and bh) or (aw, ah) == (bw, bh):
        return 1.0, 1.0, False
    return aw / bw, ah / bh, True


def vectors(a: Skeleton, b: Skeleton,
            min_confidence: float = 0.0) -> Dict[str, tuple]:
    """{joint: (dx, dy, speed)} for joints OBSERVED in both frames.

    A predicted joint is prior fill: its "displacement" between
    frames is movement of the prior, not of the person — a
    fabricated measurement, so it is excluded like a missing one.
    When the two skeletons declare different image sizes, b's
    coordinates are first rescaled into a's frame — the displacement
    then assumes the same scene seen at two resolutions."""
    sx, sy, _ = _scale_into(a, b)
    out: Dict[str, tuple] = {}
    for name, ja in a.joints.items():
        jb = b.joints.get(name)
        if not jb:
            continue
        if ja.state != "observed" or jb.state != "observed":
            continue
        if min(ja.confidence, jb.confidence) < min_confidence:
            continue
        dx, dy = jb.x * sx - ja.x, jb.y * sy - ja.y
        out[name] = (dx, dy, round(math.hypot(dx, dy), 2))
    return out


def summarize(a: Skeleton, b: Skeleton,
              min_confidence: float = 0.0) -> dict:
    """Aggregate motion: translation, fastest joints, per-limb spread."""
    _, _, scaled = _scale_into(a, b)
    v = vectors(a, b, min_confidence)
    if not v:
        return {"joints": 0, "mean_speed": 0.0, "translation": (0, 0),
                "fastest": None, "by_part": {},
                "frame_scaled": scaled,
                "frame_b": (b.image_width, b.image_height)
                           if scaled else None}
    dxs = [vv[0] for vv in v.values()]
    dys = [vv[1] for vv in v.values()]
    speeds = {n: vv[2] for n, vv in v.items()}
    fastest = max(speeds, key=speeds.get)
    by_part: Dict[str, float] = {}
    for name, (_, _, s) in v.items():
        part = name.split("_")[0]
        by_part[part] = max(by_part.get(part, 0.0), s)
    return {
        "joints": len(v),
        "frame_scaled": scaled,
        "frame_b": (b.image_width, b.image_height) if scaled else None,
        "mean_speed": round(sum(speeds.values()) / len(speeds), 2),
        "translation": (round(sum(dxs) / len(dxs), 1),
                        round(sum(dys) / len(dys), 1)),
        "fastest": {"joint": fastest, "speed": speeds[fastest]},
        "by_part": {p: round(s, 2) for p, s in sorted(by_part.items())},
    }


def shifted(skel: Skeleton, dx: float, dy: float) -> Skeleton:
    """Return a copy of the skeleton translated by (dx, dy)."""
    from .skeleton import Joint
    out = Skeleton(skel.image_width, skel.image_height)
    out.body_model = dict(skel.body_model)
    out.orientation = dict(skel.orientation)
    for n, j in skel.joints.items():
        out.set(Joint(n, j.x + dx, j.y + dy, j.confidence,
                      state=j.state, basis=j.basis))
    return out
