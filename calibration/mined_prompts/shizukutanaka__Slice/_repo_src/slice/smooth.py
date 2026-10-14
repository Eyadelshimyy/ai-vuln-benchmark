"""Temporal smoothing of joint tracks across a frame series.

Per-frame estimates jitter — a wrist jumps a few pixels frame to
frame even when the person is still. `smooth` applies a symmetric
moving average over each joint's track, cutting jitter while
keeping real motion. Missing frames break nothing: a joint absent
from a window's frames just averages what exists, and frames where
the joint is missing stay missing — smoothing never invents a
position. And an observed centre averages only observed neighbours:
predicted guesses in the window must not pull a measured joint
toward a fabricated location.
"""

from __future__ import annotations

from typing import List

from .skeleton import OBSERVED, Joint, Skeleton


def smooth(series: List[Skeleton], radius: int = 1) -> List[Skeleton]:
    """Return per-frame skeletons with positions averaged over
    [t-radius, t+radius]. Confidence and state are taken from the
    center frame — only position is smoothed."""
    n = len(series)
    out: List[Skeleton] = []
    for t in range(n):
        src = series[t]
        dst = Skeleton(src.image_width, src.image_height)
        dst.body_model = dict(src.body_model)
        dst.orientation = dict(src.orientation)
        lo, hi = max(0, t - radius), min(n, t + radius + 1)
        for name, j in src.joints.items():
            xs, ys, cnt = 0.0, 0.0, 0
            for tt in range(lo, hi):
                o = series[tt].joints.get(name)
                if o is None:
                    continue
                # an observed centre averages only observed
                # neighbours — predicted guesses must not pull a
                # measured joint toward a fabricated location
                if j.state == OBSERVED and o.state != OBSERVED:
                    continue
                xs += o.x
                ys += o.y
                cnt += 1
            dst.set(Joint(name, xs / cnt, ys / cnt, j.confidence,
                          state=j.state, basis=j.basis))
        out.append(dst)
    return out


def jitter(series: List[Skeleton], joint: str = "pelvis") -> float:
    """Mean frame-to-frame displacement of a joint — a jitter metric."""
    steps = []
    for t in range(1, len(series)):
        a = series[t - 1].joints.get(joint)
        b = series[t].joints.get(joint)
        if a and b:
            steps.append(((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5)
    return round(sum(steps) / len(steps), 3) if steps else 0.0
