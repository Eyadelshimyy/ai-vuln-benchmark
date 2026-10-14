"""Plumb-line alignment — posture quality, not posture category.

Classical posture analysis drops a vertical line through the ear:
shoulder, hip and ankle should stack under it. Head forward of the
line is forward-head posture; pelvis forward is sway-back. Works on
the skeleton's x-coordinates at each height — a continuous quality
measure where pose classification only says "standing".
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .skeleton import Skeleton, observed_point

# landmarks on the plumb chain, top to bottom
_CHAIN = ("head", "neck", "chest", "pelvis")


def line(skel: Skeleton) -> Optional[dict]:
    """{x, joints, offsets} — plumb x = head/neck average (the
    reference); offsets = each joint's horizontal deviation in px
    and as a fraction of body height."""
    head = observed_point(skel, "head")
    neck = observed_point(skel, "neck")
    if not head or not neck:
        return None
    ref_x = (head[0] + neck[0]) / 2.0
    body_h = _span(skel)
    # a 0 span means no measurable body scale — `or 1.0` would emit
    # raw px dressed as body-height fractions (the rule
    # forward_head() already honours), so report None instead

    offsets: Dict[str, dict] = {}
    for name in _CHAIN:
        p = observed_point(skel, name)
        if not p:
            continue
        dx = p[0] - ref_x
        offsets[name] = {"dx": round(dx, 1),
                         "of_body_h": (round(dx / body_h, 3)
                                       if body_h > 0 else None)}
    # support reference: mean ankle x
    ankles = [p for p in (observed_point(skel, "ankle_l"), observed_point(skel, "ankle_r"))
              if p]
    if ankles:
        ax = sum(p[0] for p in ankles) / len(ankles)
        dx = ax - ref_x
        offsets["base"] = {"dx": round(dx, 1),
                           "of_body_h": (round(dx / body_h, 3)
                                         if body_h > 0 else None)}
    return {"plumb_x": round(ref_x, 1), "offsets": offsets}


def forward_head(skel: Skeleton) -> Optional[dict]:
    """Head ahead of the torso line — the tech-neck measure."""
    r = line(skel)
    if not r or "head" not in r["offsets"] or "chest" not in r["offsets"]:
        return None
    d = r["offsets"]["head"]["dx"] - r["offsets"]["chest"]["dx"]
    span = _span(skel)
    if span <= 0:
        # no usable body scale — a px threshold would be noise
        return {"dx": round(d, 1), "posture": "unknown"}
    return {"dx": round(d, 1),
            "posture": "forward_head" if abs(d) > 0.08 * span
            else "neutral"}


def _span(skel: Skeleton) -> float:
    """Head-to-lowest span, torso-length fallback, 0 when neither
    is measurable."""
    top = observed_point(skel, "head")
    lo = max((j.y for j in skel.joints.values()
              if j.state == "observed"), default=0.0)
    s = (lo - top[1]) if top else 0.0
    if s > 0:
        return s
    n, p = observed_point(skel, "neck"), observed_point(skel, "pelvis")
    return math.hypot(n[0] - p[0], n[1] - p[1]) if n and p else 0.0


def assess(skel: Skeleton) -> dict:
    """{plumb, forward_head, stack_score} — stack_score = mean
    |offset| of torso chain in body-h fractions (lower = better)."""
    r = line(skel)
    fh = forward_head(skel)
    out = {"plumb": r, "forward_head": fh, "stack_score": None}
    if r:
        vals = [abs(v["of_body_h"]) for k, v in r["offsets"].items()
                if k in _CHAIN and v["of_body_h"] is not None]
        if vals:
            out["stack_score"] = round(sum(vals) / len(vals), 3)
    return out
