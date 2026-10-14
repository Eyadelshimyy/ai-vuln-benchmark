"""Self-contact — observed joints resting near each other.

Gestures name limb configurations; contact names what touches.
Hands meeting (clap/prayer), a hand at the head (thinking,
scratching), a hand at the opposite shoulder (arms crossed) — all
read as observed joint pairs within a fraction of body height.
Only `observed` joints count: predicted positions are guesses and
a guess touching a guess proves nothing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .skeleton import Skeleton

# pairs worth naming, and what near means (fraction of body height)
_RULES: Tuple[Tuple[str, str, str, float], ...] = (
    ("wrist_l", "wrist_r", "hands_together", 0.10),
    ("wrist_l", "head", "hand_at_head", 0.14),
    ("wrist_r", "head", "hand_at_head", 0.14),
    ("wrist_l", "elbow_r", "arms_crossed", 0.18),
    ("wrist_r", "elbow_l", "arms_crossed", 0.18),
    ("wrist_l", "shoulder_r", "arms_crossed", 0.16),
    ("wrist_r", "shoulder_l", "arms_crossed", 0.16),
    ("wrist_l", "neck", "hand_at_head", 0.12),
    ("wrist_r", "neck", "hand_at_head", 0.12),
    ("ankle_l", "ankle_r", "feet_together", 0.08),
    ("wrist_l", "wrist_r", "hands_together", 0.10),
)


def _body_h(skel: Skeleton) -> float:
    from .skeleton import observed_body_span
    return observed_body_span(skel)


def detect(skel: Skeleton) -> List[dict]:
    """[{pair, contact, distance, threshold_px}] — observed only.

    Returns [] when no honest body scale exists: "near" is defined
    as a fraction of body height, so a raw-pixel default would
    fabricate the very threshold the rule measures against."""
    seen: Dict[Tuple[str, str], bool] = {}
    h = _body_h(skel)
    if h <= 0:
        return []
    found: List[dict] = []
    for a, b, name, frac in _RULES:
        if (a, b) in seen:
            continue
        ja, jb = skel.joints.get(a), skel.joints.get(b)
        if not ja or not jb:
            continue
        if ja.state != "observed" or jb.state != "observed":
            continue
        d = math.hypot(jb.x - ja.x, jb.y - ja.y)
        if d <= frac * h:
            found.append({"pair": [a, b], "contact": name,
                          "distance": round(d, 1),
                          "threshold_px": round(frac * h, 1)})
        seen[(a, b)] = True
    return found


def summary(skel: Skeleton) -> dict:
    """{contacts, count} for knowledge output."""
    c = detect(skel)
    return {"contacts": [x["contact"] for x in c], "count": len(c)}
