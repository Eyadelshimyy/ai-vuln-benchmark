"""Horizon / camera-roll estimation — scene geometry cues.

If the person stands on level ground, the line through the two feet
is the ground line: its tilt equals the camera roll. The same line
also gives a soft depth hint — under a level camera the foot that
is lower in the frame is (usually) the nearer one.

The estimate is honest about its single assumption, carried in
`assumption`: the person stands upright on flat ground. A leaning
person on level ground and an upright person photographed with a
rolled camera are indistinguishable from one frame — the number is
the *scene* tilt under that assumption, not a camera measurement.

`horizon_y` places the camera-height horizon line on the image: the
horizon crosses the ground plane at the camera's own height, so it
sits `camera_height_cm * px_per_cm` above the ground line. Both the
170cm body-height scale and the 160cm default camera height are
stated assumptions, not measurements.

With one or no feet visible there is no ground line; `state` is
"unknown" and the numeric fields are None rather than guessed.
"""

from __future__ import annotations

import math
from typing import Optional

from .skeleton import Skeleton

# Preferred ground contacts: foot tips, then ankles.
_PAIRS = (("foot_l", "foot_r"), ("ankle_l", "ankle_r"))

# |dy| under this fraction of foot span counts as level.
_LEVEL_TOL = 0.02

# Rough adult standing height for the px->cm scale cue only.
_ASSUMED_HEIGHT_CM = 170.0


def estimate(skel: Skeleton,
             camera_height_cm: float = 160.0) -> dict:
    """Ground-line and camera-roll cues for one skeleton."""
    feet = None
    used = None
    for pair in _PAIRS:
        jl, jr = skel.joints.get(pair[0]), skel.joints.get(pair[1])
        # a predicted foot ("foot below ankle") is a guess, not a
        # ground contact — a ground line through it would measure
        # the prior's placement, not the scene
        if (jl and jr and jl.state == "observed"
                and jr.state == "observed"):
            feet, used = ((jl.x, jl.y), (jr.x, jr.y)), pair
            break
    if feet is None:
        return {
            "roll_deg": None,
            "foot_span_px": None,
            "dy_px": None,
            "depth_hint": "unknown",
            "ground_y": None,
            "px_per_cm": None,
            "horizon_y": None,
            "state": "unknown",
            "basis": "no pair of ground-contact joints present",
            "assumption": "person stands upright on level ground",
        }

    (lx, ly), (rx, ry) = feet
    dx, dy = rx - lx, ry - ly
    span = math.hypot(dx, dy)
    roll = math.degrees(math.atan2(dy, dx)) if span else 0.0
    if span <= 0 or abs(dy) < _LEVEL_TOL * span:
        hint = "level"
    else:
        # image y is down: the foot drawn lower is nearer under a
        # level camera — a hint, not a measurement.
        hint = "right_closer" if dy > 0 else "left_closer"

    ground_y = (ly + ry) / 2
    scale = _scale(skel, ground_y)
    horizon = (round(ground_y - camera_height_cm * scale, 1)
               if scale else None)
    return {
        "roll_deg": round(roll, 2),
        "foot_span_px": round(span, 1),
        "dy_px": round(dy, 1),
        "depth_hint": hint,
        "ground_y": round(ground_y, 1),
        "px_per_cm": scale,
        "horizon_y": horizon,
        "camera_height_cm": camera_height_cm,
        "state": "estimated",
        "basis": ("ground line through %s/%s" % used),
        "assumption": ("person stands upright on level ground; "
                       "height %.0fcm, camera %.0fcm"
                       % (_ASSUMED_HEIGHT_CM, camera_height_cm)),
        "feet_used": list(used),
    }


def _scale(skel: Skeleton, ground_y: float) -> Optional[float]:
    """px per cm from head-to-ground span over an assumed 170cm."""
    head = skel.point("head")
    if head is None:
        return None
    span = ground_y - head[1]
    if span <= 0:
        return None
    return round(span / _ASSUMED_HEIGHT_CM, 4)
