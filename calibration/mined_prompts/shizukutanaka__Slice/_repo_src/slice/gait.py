"""Gait phase cues — per-leg stance/swing estimate from one frame.

A still image can't prove locomotion, but leg geometry says which
leg would carry load: a near-vertical, extended leg under the hip
is a stance candidate; a flexed knee with the ankle off the ground
line reads as swing. Output is per-leg "stance|swing|unknown" —
a phase *cue*, not a claim the person is walking.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from .skeleton import Skeleton, observed_point

Point = Tuple[float, float]


def _d(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _knee_angle(hip: Point, knee: Point, ankle: Point) -> float:
    v1 = (hip[0] - knee[0], hip[1] - knee[1])
    v2 = (ankle[0] - knee[0], ankle[1] - knee[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def leg_phase(skel: Skeleton, side: str) -> dict:
    """{phase, knee_angle, reasons} for leg_l/leg_r."""
    hip = observed_point(skel, f"hip_{side}")
    knee = observed_point(skel, f"knee_{side}")
    ankle = observed_point(skel, f"ankle_{side}")
    out = {"phase": "unknown", "knee_angle": None, "reasons": []}
    if not hip or not ankle:
        out["reasons"].append("missing_joints")
        return out

    leg_len = _d(hip, knee) + _d(knee, ankle) if knee else _d(hip, ankle)
    reach = _d(hip, ankle)
    if leg_len <= 0:
        # coincident joints carry no phase evidence — `or 1.0` on
        # reach would let a zero-length leg "pass" the extension
        # test and report a fabricated swing/stance verdict
        out["reasons"].append("degenerate_leg")
        return out
    ka = _knee_angle(hip, knee, ankle) if knee else 180.0
    out["knee_angle"] = round(ka, 1)

    vertical = abs(ankle[0] - hip[0]) < 0.25 * leg_len
    extended = ka > 150 and reach > 0.85 * leg_len
    if knee is None:
        out["reasons"].append("knee_missing")
        out["phase"] = "stance" if vertical else "unknown"
        return out
    if extended and vertical:
        out["phase"] = "stance"
        out["reasons"] += ["extended", "under_hip"]
    elif ka < 150:
        out["phase"] = "swing"
        out["reasons"].append("knee_flexed")
    elif not vertical:
        out["phase"] = "swing"
        out["reasons"].append("off_axis")
    else:
        out["reasons"].append("inconclusive")
    return out


def assess(skel: Skeleton) -> dict:
    """{legs, step_width, double_support} — step width = horizontal
    ankle separation vs hip width (walking widens it)."""
    legs: Dict[str, dict] = {}
    for side in ("l", "r"):
        legs[side] = leg_phase(skel, side)
    al, ar = observed_point(skel, "ankle_l"), observed_point(skel, "ankle_r")
    hl, hr = observed_point(skel, "hip_l"), observed_point(skel, "hip_r")
    step = None
    if al and ar:
        hw = abs(hr[0] - hl[0]) if hl and hr else None
        step = {"ankle_gap": round(abs(ar[0] - al[0]), 1),
                "vs_hip_width": round(abs(ar[0] - al[0]) / hw, 2) if hw else None}
    phases = [l["phase"] for l in legs.values()]
    return {"legs": legs, "step_width": step,
            "double_support": phases == ["stance", "stance"]}
