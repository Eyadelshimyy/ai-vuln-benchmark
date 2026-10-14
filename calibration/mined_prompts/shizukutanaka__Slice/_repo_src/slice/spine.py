"""Spine curve — lateral bow and forward lean of the torso chain.

The plumb line measures global alignment; the spine curve measures
the chain itself: chest deflection off the neck–pelvis chord is a
lateral bow (scoliosis-like), and the chord's angle off vertical
is forward lean (kyphosis-like). Three observed points make a
two-segment polyline whose geometry is honest to report.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from .skeleton import Skeleton

Point = Tuple[float, float]


def _d(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def curve(skel: Skeleton) -> Optional[dict]:
    """{lateral_deflection, lean_deg, chord_len, state}.

    Only an observed torso chain is measured: a predicted chest is
    placed on the neck–pelvis chord by the prior, so its "curve"
    would always read straight — a fabricated measurement."""
    neck = skel.point("neck")
    chest = skel.point("chest")
    pelvis = skel.point("pelvis")
    if not neck or not chest or not pelvis:
        return None
    for n in ("neck", "chest", "pelvis"):
        if skel.joints[n].state != "observed":
            return None

    chord = _d(neck, pelvis)
    chain = _d(neck, chest) + _d(chest, pelvis)
    # degenerate neck≈pelvis: of_chord is unmeasurable, not a raw-px
    # fraction reported against a 1px denominator
    body_h = chord

    # signed lateral deflection: distance of chest from chord line
    ax, ay = pelvis[0] - neck[0], pelvis[1] - neck[1]
    t = 0.0 if chord < 1e-9 else (
        ((chest[0] - neck[0]) * ax + (chest[1] - neck[1]) * ay)
        / (chord * chord))
    cx = neck[0] + t * ax
    cy = neck[1] + t * ay
    lat = _d(chest, (cx, cy))
    sign = 1.0 if (chest[0] - cx) * ay - (chest[1] - cy) * ax > 0 else -1.0

    # lean: chord angle off vertical (image y is down, so vertical
    # chord has |ax|≈0)
    lean = math.degrees(math.atan2(abs(ax), abs(ay))) if chord > 1e-9 else 0.0
    return {"lateral_deflection": round(sign * lat, 1),
            "of_chord": round(lat / body_h, 3) if body_h > 1e-9 else None,
            "lean_deg": round(lean, 1),
            "chord_len": round(chord, 1),
            "curvature": round(chain / chord, 3) if chord > 1e-9 else None,
            "state": "measured"}


def classify(skel: Skeleton) -> dict:
    """{bow, lean} — bow: straight|lateral, lean: upright|leaning."""
    c = curve(skel)
    out = {"bow": "unknown", "lean": "unknown"}
    if not c:
        return out
    out["bow"] = "lateral" if abs(c["of_chord"]) > 0.06 else "straight"
    out["lean"] = "leaning" if c["lean_deg"] > 20 else "upright"
    return out
