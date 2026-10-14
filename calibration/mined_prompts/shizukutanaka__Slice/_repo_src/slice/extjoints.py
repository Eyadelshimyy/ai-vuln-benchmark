"""Extended landmark vocabulary (v2 joint set) — AUDIT P2-15.

The v1 `JOINTS` list stays frozen: stored documents and every consumer
depend on it. This module defines additional landmarks that are
*derived* from the v1 skeleton rather than measured — midpoints,
hand/foot terminations, and facial anchors that current evidence
cannot see.

Honesty contract, stricter than v1: every extended joint is
`state="predicted"` with a basis in the shared vocabulary
(`interpolated` / `prior`), because no silhouette feature currently
supports them. The day a detector can see a fingertip, the basis
moves to the observation vocabulary — the joint name stays.

    DERIVED  — placed between/around existing joints:
        mid_hip        pelvis-line midpoint (hip_l..hip_r)
        waist          chest..pelvis midpoint
        mid_shoulder   shoulder_l..shoulder_r midpoint
        mid_thigh_l/r  hip..knee midpoint
        mid_shin_l/r   knee..ankle midpoint
        fingertip_l/r  wrist + forearm direction x hand_length
        toe_l/r        foot + leg direction x foot_length
        heel_l/r       ankle mirrored across foot
    ANCHORED_BUT_INVISIBLE — vocabulary reserved, never emitted:
        nose, eye_l/r, ear_l/r — no cue exists in a silhouette; listed
        so consumers know the names are claimed, not missing by bug.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .skeleton import Joint, PREDICTED, Skeleton

DERIVED: List[str] = [
    "mid_hip", "waist", "mid_shoulder",
    "mid_thigh_l", "mid_thigh_r", "mid_shin_l", "mid_shin_r",
    "fingertip_l", "fingertip_r", "toe_l", "toe_r",
    "heel_l", "heel_r",
]

RESERVED: List[str] = ["nose", "eye_l", "eye_r", "ear_l", "ear_r"]

HAND_LEN_RATIO = 0.18   # hand ≈ 0.75 x forearm anatomically; we use a
                        # conservative fraction of forearm length
TOE_LEN_RATIO = 0.25    # toes ≈ quarter of the shin


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _put(out: Dict[str, Joint], name: str, pt, conf: float,
         basis: str):
    out[name] = Joint(name, round(pt[0], 3), round(pt[1], 3),
                      round(max(0.0, min(1.0, conf)), 3),
                      PREDICTED, basis)


def derive(skel: Skeleton,
           min_confidence: float = 0.0) -> Dict[str, Joint]:
    """Extended joints derivable from this skeleton.

    Both anchors must exist; each joint inherits the weaker of the
    two anchors' confidence (x0.8 for being a derivation). Points that
    are themselves predicted still count — the basis says which.
    """
    out: Dict[str, Joint] = {}

    def mid(name, a, b, conf_scale=0.8):
        ja, jb = skel.joints.get(a), skel.joints.get(b)
        if not ja or not jb:
            return
        if min(ja.confidence, jb.confidence) < min_confidence:
            return
        conf = min(ja.confidence, jb.confidence) * conf_scale
        _put(out, name, _mid((ja.x, ja.y), (jb.x, jb.y)),
             conf, f"interpolated {a}-{b}")

    mid("mid_hip", "hip_l", "hip_r")
    mid("waist", "chest", "pelvis")
    mid("mid_shoulder", "shoulder_l", "shoulder_r")
    for side in ("l", "r"):
        mid(f"mid_thigh_{side}", f"hip_{side}", f"knee_{side}")
        mid(f"mid_shin_{side}", f"knee_{side}", f"ankle_{side}")

        # fingertip: wrist pushed past itself along elbow->wrist
        e, w = skel.joints.get(f"elbow_{side}"), skel.joints.get(
            f"wrist_{side}")
        if e and w and min(e.confidence, w.confidence) >= min_confidence:
            dx, dy = w.x - e.x, w.y - e.y
            L = math.hypot(dx, dy)
            if L > 1e-6:
                hx, hy = dx * HAND_LEN_RATIO, dy * HAND_LEN_RATIO
                _put(out, f"fingertip_{side}", (w.x + hx, w.y + hy),
                     min(e.confidence, w.confidence) * 0.5,
                     f"prior off wrist_{side} (forearm direction)")

        # toe: foot pushed along ankle->foot; heel mirrored the other way
        a, f = skel.joints.get(f"ankle_{side}"), skel.joints.get(
            f"foot_{side}")
        if a and f and min(a.confidence, f.confidence) >= min_confidence:
            dx, dy = f.x - a.x, f.y - a.y
            L = math.hypot(dx, dy)
            if L > 1e-6:
                ux, uy = dx / L, dy / L
                seg = L
                _put(out, f"toe_{side}",
                     (f.x + ux * seg * TOE_LEN_RATIO,
                      f.y + uy * seg * TOE_LEN_RATIO),
                     min(a.confidence, f.confidence) * 0.5,
                     f"prior off foot_{side} (foot direction)")
                _put(out, f"heel_{side}",
                     (a.x - ux * seg * TOE_LEN_RATIO,
                      a.y - uy * seg * TOE_LEN_RATIO),
                     min(a.confidence, f.confidence) * 0.4,
                     f"prior off ankle_{side} (mirrored toe axis)")
    return out


def vocabulary() -> Dict[str, dict]:
    """The full v2 claim map: derivable now vs reserved-invisible."""
    return {
        "derived": {n: "interpolated/prior from v1 joints"
                    for n in DERIVED},
        "reserved": {n: "vocabulary claimed; no silhouette cue exists — "
                     "never emitted until evidence lands"
                     for n in RESERVED},
        "contract": "every v2 joint is predicted; basis discloses how",
    }
