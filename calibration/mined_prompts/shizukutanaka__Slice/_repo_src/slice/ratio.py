"""Ratio Engine: anatomical proportions and balance from a skeleton."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from .skeleton import OBSERVED, Skeleton, head_length_px


def _dist(a: Optional[Tuple[float, float]],
          b: Optional[Tuple[float, float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _obs(skel: Skeleton, name: str) -> Optional[Tuple[float, float]]:
    """Point only when the joint is observed.

    Predicted limbs are placed at prior proportions (or mirrored),
    so ratios measured on them would report the prior itself as a
    measurement — e.g. a mirrored arm makes `limb_symmetry` read
    a structural 1.0."""
    j = skel.joints.get(name)
    return (j.x, j.y) if j and j.state == "observed" else None


def analyze(skel: Skeleton, *, centroid: Optional[Tuple[float, float]] = None
            ) -> dict:
    """Return normalized body ratios (all /body height) plus balance."""
    top = skel.point("head")
    feet = [p for p in (skel.point("foot_l"), skel.point("foot_r")) if p]
    if not top or not feet:
        return {"error": "insufficient joints"}
    body_h = _body_h(skel)
    if body_h <= 0:
        return {"error": "degenerate skeleton"}

    def norm(d):
        return round(d / body_h, 4) if d is not None else None

    head_px = _head_height(skel)
    ratios = {
        "body_height_px": round(body_h, 2),
        "head_to_body": round(body_h / head_px, 2) if head_px else None,
        "shoulder_width": norm(_dist(_obs(skel, "shoulder_l"),
                                     _obs(skel, "shoulder_r"))),
        "hip_width": norm(_dist(_obs(skel, "hip_l"),
                                _obs(skel, "hip_r"))),
        "torso_length": norm(abs((pelvis_p[1] - neck_p[1]))
                             if (pelvis_p := _obs(skel, "pelvis"))
                             and (neck_p := _obs(skel, "neck"))
                             else None),
        "arm_l": norm(_dist(_obs(skel, "shoulder_l"),
                            _obs(skel, "wrist_l"))),
        "arm_r": norm(_dist(_obs(skel, "shoulder_r"),
                            _obs(skel, "wrist_r"))),
        "leg_l": norm(_dist(_obs(skel, "hip_l"),
                            _obs(skel, "ankle_l"))),
        "leg_r": norm(_dist(_obs(skel, "hip_r"),
                            _obs(skel, "ankle_r"))),
        # fingertip-to-fingertip span — Vitruvian ≈ body height when the
        # arms are extended horizontally; drops for arms-down poses
        "arm_span": norm(_dist(_obs(skel, "wrist_l"),
                               _obs(skel, "wrist_r"))),
    }
    torso_px = (ratios["torso_length"] or 0) * body_h
    leg_px = [d for d in (_dist(_obs(skel, "hip_l"),
                               _obs(skel, "ankle_l")),
                        _dist(_obs(skel, "hip_r"),
                              _obs(skel, "ankle_r"))) if d]
    # 脚長/胴長 — proportion cue (typical adults ≈ 1.5-1.8)
    ratios["leg_to_torso"] = (round((sum(leg_px) / len(leg_px)) / torso_px, 3)
                             if leg_px and torso_px > 0 else None)
    sym = _symmetry(skel)
    ratios["limb_symmetry"] = round(sym, 3) if sym is not None else None
    if centroid:
        ratios["center_of_mass"] = {
            "x": round(centroid[0], 2), "y": round(centroid[1], 2),
            "y_ratio": round((centroid[1] - _top_y(skel)) / body_h, 3),
        }
    return ratios


def _body_h(skel: Skeleton) -> float:
    """Observed vertical extent — the normalizer for every ratio
    must come from evidence, not prior-placed endpoints. Predicted
    head/feet sit at prior positions and can stretch the frame
    beyond what was measured, scaling every ratio by inference."""
    js = [j for j in skel.joints.values() if j.state == OBSERVED]
    if not js:
        js = list(skel.joints.values())
    if not js:
        return 0.0
    return max(j.y for j in js) - min(j.y for j in js)


def _top_y(skel: Skeleton) -> float:
    ys = [j.y for j in skel.joints.values() if j.state == OBSERVED]
    if not ys:
        ys = [j.y for j in skel.joints.values()]
    return min(ys) if ys else 0.0


def _head_height(skel: Skeleton) -> Optional[float]:
    head = skel.get("head")
    if not head:
        return None
    d = head_length_px(skel)
    if d is not None:
        return d
    return head.y - _top_y(skel) + (head.y - _top_y(skel))


def _symmetry(skel: Skeleton) -> Optional[float]:
    pairs = [("arm_l", "arm_r"), ("leg_l", "leg_r")]
    vals = []
    for a, b in pairs:
        la = _limb_len(skel, a)
        lb = _limb_len(skel, b)
        if la and lb and max(la, lb) > 0:
            vals.append(1 - abs(la - lb) / max(la, lb))
    return sum(vals) / len(vals) if vals else None


def _limb_len(skel: Skeleton, name: str) -> Optional[float]:
    if name.startswith("arm"):
        s = "shoulder_" + name[-1]
        return _dist(_obs(skel, s), _obs(skel, "wrist_" + name[-1]))
    s = "hip_" + name[-1]
    return _dist(_obs(skel, s), _obs(skel, "ankle_" + name[-1]))
