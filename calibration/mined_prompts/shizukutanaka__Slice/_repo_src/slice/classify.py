"""Pose classification from skeleton geometry (Phase 2 item).

Rule-based, evidence-first: each class is asserted only when the
observed joints support it, and confidence is scaled down by how much
of the skeleton was predicted rather than seen. When the geometry does
not discriminate we return "不明" instead of forcing a label — the same
honesty contract as observed/predicted.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from .skeleton import OBSERVED, Skeleton

Point = Tuple[float, float]

LABELS = {
    "stand": "立つ", "sit": "座る", "walk": "歩く", "run": "走る",
    "lie": "寝る", "crouch": "しゃがむ", "bend": "前傾",
    "invert": "逆さま", "unknown": "不明",
}


def _d(a: Optional[Point], b: Optional[Point]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _observed_ratio(skel: Skeleton) -> float:
    if not skel.joints:
        return 0.0
    obs = sum(1 for j in skel.joints.values() if j.state == OBSERVED)
    return obs / len(skel.joints)


def analyze(skel: Skeleton) -> dict:
    """Return {pose, label, confidence, signals} for the skeleton."""
    J = {n: (j.x, j.y, j.state) for n, j in skel.joints.items()}

    def pt(name):
        v = J.get(name)
        return (v[0], v[1]) if v else None

    head, ankle_l, ankle_r = pt("head"), pt("ankle_l"), pt("ankle_r")
    hip_l, hip_r = pt("hip_l"), pt("hip_r")
    knee_l, knee_r = pt("knee_l"), pt("knee_r")
    feet = [p for p in (pt("foot_l"), pt("foot_r"), ankle_l, ankle_r) if p]
    if not head or not feet:
        return {"pose": "unknown", "label": LABELS["unknown"],
                "confidence": 0.0, "signals": {"reason": "insufficient joints"}}

    top_y = min(j.y for j in skel.joints.values())
    # The lowest joint, not the lowest foot: for an inverted figure
    # the feet are the top, and a foot-only bottom collapses the
    # vertical span to ~0 (which then masquerades as "lie").
    bot_y = max(j.y for j in skel.joints.values())
    span_y = bot_y - top_y
    span_x = max(j.x for j in skel.joints.values()) \
        - min(j.x for j in skel.joints.values())

    thigh_l = _d(hip_l, knee_l)
    thigh_r = _d(hip_r, knee_r)
    shin_l = _d(knee_l, ankle_l)
    shin_r = _d(knee_r, ankle_r)
    leg_l = _d(hip_l, ankle_l)
    leg_r = _d(hip_r, ankle_r)
    hip_w = _d(hip_l, hip_r)
    ankle_w = _d(ankle_l, ankle_r)

    signals = {
        "span_y": round(span_y, 1), "span_x": round(span_x, 1),
        "leg_l": round(leg_l, 1) if leg_l else None,
        "leg_r": round(leg_r, 1) if leg_r else None,
        "ankle_width": round(ankle_w, 1) if ankle_w else None,
    }

    def result(pose: str, conf: float) -> dict:
        conf = round(min(conf, 0.95) * (0.4 + 0.6 * _observed_ratio(skel)), 3)
        return {"pose": pose, "label": LABELS[pose],
                "confidence": conf, "signals": signals}

    # 寝る: the figure's long axis is horizontal.
    if span_x > span_y * 1.15:
        return result("lie", 0.8)

    # 逆さま: every foot is measured ABOVE the head — the figure is
    # upside down (handstand, head-down lie, inverted capture), not
    # standing. After "lie" so a diagonal head-low recline still
    # classifies as lying.
    if max(p[1] for p in feet) < head[1]:
        return result("invert", 0.8)

    if not all((hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r)):
        return result("unknown", 0.3)

    def thigh_horizontal(hip, knee, tlen):
        return hip and knee and tlen and abs(hip[1] - knee[1]) < tlen * 0.45

    horiz = [thigh_horizontal(h, k, t) for h, k, t in
             ((hip_l, knee_l, thigh_l), (hip_r, knee_r, thigh_r))]

    # 座る: thighs roughly horizontal AND shins drop near-vertical
    # (chair-like). A squat also levels the thighs but angles the shins,
    # so it falls through to the crouch rule below.
    def shin_vertical(knee, ankle, slen):
        return knee and ankle and slen \
            and abs(knee[0] - ankle[0]) < slen * 0.2

    if all(horiz) and pt("neck") \
            and pt("neck")[1] < min(hip_l[1], hip_r[1]) \
            and all(shin_vertical(k, a, s) for k, a, s in
                    ((knee_l, ankle_l, shin_l), (knee_r, ankle_r, shin_r))):
        return result("sit", 0.75)

    # しゃがむ: BOTH legs folded — straight hip->ankle much shorter than
    # the knee-detoured chain. One folded leg is gait, not a crouch.
    fold_l = leg_l / (thigh_l + shin_l) if thigh_l and shin_l else None
    fold_r = leg_r / (thigh_r + shin_r) if thigh_r and shin_r else None
    if fold_l is not None and fold_r is not None:
        signals["leg_fold_l"] = round(fold_l, 3)
        signals["leg_fold_r"] = round(fold_r, 3)
        if max(fold_l, fold_r) < 0.7:
            return result("crouch", 0.7)

    # 前傾（お辞儀/屈み）: legs upright but the torso axis leans well
    # off vertical. After sit/crouch — a seated lean is still sitting.
    neck, pelvis = pt("neck"), pt("pelvis")
    torso_len = _d(neck, pelvis)
    if torso_len:
        tilt = abs(neck[0] - pelvis[0]) / torso_len
        signals["torso_tilt"] = round(tilt, 3)
        if tilt > 0.45:
            return result("bend", 0.6)

    # 歩く/走る: clear left-right leg separation.
    if ankle_w and hip_w and ankle_w > hip_w * 1.6:
        knee_gap = abs((knee_l[1] if knee_l else 0)
                       - (knee_r[1] if knee_r else 0))
        shin = max(s for s in (shin_l, shin_r) if s)
        if shin and knee_gap > shin * 0.6:
            return result("run", 0.6)
        return result("walk", 0.6)

    return result("stand", 0.65)
