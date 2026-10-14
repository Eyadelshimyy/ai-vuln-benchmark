"""Implied dynamics — does the still look mid-motion?

A camera freezes motion; body geometry keeps the signature. A leg
off its stance axis, arms far from the torso line, the body's
balance point outside the feet — each is a cue that the frame
caught movement rather than rest. The score is a weighted vote of
observed cues, labelled `implied`, never "the person is moving".
"""

from __future__ import annotations

import math
from typing import List

from .skeleton import Skeleton, observed_body_span, observed_point


def _d(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def cues(skel: Skeleton) -> List[dict]:
    """[{cue, weight, detail}] — all on observed joints."""
    out: List[dict] = []
    pelvis = observed_point(skel, "pelvis")
    for side in ("l", "r"):
        hip = observed_point(skel, f"hip_{side}")
        ankle = observed_point(skel, f"ankle_{side}")
        if hip and ankle:
            leg = _d(hip, ankle) or 1.0
            off = abs(ankle[0] - hip[0]) / leg
            if off > 0.3:
                out.append({"cue": f"leg_off_axis_{side}",
                            "weight": 0.25,
                            "detail": round(off, 2)})
        shoulder = observed_point(skel, f"shoulder_{side}")
        wrist = observed_point(skel, f"wrist_{side}")
        if shoulder and wrist and pelvis:
            # wrist far laterally from torso → arm swing
            far = abs(wrist[0] - pelvis[0]) > 0.6 * _d(shoulder, wrist)
            if far:
                out.append({"cue": f"arm_out_{side}",
                            "weight": 0.15, "detail": None})

    al, ar = observed_point(skel, "ankle_l"), observed_point(skel, "ankle_r")
    if pelvis and al and ar:
        mid = (al[0] + ar[0]) / 2.0
        span = abs(ar[0] - al[0]) or 1.0
        if abs(pelvis[0] - mid) > 0.5 * span:
            out.append({"cue": "com_outside_feet", "weight": 0.35,
                        "detail": round(abs(pelvis[0] - mid) / span, 2)})
        body_h = _span(skel)
        # "wide" is a fraction of body height — with no honest
        # scale a raw-pixel default fabricates the threshold.
        if body_h > 0 and span > 0.5 * body_h:
            out.append({"cue": "wide_step", "weight": 0.2,
                        "detail": round(span / body_h, 2)})
    return out


def _span(skel: Skeleton) -> float:
    # observed-only span: a predicted head/foot is prior fill
    return observed_body_span(skel)


def score(skel: Skeleton) -> dict:
    """{score 0-1, verdict, cues} — score = Σweights capped at 1."""
    c = cues(skel)
    s = min(1.0, sum(x["weight"] for x in c))
    verdict = ("dynamic" if s >= 0.6 else
               "possibly_dynamic" if s >= 0.3 else "static")
    return {"score": round(s, 2), "verdict": verdict,
            "state": "implied", "cues": c}
