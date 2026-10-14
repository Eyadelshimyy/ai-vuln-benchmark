"""Weight dominance — which leg carries the load.

Contrapposto is visible in a single frame: the pelvis shifts over
the stance leg. Pelvis x vs the ankle midpoint approximates where
mass sits. Two cues combine — pelvis offset toward a side, and
knee flexion unloading a leg — each reported as evidence with a
direction and strength, not a bare label.
"""

from __future__ import annotations

from typing import Dict, List

from .skeleton import Skeleton, observed_point
from .skeleton import Skeleton, observed_body_span


def cues(skel: Skeleton) -> List[dict]:
    """[{cue, side, strength 0-1}] — evidence, honest direction."""
    out: List[dict] = []
    pelvis = observed_point(skel, "pelvis")
    al = observed_point(skel, "ankle_l")
    ar = observed_point(skel, "ankle_r")
    if pelvis and al and ar:
        mid = (al[0] + ar[0]) / 2.0
        span = abs(ar[0] - al[0])
        # Normalise by half the stance; a feet-together pose gives a
        # degenerate denominator where sub-pixel noise reads as a
        # full-strength shift. Fall back to ~half-stance of body
        # height; with no honest scale the offset is unmeasurable.
        half = span / 2.0
        if half < 1.0:
            half = observed_body_span(skel) / 8.0
        if half > 0:
            off = (pelvis[0] - mid) / half
            if abs(off) > 0.15:
                # +x = right-shifted → right dominant
                out.append({"cue": "pelvis_shift",
                            "side": "r" if off > 0 else "l",
                            "strength": round(min(abs(off), 1.0), 2)})
            else:
                out.append({"cue": "pelvis_centered",
                            "side": "even", "strength": 0.5})

    for side in ("l", "r"):
        hip = observed_point(skel, f"hip_{side}")
        knee = observed_point(skel, f"knee_{side}")
        ankle = observed_point(skel, f"ankle_{side}")
        if not (hip and knee and ankle):
            continue
        # flexed knee unloads that leg → other side dominant
        straight = abs(ankle[0] - hip[0])
        leg = abs(knee[0] - hip[0]) + abs(ankle[0] - knee[0])
        # a perfectly collinear leg (hip–knee–ankle on one vertical)
        # is straight, not maximally bent — 0/1.0 reported it as a
        # full fold and fired "unloaded" on a straight-leg pose.
        bend = straight / leg if leg else 1.0
        if bend < 0.5:
            out.append({"cue": f"unloaded_{side}",
                        "side": "r" if side == "l" else "l",
                        "strength": 0.7})
    # bilateral flexion is not dominance evidence: a squat fires
    # unload on both sides, and counting both would fabricate an
    # arbitrary dominant leg on a symmetric pose
    if (any(c["cue"] == "unloaded_l" for c in out)
            and any(c["cue"] == "unloaded_r" for c in out)):
        out = [c for c in out if not c["cue"].startswith("unloaded_")]
        out.append({"cue": "both_legs_flexed",
                    "side": "even", "strength": 0.6})
    return out


def assess(skel: Skeleton) -> dict:
    """{dominant, confidence, cues} — dominant in l|r|even|unknown."""
    c = cues(skel)
    votes: Dict[str, float] = {"l": 0.0, "r": 0.0, "even": 0.0}
    for cue in c:
        votes[cue["side"]] = votes.get(cue["side"], 0.0) + cue["strength"]
    total = sum(votes.values())
    if total < 0.5:
        return {"dominant": "unknown", "confidence": 0.0, "cues": c}
    side = max(votes, key=votes.get)
    conf = round(votes[side] / total, 2)
    return {"dominant": side, "confidence": conf, "cues": c}
