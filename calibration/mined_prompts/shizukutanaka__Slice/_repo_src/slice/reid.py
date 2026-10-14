"""Person re-identification — pose-invariant identity cues.

`slice.signature`-style pose vectors answer "same pose?". Re-id asks
the opposite question: "same person, whatever the pose?". The cues that
survive a pose change are proportions — bone-length ratios are fixed
per person (forearm/upper-arm, shin/thigh, shoulder/torso ...), so a
ratio vector is (mostly) invariant to how the limbs happen to be
arranged in the frame.

Honesty contract: every reported number is `state: "estimated"` — a
2D projection makes true bone lengths pose-dependent in reality, so
this is a soft cue for triage, not biometric proof. Features whose
source joints were predicted rather than observed are counted,
named in `predicted_features`, and excluded from comparison — a
ratio placed by the anatomy prior is the *same table constant* for
every person, so it adds zero identity signal while pretending to
be evidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .skeleton import Skeleton

# Feature name -> (numerator chain, denominator chain).
# Each chain is a list of (joint_a, joint_b) segments whose lengths sum.
_FEATURES: Dict[str, Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]] = {
    "forearm_over_upperarm_l": (
        [("elbow_l", "wrist_l")], [("shoulder_l", "elbow_l")]),
    "forearm_over_upperarm_r": (
        [("elbow_r", "wrist_r")], [("shoulder_r", "elbow_r")]),
    "shin_over_thigh_l": (
        [("knee_l", "ankle_l")], [("hip_l", "knee_l")]),
    "shin_over_thigh_r": (
        [("knee_r", "ankle_r")], [("hip_r", "knee_r")]),
    "shoulder_over_torso": (
        [("shoulder_l", "shoulder_r")], [("neck", "pelvis")]),
    "hip_over_torso": (
        [("hip_l", "hip_r")], [("neck", "pelvis")]),
    "arm_over_leg_l": (
        [("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l")],
        [("hip_l", "knee_l"), ("knee_l", "ankle_l")]),
    "arm_over_leg_r": (
        [("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r")],
        [("hip_r", "knee_r"), ("knee_r", "ankle_r")]),
    "head_over_torso": (
        [("head", "neck")], [("neck", "chest")]),
}

# Ratios of neighbouring segments vary by a few percent under
# re-projection noise; 8% mean absolute difference is a soft cut.
DEFAULT_THRESHOLD = 0.08


def _seg(skel: Skeleton, a: str, b: str) -> Optional[float]:
    pa, pb = skel.point(a), skel.point(b)
    if pa is None or pb is None:
        return None
    return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5


def _chain(skel: Skeleton, segs: List[Tuple[str, str]]) -> Optional[float]:
    total = 0.0
    for a, b in segs:
        d = _seg(skel, a, b)
        if d is None:
            return None
        total += d
    return total if total > 0 else None


def _joint_states(skel: Skeleton, chains) -> Tuple[int, int]:
    seen = set()
    for chain in chains:
        for a, b in chain:
            seen.add(a)
            seen.add(b)
    obs = sum(1 for n in seen
              if skel.joints.get(n) and skel.joints[n].state == "observed")
    return obs, len(seen) - obs


def features(skel: Skeleton) -> dict:
    """Pose-invariant proportion vector for one skeleton.

    Returns {"vector": {name: ratio}, "joints": {"observed": n,
    "predicted": n}, "predicted_features": [names], "basis": str}.
    Missing chains leave the feature out entirely — never imputed;
    prior-derived features stay in the vector but are named for
    exclusion from comparison.
    """
    vec: Dict[str, float] = {}
    obs_total = pred_total = 0
    pred_features: List[str] = []
    for name, (num, den) in _FEATURES.items():
        n = _chain(skel, num)
        d = _chain(skel, den)
        if n is None or d is None:
            continue
        vec[name] = n / d
        o, p = _joint_states(skel, (num, den))
        obs_total += o
        pred_total += p
        if p:
            pred_features.append(name)
    return {
        "vector": {k: round(v, 4) for k, v in vec.items()},
        "joints": {"observed": obs_total, "predicted": pred_total},
        "predicted_features": sorted(pred_features),
        "basis": "bone-length ratios (pose-invariant to first order)",
    }


def compare(a: dict, b: dict,
            threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Match two feature dicts from `features`.

    Returns {"distance": mean |diff| over shared features or None,
    "same_person": bool|None, "compared": n, "per_feature": {name: diff},
    "excluded_predicted": [names], "state": "estimated",
    "basis": str}. With no shared observed features the verdict is
    honestly None rather than a guess.
    """
    va, vb = a.get("vector", {}), b.get("vector", {})
    shared = sorted(set(va) & set(vb))
    # a feature predicted on either side is prior-table geometry,
    # not person evidence — excluding it is what keeps `distance`
    # an identity cue instead of a body-model similarity check
    excluded = sorted(
        set(shared) & (set(a.get("predicted_features") or [])
                       | set(b.get("predicted_features") or [])))
    shared = [k for k in shared if k not in excluded]
    per = {k: round(abs(va[k] - vb[k]), 4) for k in shared}
    if not shared:
        return {
            "distance": None,
            "same_person": None,
            "compared": 0,
            "per_feature": {},
            "excluded_predicted": excluded,
            "state": "estimated",
            "basis": "no proportion features shared by both skeletons",
        }
    dist = sum(per.values()) / len(shared)
    return {
        "distance": round(dist, 4),
        "same_person": dist <= threshold,
        "compared": len(shared),
        "per_feature": per,
        "excluded_predicted": excluded,
        "state": "estimated",
        "basis": ("mean |ratio diff| over %d shared proportions; "
                  "a 2D cue, not biometric proof" % len(shared)),
    }
