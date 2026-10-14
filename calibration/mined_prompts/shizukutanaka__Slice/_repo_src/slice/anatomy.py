"""Statistical human body priors (Anatomy Engine).

These are population-level templates, NOT claims about the person in an
image. `select_model` picks the prior whose canonical ratios best match
the measured silhouette and reports it as an estimate with confidence.
"""

from __future__ import annotations

from typing import Dict, Tuple

# All lengths are fractions of total body height.
BODY_MODELS: Dict[str, dict] = {
    "adult": {
        "label": "成人（統計モデル）",
        "head_ratio": 1 / 7.5,
        "shoulder_ratio": 0.24,
        "hip_ratio": 0.17,
        "torso_ratio": 0.30,   # shoulder line -> hip line
        "upper_arm_ratio": 0.19,
        "forearm_ratio": 0.16,
        "thigh_ratio": 0.25,
        "shin_ratio": 0.22,
    },
    "child": {
        "label": "子供（統計モデル）",
        "head_ratio": 1 / 5.5,
        "shoulder_ratio": 0.21,
        "hip_ratio": 0.15,
        "torso_ratio": 0.32,
        "upper_arm_ratio": 0.18,
        "forearm_ratio": 0.15,
        "thigh_ratio": 0.22,
        "shin_ratio": 0.18,
    },
    "deformed": {
        "label": "デフォルメ（2.5〜3.5頭身）",
        "head_ratio": 1 / 3.0,
        "shoulder_ratio": 0.20,
        "hip_ratio": 0.16,
        "torso_ratio": 0.28,
        "upper_arm_ratio": 0.15,
        "forearm_ratio": 0.13,
        "thigh_ratio": 0.20,
        "shin_ratio": 0.17,
    },
}

DEFAULT_MODEL = "adult"


def select_model(head_to_body: float) -> Tuple[str, float]:
    """Pick the prior whose head ratio is closest to the measurement.

    Returns (model_name, confidence 0..1). Confidence decays as the
    measurement deviates from the chosen prior AND as the runner-up
    approaches — a reading near the boundary between two models is
    honest ambiguity, not a confident pick.
    """
    ranked = sorted(BODY_MODELS.items(),
                    key=lambda kv: abs(head_to_body - kv[1]["head_ratio"]))
    best_name, best = ranked[0]
    best_err = abs(head_to_body - best["head_ratio"])
    margin = abs(head_to_body - ranked[1][1]["head_ratio"]) - best_err
    conf = 0.9 - best_err * 6
    # Boundaries: halfway between the two nearest head ratios the pick
    # is a coin flip — decay toward the ambiguity floor accordingly.
    conf = min(conf, 0.5 + margin * 4)
    return best_name, round(max(0.2, min(0.9, conf)), 3)
