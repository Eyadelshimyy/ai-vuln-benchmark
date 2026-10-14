"""Chosen-model consistency audit — does the prior fit the body?

`select_model` picks a `BODY_MODEL` from one number: the measured
head-to-body ratio. But head ratio says nothing about shoulders, arms
or legs — a stocky adult and a child can share a head size. This audit
re-checks the choice against every other ratio the skeleton actually
measured:

- per-dimension: measured ratio vs the chosen model's prior —
  shoulder_width/shoulder_ratio, torso_length/torso_ratio,
  mean arm / (upper_arm + forearm), mean leg / (thigh + shin)
- `better_model`: the model with the lowest total relative error
  across all measurable dimensions — advisory only, recorded as a
  suggestion because overriding the selection is a guess too

Verdicts: `consistent` | `mismatch` | `unmeasurable` (no ratios or no
chosen model — refusing to audit is the honest answer).
"""

from __future__ import annotations

from typing import Dict, Optional

from .anatomy import BODY_MODELS

# relative error tolerated per dimension — silhouette measurements
# are approximate, so a model only fails beyond a wide band
REL_TOL = 0.35


def _dims(model: dict) -> Dict[str, float]:
    return {
        "shoulder": model["shoulder_ratio"],
        "hip": model["hip_ratio"],
        "torso": model["torso_ratio"],
        "arm": model["upper_arm_ratio"] + model["forearm_ratio"],
        "leg": model["thigh_ratio"] + model["shin_ratio"],
    }


def _measured(ratios: dict) -> Dict[str, Optional[float]]:
    def mean(*keys):
        vals = [ratios[k] for k in keys if ratios.get(k)]
        return sum(vals) / len(vals) if vals else None
    return {
        "shoulder": ratios.get("shoulder_width"),
        "hip": ratios.get("hip_width"),
        "torso": ratios.get("torso_length"),
        "arm": mean("arm_l", "arm_r"),
        "leg": mean("leg_l", "leg_r"),
    }


def _score(expected: Dict[str, float],
           measured: Dict[str, Optional[float]]) -> Optional[float]:
    errs = []
    for k, exp in expected.items():
        m = measured.get(k)
        if m is not None and exp > 0:
            errs.append(abs(m - exp) / exp)
    return sum(errs) / len(errs) if errs else None


def check(skel, ratios: dict,
          models: Optional[Dict[str, dict]] = None) -> dict:
    """Audit the chosen body model against measured ratios."""
    if models is None:
        models = BODY_MODELS
    chosen = (skel.body_model or {}).get("name")
    if chosen not in models:
        return {"verdict": "unmeasurable",
                "reason": "no chosen body_model on skeleton"}

    expected = _dims(models[chosen])
    measured = _measured(ratios)
    per_dim: Dict[str, dict] = {}
    for k, exp in expected.items():
        m = measured.get(k)
        if m is None:
            per_dim[k] = {"verdict": "unmeasurable",
                          "expected": round(exp, 4)}
            continue
        rel = abs(m - exp) / exp
        per_dim[k] = {
            "verdict": "ok" if rel <= REL_TOL else "off",
            "measured": round(m, 4), "expected": round(exp, 4),
            "rel_error": round(rel, 3),
        }
    scored = [d for d in per_dim.values() if "rel_error" in d]
    n_off = sum(1 for d in scored if d["verdict"] == "off")
    total = _score(expected, measured)

    best, best_err = chosen, total
    for name, model in models.items():
        e = _score(_dims(model), measured)
        if e is not None and (best_err is None or e < best_err):
            best, best_err = name, e
    better = best if best != chosen else None

    if not scored:
        verdict = "unmeasurable"
    elif n_off >= 2 or (total is not None and total > REL_TOL):
        verdict = "mismatch"
    else:
        verdict = "consistent"
    return {
        "verdict": verdict,
        "chosen": chosen,
        "dimensions": per_dim,
        "total_rel_error": round(total, 3) if total is not None else None,
        "better_model": better,
        "note": ("better_model is advisory — re-selecting from "
                 "measured limbs is still a guess"),
    }
