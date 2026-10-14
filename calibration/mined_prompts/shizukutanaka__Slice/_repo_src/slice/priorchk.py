"""Audit the priors themselves — are the model tables sane?

Everything downstream trusts `BODY_MODELS`: model selection, height
proportions, predicted limb placement. Those numbers were written
by hand and never checked — a typo'd ratio (say, shin longer than
thigh) would flow through every estimate as "the prior says so"
with no one the wiser.

`audit` runs five structural checks over each model:

- `keyset`      — all models carry the same required keys
- `bounds`      — every ratio is a plausible fraction of height
- `limb_order`  — thigh ≥ shin, upper_arm ≥ forearm (anatomically
                  the upper segment is the longer one)
- `stack`       — head+torso+thigh+shin sums to a coherent share of
                  height (0.7–1.1; neck/hand/foot fill the rest)
- `head_order`  — deformed > child > adult head_ratio (the declared
                  stylization ordering; if it inverts, `select_model`
                  drifts)

Each fired issue names model, check, and the raw values — the
audit reports what the table *says*, not what it "meant".
"""

from __future__ import annotations

from typing import Dict, List

from .anatomy import BODY_MODELS

REQUIRED = ("head_ratio", "shoulder_ratio", "hip_ratio",
            "torso_ratio", "upper_arm_ratio", "forearm_ratio",
            "thigh_ratio", "shin_ratio")
# plausible fraction-of-height range for every ratio
LO, HI = 0.05, 0.45
# head+torso+thigh+shin should nearly span the body
STACK_LO, STACK_HI = 0.7, 1.1
# declared stylization ordering of head size
HEAD_ORDER = ("deformed", "child", "adult")


def _issue(model: str, code: str, detail: str) -> Dict:
    return {"model": model, "code": code, "detail": detail}


def audit(models: Dict[str, dict] = None) -> Dict:
    """Check every model table; return issues + verdict."""
    if models is None:
        models = BODY_MODELS
    issues: List[Dict] = []
    checked = 0

    for name, m in models.items():
        checked += 1
        missing = [k for k in REQUIRED if k not in m]
        if missing:
            issues.append(_issue(name, "keyset",
                                 "missing keys: %s" % ", ".join(missing)))
            continue
        for k in REQUIRED:
            v = m[k]
            if not (LO <= v <= HI):
                issues.append(_issue(
                    name, "bounds",
                    "%s=%.3f outside plausible %.2f–%.2f"
                    % (k, v, LO, HI)))
        if m["thigh_ratio"] < m["shin_ratio"]:
            issues.append(_issue(
                name, "limb_order",
                "thigh %.3f < shin %.3f" % (m["thigh_ratio"],
                                          m["shin_ratio"])))
        if m["upper_arm_ratio"] < m["forearm_ratio"]:
            issues.append(_issue(
                name, "limb_order",
                "upper_arm %.3f < forearm %.3f"
                % (m["upper_arm_ratio"], m["forearm_ratio"])))
        stack = (m["head_ratio"] + m["torso_ratio"]
                 + m["thigh_ratio"] + m["shin_ratio"])
        if not (STACK_LO <= stack <= STACK_HI):
            issues.append(_issue(
                name, "stack",
                "head+torso+thigh+shin=%.3f outside %.1f–%.1f"
                % (stack, STACK_LO, STACK_HI)))

    heads = [(name, models[name]["head_ratio"])
             for name in HEAD_ORDER if name in models]
    for i in range(len(heads) - 1):
        if heads[i][1] <= heads[i + 1][1]:
            issues.append(_issue(
                heads[i][0] + "/" + heads[i + 1][0], "head_order",
                "%s head %.3f <= %s head %.3f — stylization order "
                "inverted" % (heads[i][0], heads[i][1],
                              heads[i + 1][0], heads[i + 1][1])))

    return {
        "verdict": "sane" if not issues else "suspicious",
        "issues": issues,
        "n_issues": len(issues),
        "models_checked": checked,
        "checks": ["keyset", "bounds", "limb_order", "stack",
                   "head_order"],
        "state": "audited",
        "basis": "structural rules over the prior tables themselves",
    }
