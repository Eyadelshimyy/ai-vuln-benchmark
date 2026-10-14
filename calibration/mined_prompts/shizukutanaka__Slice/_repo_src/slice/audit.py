"""Honesty lint — is this document as truthful as it looks?

`validate` says a document is *well-formed*. `audit` says whether it
is *trustworthy*: a schema-perfect doc can still be 90% prediction
with a confident pose label on top. Every check produces a warning
code — nothing here fails hard; consumers decide what to tolerate.

Codes
-----
low_observed        observed joints < threshold of the standard set
no_basis            joints missing their provenance string
confidence_suspect  observed confidence == 1.0 or predicted > 0.9
prediction_mismatch prediction.observed/predicted lists disagree
                    with actual joint states (stale bookkeeping)
semantics_on_prediction  a pose label sits on top of < 40% evidence
"""

from __future__ import annotations


from .knowledge import validate
from .landmarks import JOINTS
from .skeleton import OBSERVED

_MIN_OBSERVED = 0.5
_SEMANTIC_MIN = 0.4


def audit(doc: dict) -> dict:
    errors = validate(doc)
    warnings = []
    joints = (doc.get("skeleton") or {}).get("joints") or {}

    present = [n for n in JOINTS if n in joints]
    observed = [n for n in present
                if joints[n].get("state") == OBSERVED]
    obs_frac = len(observed) / len(JOINTS) if JOINTS else 0.0
    if obs_frac < _MIN_OBSERVED:
        warnings.append({
            "code": "low_observed",
            "detail": "only %d/%d joints observed (%.0f%%)" % (
                len(observed), len(JOINTS), obs_frac * 100)})

    missing_basis = [n for n, j in joints.items()
                     if not j.get("basis")]
    if missing_basis:
        warnings.append({
            "code": "no_basis",
            "detail": "%d joints lack provenance" % len(missing_basis),
            "joints": sorted(missing_basis)})

    suspect = [n for n, j in joints.items()
               if (j.get("state") == OBSERVED
                   and j.get("confidence") == 1.0)
               or (j.get("state") != OBSERVED
                   and j.get("confidence", 0) > 0.9)]
    if suspect:
        warnings.append({
            "code": "confidence_suspect",
            "detail": "implausibly confident joints",
            "joints": sorted(suspect)})

    pred = doc.get("prediction") or {}
    if set(pred.get("observed") or []) != set(observed):
        warnings.append({
            "code": "prediction_mismatch",
            "detail": "prediction.observed list disagrees with "
                      "joint states"})

    if (doc.get("pose") or {}).get("label") \
            and obs_frac < _SEMANTIC_MIN:
        warnings.append({
            "code": "semantics_on_prediction",
            "detail": "pose label '%s' asserted on %.0f%% evidence"
                      % (doc["pose"]["label"], obs_frac * 100)})

    # warnings-block lint: a warning code that contradicts the joint
    # states it describes is a stale claim — consumers treat warnings
    # as the doc's own account of its evidence, so a wrong one
    # misleads them exactly like a stale basis.
    obs_set = {n for n, j in joints.items()
               if isinstance(j, dict) and j.get("state") == OBSERVED}
    torso = ("pelvis", "chest", "neck", "head")
    backed = {
        "few_observed_joints": len(obs_set) < 8,
        "no_observed_torso": not any(n in torso for n in obs_set),
        "no_observed_wrists":
            not any(n.startswith("wrist") for n in obs_set),
        "no_observed_feet":
            not any(n.startswith(("ankle", "foot")) for n in obs_set),
        "no_observed_head": "head" not in obs_set,
    }
    stale = sorted(w for w in (doc.get("warnings") or [])
                   if w in backed and not backed[w])
    if stale:
        warnings.append({
            "code": "stale_warning",
            "detail": "warning codes contradicted by joint states: "
                      + ", ".join(stale),
            "warnings": stale})

    score = 1.0 - 0.15 * len(warnings) - (0.5 if errors else 0)
    return {
        "errors": errors,
        "warnings": warnings,
        "n_warnings": len(warnings),
        "observed_fraction": round(obs_frac, 3),
        "basis_coverage": round(
            1 - len(missing_basis) / max(1, len(joints)), 3),
        "score": round(max(0.0, score), 3),
        "verdict": "invalid" if errors else (
            "clean" if not warnings else "flagged"),
    }
