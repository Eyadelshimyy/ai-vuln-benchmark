"""Deterministic attack-surface ordering. Not an AI ranker."""

from __future__ import annotations

from collections.abc import Sequence

_WEIGHTS = {
    "sol.sibling_auth": 8,
    "sol.accounting_desync": 8,
    "sol.storage_collision": 7,
    "sol.erc4626_inflation": 7,
    "sol.flash_spot": 6,
    "sol.reentrancy": 6,
    "sol.boundary": 4,
}


def priority_score(rule_id: str) -> int:
    return _WEIGHTS.get(rule_id, 1)


def prioritize(rule_ids: Sequence[str]) -> list[str]:
    return sorted(set(rule_ids), key=lambda item: (-priority_score(item), item))


def score_lead(
    *,
    rule_id: str = "",
    impact: str = "medium",
    evidence: str = "none",
    reproducible: bool = False,
    complete: bool = True,
    chain_step: int = 0,
    duplicate: bool = False,
    tool_available: bool = True,
) -> int:
    """Deterministic planning score. It never marks a lead verified."""
    impact_weight = {"low": 1, "medium": 2, "high": 4}.get(impact, 2)
    evidence_weight = {"none": 0, "static": 1, "reproduced": 2, "verified": 3}.get(evidence, 0)
    score = priority_score(rule_id) + impact_weight + evidence_weight
    if reproducible:
        score += 2
    if not complete:
        score -= 3
    if chain_step:
        score += min(chain_step, 3)
    if duplicate:
        score -= 5
    if not tool_available:
        score -= 2
    return score
