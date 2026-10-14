"""Factual summary of a campaign. It reports what happened and never upgrades a finding."""

from __future__ import annotations

from typing import Any

from app.discovery.orchestration.model import EvidenceQuality, ResearchState


def build_report(state: ResearchState) -> dict[str, Any]:
    quality: dict[str, int] = {}
    for item in state.evidence:
        quality[item.quality.value] = quality.get(item.quality.value, 0) + 1
    ledger = state.budget
    return {
        "campaign_id": state.identity.campaign_id,
        "state": state.state.value,
        "stop_reason": state.stop_reason,
        "rounds": state.round,
        "decisions": state.decision_number,
        "executed_engines": list(state.successful_engines),
        "failed_engines": list(state.failed_engines),
        "unavailable_engines": list(state.unavailable_engines),
        "skipped_engines": list(state.skipped_engines),
        "exercised_capabilities": list(state.exercised_capabilities),
        "evidence_by_quality": dict(sorted(quality.items())),
        "candidate_evidence": sum(
            1
            for item in state.evidence
            if item.quality in {EvidenceQuality.CANDIDATE, EvidenceQuality.CORROBORATED}
        ),
        "corroborated_evidence": quality.get(EvidenceQuality.CORROBORATED.value, 0),
        "contradictions": [
            {
                "id": item.contradiction_id,
                "kind": item.kind,
                "identity": item.identity_key,
                "status": item.status,
                "discriminators": list(item.discriminators),
            }
            for item in state.contradictions
        ],
        "negative_evidence": [
            {"capability": item.capability, "engine": item.engine, "statement": item.statement}
            for item in state.negatives
        ],
        "uncertainties": list(state.uncertainties),
        "next_capability": state.next_capability,
        "recommend_verification_review": state.recommend_verification_review,
        "budget": {
            "limits": dict(sorted(ledger.limits.items())),
            "consumed": dict(sorted(ledger.consumed.items())),
            "overrun": ledger.overrun,
        },
        "resume_status": state.resume_status.value,
        "verified": False,
        "llm_invoked": False,
    }
