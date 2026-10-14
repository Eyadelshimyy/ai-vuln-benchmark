"""Fail-first contracts for evidence-regime interpretation limits in RAG scoring."""

from __future__ import annotations

import pytest

import fast_mlsirm.scoring as scoring
from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.rag import RAGEvidenceRegime
from fast_mlsirm.scoring.rag_evidence import (
    RAGEvidenceRegimeLimitations,
    rag_evidence_regime_limitations,
)


def test_retrieved_context_cannot_claim_world_correctness_or_absolute_recall() -> None:
    """Target-retriever context is evidence authority, not an exhaustive truth oracle."""
    limits = rag_evidence_regime_limitations(RAGEvidenceRegime.RETRIEVED_CONTEXT)

    assert limits.regime is RAGEvidenceRegime.RETRIEVED_CONTEXT
    assert "absolute_retrieval_recall_not_identified" in limits.limitation_codes
    assert "world_correctness_not_identified" in limits.limitation_codes
    assert limits.to_dict()["limitation_codes"] == list(limits.limitation_codes)
    assert len(limits.limitations_fingerprint) == 64


def test_every_evidence_regime_has_explicit_nonempty_limitations() -> None:
    """Reports must never silently promote an evidence regime to unlimited authority."""
    for regime in RAGEvidenceRegime:
        limits = rag_evidence_regime_limitations(regime)
        assert limits.limitation_codes
        assert len(set(limits.limitation_codes)) == len(limits.limitation_codes)
        assert tuple(sorted(limits.limitation_codes)) == limits.limitation_codes


def test_evidence_regime_label_alone_never_identifies_world_correctness() -> None:
    """No regime enum value by itself is sufficient evidence of world correctness."""
    for regime in RAGEvidenceRegime:
        limits = rag_evidence_regime_limitations(regime)
        assert "world_correctness_not_identified" in limits.limitation_codes


def test_authoritative_corpus_still_requires_scope_and_completeness_evidence() -> None:
    """The regime label alone cannot prove its authority scope or exhaustiveness."""
    limits = rag_evidence_regime_limitations(RAGEvidenceRegime.AUTHORITATIVE_CORPUS)

    assert "authority_scope_requires_external_validation" in limits.limitation_codes
    assert "corpus_completeness_not_implied" in limits.limitation_codes


def test_human_anchor_is_recorded_as_fallible_observation() -> None:
    """Human anchors remain measurements rather than package-owned ground truth."""
    limits = rag_evidence_regime_limitations(RAGEvidenceRegime.HUMAN_ANCHOR)

    assert "human_anchor_is_fallible_observation" in limits.limitation_codes


def test_unknown_evidence_regime_fails_closed() -> None:
    """Interpretation limits cannot be guessed for an unknown evidence regime."""
    with pytest.raises(AssessmentSpecError) as captured:
        rag_evidence_regime_limitations("web_truth")
    assert captured.value.code == "invalid_rag_evidence_regime"


def test_limitations_contract_is_factory_sealed() -> None:
    """Callers cannot replace package-owned limitation semantics."""
    with pytest.raises(AssessmentSpecError, match="unverified_rag_evidence_limitations"):
        RAGEvidenceRegimeLimitations(
            regime=RAGEvidenceRegime.RETRIEVED_CONTEXT,
            limitation_codes=("world_correctness_identified",),
        )


def test_limitations_replay_rejects_rebound_container_before_callbacks() -> None:
    """Post-construction container rebinding cannot execute during replay."""
    callbacks: list[str] = []

    class HostileTuple(tuple):
        def __iter__(self):  # type: ignore[override]
            """Fail if package replay invokes caller-controlled iteration."""
            callbacks.append("iter")
            raise AssertionError("hostile limitation iteration executed")

    for projection in (
        lambda value: value.to_dict(),
        lambda value: value.limitations_fingerprint,
    ):
        limits = rag_evidence_regime_limitations(RAGEvidenceRegime.RETRIEVED_CONTEXT)
        object.__setattr__(
            limits,
            "limitation_codes",
            HostileTuple(limits.limitation_codes),
        )

        with pytest.raises(AssessmentSpecError) as captured:
            projection(limits)

        assert captured.value.code == "invalid_rag_evidence_limitations"
        assert callbacks == []


def test_evidence_limitations_are_explicit_scoring_package_attributes() -> None:
    """Governed reporting callers should not need an internal-module import path."""
    assert scoring.RAGEvidenceRegimeLimitations is RAGEvidenceRegimeLimitations
    assert scoring.rag_evidence_regime_limitations is rag_evidence_regime_limitations
    assert "RAGEvidenceRegimeLimitations" not in scoring.__all__
    assert "rag_evidence_regime_limitations" not in scoring.__all__
