"""Canonical identity contracts for governed RAG perturbation anchors."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import runpy
from typing import Any

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, ScoringRequest
from fast_mlsirm.scoring.rag import (
    RAGPerturbationAnchor,
    RAGPerturbationDirection,
    RAGPerturbationKind,
    build_rag_perturbation_anchor,
    build_rag_scoring_request,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
rubric = _FIXTURES["rubric"]

QUERY_FP = hashlib.sha256(b"rag-anchor-query-revision").hexdigest()
SYSTEM_FP = hashlib.sha256(b"rag-anchor-system-configuration").hexdigest()
RETRIEVAL_FP = hashlib.sha256(b"rag-anchor-retrieval-run").hexdigest()
RESPONSE_FP = hashlib.sha256(b"rag-anchor-response-content").hexdigest()
SECOND_RESPONSE_FP = hashlib.sha256(b"rag-anchor-response-content-second").hexdigest()
THIRD_RESPONSE_FP = hashlib.sha256(b"rag-anchor-response-content-third").hexdigest()
SPECIFICATION_FP = hashlib.sha256(b"rag-anchor-specification").hexdigest()
RUN_FP = hashlib.sha256(b"rag-anchor-run").hexdigest()


def _request(**overrides: Any) -> ScoringRequest:
    """Build one deterministic canonical RAG scoring request."""
    values: dict[str, Any] = {
        "request_id": "rag_anchor_baseline_request",
        "assessment": assessment(),
        "rubric": rubric(),
        "query_id": "refund_policy_query",
        "query_revision_fingerprint": QUERY_FP,
        "query_testlet_id": "evidence_review",
        "evidence_regime": "retrieved_context",
        "candidate_visibility": "candidate_blind",
        "system_configuration_id": "retrieval_stack_a",
        "system_configuration_fingerprint": SYSTEM_FP,
        "system_run_id": "retrieval_stack_a_run_001",
        "response_id": "generated_response_001",
        "retrieval_run_fingerprint": RETRIEVAL_FP,
        "response_content_fingerprint": RESPONSE_FP,
        "occasion_id": "evaluation_wave_001",
        "criterion_ids": ("grounded_generation", "answer_relevance"),
        "response_character_count": 412,
        "response_unit_count": 7,
        "metadata": {"evaluation_split": "offline_holdout"},
    }
    values.update(overrides)
    return build_rag_scoring_request(**values)


def _anchor(
    *,
    kind: RAGPerturbationKind | str,
    perturbed: ScoringRequest,
    baseline: ScoringRequest | None = None,
    anchor_id: str | None = None,
) -> RAGPerturbationAnchor:
    """Build one deterministic response-axis anchor."""
    normalized_kind = RAGPerturbationKind(kind)
    return build_rag_perturbation_anchor(
        anchor_id=anchor_id or f"{normalized_kind.value}_anchor",
        baseline_request=_request() if baseline is None else baseline,
        perturbed_request=perturbed,
        perturbation_specification_fingerprint=SPECIFICATION_FP,
        perturbation_run_fingerprint=RUN_FP,
        perturbation_kind=normalized_kind,
    )


@pytest.mark.parametrize(
    ("kind", "construct", "direction"),
    (
        ("unsupported_claim", "grounded_generation", "decrease"),
        ("explicit_contradiction", "grounded_generation", "decrease"),
        ("citation_target_swap", "citation_attribution", "decrease"),
        ("style_only_rewrite", "robustness", "invariant"),
    ),
)
def test_response_perturbation_has_one_construct_specific_expected_direction(
    kind: str,
    construct: str,
    direction: str,
) -> None:
    """Response perturbations preserve preregistered construct directions."""
    anchor = _anchor(
        kind=kind,
        perturbed=_request(
            request_id=f"{kind}_request",
            response_id=f"{kind}_response",
            response_content_fingerprint=SECOND_RESPONSE_FP,
        ),
    )

    assert anchor.perturbation_kind is RAGPerturbationKind(kind)
    assert anchor.expected_construct == construct
    assert anchor.expected_direction is RAGPerturbationDirection(direction)


def test_perturbation_enums_are_finite_and_explicit() -> None:
    """The public contract must not accept arbitrary perturbation semantics."""
    assert {member.value for member in RAGPerturbationKind} == {
        "unsupported_claim",
        "explicit_contradiction",
        "irrelevant_context",
        "required_evidence_removal",
        "citation_target_swap",
        "semantic_query_paraphrase",
        "style_only_rewrite",
        "unanswerable_query",
    }
    assert {member.value for member in RAGPerturbationDirection} == {
        "decrease",
        "invariant",
        "increase",
    }


def test_anchor_is_content_addressed_and_identity_bearing() -> None:
    """The request pair and transformation provenance participate in identity."""
    baseline = _request()
    perturbed = _request(
        request_id="unsupported_claim_request",
        response_id="generated_response_002",
        response_content_fingerprint=SECOND_RESPONSE_FP,
    )
    first = _anchor(
        kind="unsupported_claim",
        baseline=baseline,
        perturbed=perturbed,
    )
    replay = _anchor(
        kind="unsupported_claim",
        baseline=baseline,
        perturbed=perturbed,
    )
    second = _anchor(
        kind="unsupported_claim",
        baseline=baseline,
        perturbed=_request(
            request_id="unsupported_claim_second_request",
            response_id="generated_response_003",
            response_content_fingerprint=THIRD_RESPONSE_FP,
        ),
    )

    assert first.anchor_fingerprint == replay.anchor_fingerprint
    assert first.anchor_fingerprint != second.anchor_fingerprint
    payload = first.to_dict()
    assert payload["baseline_request_fingerprint"] == baseline.request_fingerprint
    assert payload["perturbed_request_fingerprint"] == perturbed.request_fingerprint
    assert payload["perturbation_specification_fingerprint"] == SPECIFICATION_FP
    assert payload["perturbation_run_fingerprint"] == RUN_FP
    assert payload["perturbation_kind"] == "unsupported_claim"
    assert payload["expected_construct"] == "grounded_generation"
    assert payload["expected_direction"] == "decrease"
    assert payload["construct_basis"] == "literature_aligned_construct"


def test_anchor_requires_distinct_baseline_and_perturbed_requests() -> None:
    """A no-op request pair cannot masquerade as perturbation evidence."""
    request = _request()
    with pytest.raises(AssessmentSpecError) as caught:
        _anchor(
            kind="style_only_rewrite",
            baseline=request,
            perturbed=request,
            anchor_id="no_op_anchor",
        )
    assert caught.value.code == "identical_rag_perturbation_requests"


def test_anchor_rejects_unknown_semantics() -> None:
    """Unknown perturbation semantics fail closed."""
    with pytest.raises(AssessmentSpecError) as caught:
        build_rag_perturbation_anchor(
            anchor_id="unknown_anchor",
            baseline_request=_request(),
            perturbed_request=_request(
                request_id="unknown_perturbed_request",
                response_id="unknown_response",
                response_content_fingerprint=SECOND_RESPONSE_FP,
            ),
            perturbation_specification_fingerprint=SPECIFICATION_FP,
            perturbation_run_fingerprint=RUN_FP,
            perturbation_kind="improve_everything",
        )
    assert caught.value.code == "invalid_rag_perturbation_kind"


def test_anchor_cannot_be_constructed_without_relationship_validation() -> None:
    """Direct construction cannot bypass canonical pair validation."""
    with pytest.raises(AssessmentSpecError) as caught:
        RAGPerturbationAnchor(
            anchor_id="unverified_anchor",
            baseline_request_fingerprint=RESPONSE_FP,
            perturbed_request_fingerprint=SECOND_RESPONSE_FP,
            perturbation_specification_fingerprint=SPECIFICATION_FP,
            perturbation_run_fingerprint=RUN_FP,
            perturbation_kind=RAGPerturbationKind.UNSUPPORTED_CLAIM,
        )
    assert caught.value.code == "unverified_rag_perturbation_anchor"


def test_anchor_contract_cannot_store_raw_query_context_or_answer_text() -> None:
    """Perturbation evidence is source-free identity metadata, not content."""
    parameters = inspect.signature(build_rag_perturbation_anchor).parameters
    assert not set(parameters).intersection(
        {
            "query_text",
            "question_text",
            "context_text",
            "retrieved_text",
            "answer_text",
            "response_text",
            "source_text",
        }
    )
