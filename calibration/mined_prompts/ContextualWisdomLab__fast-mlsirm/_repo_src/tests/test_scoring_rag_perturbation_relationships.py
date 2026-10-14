"""Relationship contracts for governed RAG perturbation anchors."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import runpy
from typing import Any

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, ScoringRequest
from fast_mlsirm.scoring.rag import (
    RAGPerturbationConstructBasis,
    RAGPerturbationKind,
    build_rag_perturbation_anchor,
    build_rag_scoring_request,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
rubric = _FIXTURES["rubric"]

QUERY_FP = hashlib.sha256(b"rag-anchor-query").hexdigest()
SECOND_QUERY_FP = hashlib.sha256(b"rag-anchor-query-second").hexdigest()
SYSTEM_FP = hashlib.sha256(b"rag-anchor-system").hexdigest()
OTHER_SYSTEM_FP = hashlib.sha256(b"rag-anchor-other-system").hexdigest()
RETRIEVAL_FP = hashlib.sha256(b"rag-anchor-retrieval").hexdigest()
SECOND_RETRIEVAL_FP = hashlib.sha256(b"rag-anchor-retrieval-second").hexdigest()
RESPONSE_FP = hashlib.sha256(b"rag-anchor-response").hexdigest()
SECOND_RESPONSE_FP = hashlib.sha256(b"rag-anchor-response-second").hexdigest()
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
):
    """Build one anchor with deterministic transformation provenance."""
    return build_rag_perturbation_anchor(
        anchor_id=f"{RAGPerturbationKind(kind).value}_anchor",
        baseline_request=_request() if baseline is None else baseline,
        perturbed_request=perturbed,
        perturbation_specification_fingerprint=SPECIFICATION_FP,
        perturbation_run_fingerprint=RUN_FP,
        perturbation_kind=kind,
    )


def _assert_relationship_error(kind: str, perturbed: ScoringRequest) -> None:
    """Assert one stable invalid-relationship error."""
    with pytest.raises(AssessmentSpecError) as caught:
        _anchor(kind=kind, perturbed=perturbed)
    assert caught.value.code == "invalid_rag_perturbation_relationship"


def test_anchor_builder_accepts_canonical_requests_not_opaque_fingerprints() -> None:
    """The boundary must inspect the governed pair before storing identities."""
    parameters = inspect.signature(build_rag_perturbation_anchor).parameters
    assert {"baseline_request", "perturbed_request"}.issubset(parameters)
    assert {
        "perturbation_specification_fingerprint",
        "perturbation_run_fingerprint",
    }.issubset(parameters)
    assert "baseline_request_fingerprint" not in parameters
    assert "perturbed_request_fingerprint" not in parameters


def test_response_perturbation_binds_only_a_changed_response_artifact() -> None:
    """Response kinds preserve query/retrieval provenance and change response."""
    baseline = _request()
    perturbed = _request(
        request_id="rag_anchor_response_perturbed_request",
        response_id="generated_response_002",
        response_content_fingerprint=SECOND_RESPONSE_FP,
        response_character_count=430,
        response_unit_count=8,
    )

    anchor = _anchor(
        kind=RAGPerturbationKind.UNSUPPORTED_CLAIM,
        baseline=baseline,
        perturbed=perturbed,
    )

    assert anchor.baseline_request_fingerprint == baseline.request_fingerprint
    assert anchor.perturbed_request_fingerprint == perturbed.request_fingerprint
    assert anchor.perturbation_specification_fingerprint == SPECIFICATION_FP
    assert anchor.perturbation_run_fingerprint == RUN_FP


def test_retrieval_perturbation_binds_only_a_changed_retrieval_artifact() -> None:
    """Retrieval kinds preserve query/response while changing retrieval."""
    baseline = _request()
    perturbed = _request(
        request_id="rag_anchor_retrieval_perturbed_request",
        retrieval_run_fingerprint=SECOND_RETRIEVAL_FP,
    )

    anchor = _anchor(
        kind=RAGPerturbationKind.IRRELEVANT_CONTEXT,
        baseline=baseline,
        perturbed=perturbed,
    )

    assert anchor.baseline_request_fingerprint == baseline.request_fingerprint
    assert anchor.perturbed_request_fingerprint == perturbed.request_fingerprint


def test_query_perturbation_requires_a_changed_query_revision() -> None:
    """Query kinds preserve task identity while changing its revision."""
    baseline = _request()
    perturbed = _request(
        request_id="rag_anchor_query_perturbed_request",
        query_revision_fingerprint=SECOND_QUERY_FP,
        retrieval_run_fingerprint=SECOND_RETRIEVAL_FP,
        response_id="generated_response_002",
        response_content_fingerprint=SECOND_RESPONSE_FP,
        response_character_count=215,
        response_unit_count=4,
    )

    anchor = _anchor(
        kind=RAGPerturbationKind.SEMANTIC_QUERY_PARAPHRASE,
        baseline=baseline,
        perturbed=perturbed,
    )

    assert anchor.baseline_request_fingerprint == baseline.request_fingerprint
    assert anchor.perturbed_request_fingerprint == perturbed.request_fingerprint


def test_unrelated_system_configuration_pair_fails_closed() -> None:
    """A different evaluated system cannot masquerade as one perturbation."""
    with pytest.raises(AssessmentSpecError) as caught:
        _anchor(
            kind="unsupported_claim",
            perturbed=_request(
                request_id="rag_anchor_other_system_request",
                system_configuration_id="retrieval_stack_b",
                system_configuration_fingerprint=OTHER_SYSTEM_FP,
                response_id="generated_response_002",
                response_content_fingerprint=SECOND_RESPONSE_FP,
            ),
        )
    assert caught.value.code == "unrelated_rag_perturbation_requests"


def test_response_kind_rejects_a_simultaneous_retrieval_change() -> None:
    """A response perturbation cannot include a second changed treatment."""
    _assert_relationship_error(
        "unsupported_claim",
        _request(
            request_id="rag_anchor_mixed_response_request",
            response_id="generated_response_002",
            response_content_fingerprint=SECOND_RESPONSE_FP,
            retrieval_run_fingerprint=SECOND_RETRIEVAL_FP,
        ),
    )


def test_retrieval_kind_rejects_a_simultaneous_response_change() -> None:
    """A retrieval perturbation must keep the response artifact fixed."""
    _assert_relationship_error(
        "irrelevant_context",
        _request(
            request_id="rag_anchor_mixed_retrieval_request",
            retrieval_run_fingerprint=SECOND_RETRIEVAL_FP,
            response_id="generated_response_002",
            response_content_fingerprint=SECOND_RESPONSE_FP,
        ),
    )


def test_query_kind_rejects_an_unchanged_query_revision() -> None:
    """Changing only the response is not evidence of a query perturbation."""
    _assert_relationship_error(
        "semantic_query_paraphrase",
        _request(
            request_id="rag_anchor_unchanged_query_request",
            response_id="generated_response_002",
            response_content_fingerprint=SECOND_RESPONSE_FP,
        ),
    )


def test_construct_basis_distinguishes_sources_from_model_design() -> None:
    """Unsupported mappings remain explicit model-design hypotheses."""
    assert RAGPerturbationConstructBasis.LITERATURE_ALIGNED_CONSTRUCT.value == (
        "literature_aligned_construct"
    )
    assert RAGPerturbationConstructBasis.MODEL_DESIGN_HYPOTHESIS.value == (
        "model_design_hypothesis"
    )

    literature_aligned = _anchor(
        kind="unsupported_claim",
        perturbed=_request(
            request_id="rag_anchor_grounded_perturbed_request",
            response_id="generated_response_002",
            response_content_fingerprint=SECOND_RESPONSE_FP,
        ),
    )
    model_design = _anchor(
        kind="citation_target_swap",
        perturbed=_request(
            request_id="rag_anchor_citation_perturbed_request",
            response_id="generated_response_002",
            response_content_fingerprint=SECOND_RESPONSE_FP,
        ),
    )

    assert literature_aligned.construct_basis is (
        RAGPerturbationConstructBasis.LITERATURE_ALIGNED_CONSTRUCT
    )
    assert model_design.construct_basis is (
        RAGPerturbationConstructBasis.MODEL_DESIGN_HYPOTHESIS
    )


def test_relation_provenance_fingerprints_fail_closed() -> None:
    """The external controlled-transform protocol and run are content-addressed."""
    perturbed = _request(
        request_id="rag_anchor_bad_relation_request",
        response_id="generated_response_002",
        response_content_fingerprint=SECOND_RESPONSE_FP,
    )
    with pytest.raises(AssessmentSpecError) as caught:
        build_rag_perturbation_anchor(
            anchor_id="bad_relation_anchor",
            baseline_request=_request(),
            perturbed_request=perturbed,
            perturbation_specification_fingerprint="not-a-sha256",
            perturbation_run_fingerprint=RUN_FP,
            perturbation_kind="unsupported_claim",
        )
    assert caught.value.code == "invalid_perturbation_specification_fingerprint"
