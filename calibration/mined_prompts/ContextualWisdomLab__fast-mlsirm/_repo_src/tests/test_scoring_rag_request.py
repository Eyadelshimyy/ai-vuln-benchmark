"""Fail-first contracts for governed reference-free RAG scoring requests."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import runpy
from typing import Any

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, ObservationGranularity, ScoringRequest
from fast_mlsirm.scoring.rag import (
    RAGCandidateVisibility,
    RAGEvidenceRegime,
    build_rag_scoring_request,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
rubric = _FIXTURES["rubric"]

QUERY_FP = hashlib.sha256(b"rag-query-revision").hexdigest()
SYSTEM_FP = hashlib.sha256(b"rag-system-configuration").hexdigest()
RETRIEVAL_FP = hashlib.sha256(b"rag-retrieval-run").hexdigest()
RESPONSE_FP = hashlib.sha256(b"rag-response-content").hexdigest()
SECOND_RETRIEVAL_FP = hashlib.sha256(b"rag-retrieval-run-second").hexdigest()
SECOND_RESPONSE_FP = hashlib.sha256(b"rag-response-content-second").hexdigest()


def _request(**overrides: Any) -> ScoringRequest:
    """Build one deterministic reference-free RAG criterion request."""
    values: dict[str, Any] = {
        "request_id": "rag_quality_request",
        "assessment": assessment(),
        "rubric": rubric(),
        "query_id": "refund_policy_query",
        "query_revision_fingerprint": QUERY_FP,
        "query_testlet_id": "evidence_review",
        "evidence_regime": RAGEvidenceRegime.RETRIEVED_CONTEXT,
        "candidate_visibility": RAGCandidateVisibility.CANDIDATE_BLIND,
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


def _assert_error(code: str, callback) -> None:
    """Assert one stable shared scoring error code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code


def test_public_rag_contract_uses_explicit_evidence_and_visibility_enums() -> None:
    """Reference-free evaluation must declare what evidence and candidate access mean."""
    assert {member.value for member in RAGEvidenceRegime} == {
        "prompt_only",
        "retrieved_context",
        "pooled_corpus",
        "authoritative_corpus",
        "human_anchor",
    }
    assert {member.value for member in RAGCandidateVisibility} == {
        "candidate_blind",
        "candidate_visible_crossfit",
    }
    assert build_rag_scoring_request.__doc__


def test_rag_request_reuses_shared_scoring_axes_without_parallel_schema() -> None:
    """System run, generated response, and query revision map to shared axes."""
    request = _request()

    assert isinstance(request, ScoringRequest)
    assert request.granularity is ObservationGranularity.CRITERION_LEVEL
    assert request.respondent_id == "retrieval_stack_a_run_001"
    assert request.response_id == "generated_response_001"
    assert request.task_id == "refund_policy_query"
    assert request.task_revision_fingerprint == QUERY_FP
    assert request.task_family_id == "evidence_review"
    assert request.response_content_fingerprint == RESPONSE_FP
    assert request.occasion_id == "evaluation_wave_001"
    assert request.criterion_ids == ("answer_relevance", "grounded_generation")
    assert request.to_dict()["metadata"]["rag_system_configuration_id"] == (
        "retrieval_stack_a"
    )


def test_rag_request_preserves_reference_free_provenance_without_raw_text() -> None:
    """The governed numerical artifact retains identities, not question/context/answer text."""
    request = _request()
    payload = request.to_dict()
    metadata = payload["metadata"]

    assert metadata["evaluation_split"] == "offline_holdout"
    assert metadata["rag_evidence_regime"] == "retrieved_context"
    assert metadata["rag_candidate_visibility"] == "candidate_blind"
    assert metadata["rag_system_configuration_id"] == "retrieval_stack_a"
    assert metadata["rag_system_configuration_fingerprint"] == SYSTEM_FP
    assert metadata["rag_retrieval_run_fingerprint"] == RETRIEVAL_FP
    assert metadata["rag_query_revision_fingerprint"] == QUERY_FP
    assert metadata["engine_policy_fingerprint"]

    serialized = repr(payload).lower()
    for forbidden in (
        "query_text",
        "question_text",
        "context_text",
        "retrieved_text",
        "answer_text",
        "response_text",
        "source_text",
    ):
        assert forbidden not in serialized

    parameters = inspect.signature(build_rag_scoring_request).parameters
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


def test_stochastic_runs_do_not_collapse_into_one_system_identity() -> None:
    """One configuration can produce distinct governed run respondents."""
    first = _request()
    second = _request(
        system_run_id="retrieval_stack_a_run_002",
        response_id="generated_response_002",
        retrieval_run_fingerprint=SECOND_RETRIEVAL_FP,
        response_content_fingerprint=SECOND_RESPONSE_FP,
    )

    assert first.respondent_id != second.respondent_id
    assert first.response_id != second.response_id
    assert first.response_content_fingerprint != second.response_content_fingerprint
    assert first.request_fingerprint != second.request_fingerprint
    assert first.to_dict()["metadata"]["rag_system_configuration_id"] == (
        second.to_dict()["metadata"]["rag_system_configuration_id"]
    )
    assert first.to_dict()["metadata"]["rag_system_configuration_fingerprint"] == (
        second.to_dict()["metadata"]["rag_system_configuration_fingerprint"]
    )


def test_evidence_regime_and_candidate_visibility_are_identity_bearing() -> None:
    """Changing the evidence claim or candidate access must change request identity."""
    baseline = _request()
    authoritative = _request(evidence_regime=RAGEvidenceRegime.AUTHORITATIVE_CORPUS)
    crossfit = _request(
        candidate_visibility=RAGCandidateVisibility.CANDIDATE_VISIBLE_CROSSFIT
    )

    assert baseline.request_fingerprint != authoritative.request_fingerprint
    assert baseline.request_fingerprint != crossfit.request_fingerprint


def test_rag_managed_metadata_cannot_be_spoofed_by_callers() -> None:
    """Caller metadata cannot rebind package-managed RAG provenance."""
    for key, value in (
        ("rag_evidence_regime", "human_anchor"),
        ("rag_candidate_visibility", "candidate_visible_crossfit"),
        ("rag_system_configuration_id", "spoofed_configuration"),
        ("rag_system_configuration_fingerprint", SYSTEM_FP),
        ("rag_retrieval_run_fingerprint", RETRIEVAL_FP),
        ("rag_query_revision_fingerprint", QUERY_FP),
    ):
        _assert_error(
            "reserved_rag_metadata",
            lambda key=key, value=value: _request(metadata={key: value}),
        )


def test_rag_contract_rejects_unknown_enums_and_malformed_fingerprints() -> None:
    """Evidence meaning and exact provenance fail closed instead of being guessed."""
    _assert_error(
        "invalid_rag_evidence_regime",
        lambda: _request(evidence_regime="web_truth"),
    )
    _assert_error(
        "invalid_rag_candidate_visibility",
        lambda: _request(candidate_visibility="candidate_visible"),
    )
    _assert_error(
        "invalid_system_configuration_fingerprint",
        lambda: _request(system_configuration_fingerprint="not-a-sha256"),
    )
    _assert_error(
        "invalid_retrieval_run_fingerprint",
        lambda: _request(retrieval_run_fingerprint="not-a-sha256"),
    )


def test_query_revision_metadata_replays_shared_task_revision_identity() -> None:
    """RAG metadata cannot disagree with the shared scoring task-revision axis."""
    request = _request()
    assert request.task_revision_fingerprint == (
        request.to_dict()["metadata"]["rag_query_revision_fingerprint"]
    )


def test_rag_identity_separates_configuration_run_and_generated_response() -> None:
    """A system run is the measured respondent while each generated answer is a response."""
    request = _request(response_id="generated_response_001")

    assert request.respondent_id == "retrieval_stack_a_run_001"
    assert request.response_id == "generated_response_001"
    assert request.to_dict()["metadata"]["rag_system_configuration_id"] == (
        "retrieval_stack_a"
    )


@pytest.mark.parametrize(
    "metadata_key",
    ("audit_note", "query_text", "context_text", "answer_text", "source_text"),
)
def test_rag_metadata_rejects_non_allowlisted_content_keys(metadata_key: str) -> None:
    """Caller metadata cannot smuggle source or response content into artifacts."""
    _assert_error(
        "unsupported_rag_metadata",
        lambda: _request(metadata={metadata_key: "sensitive_content"}),
    )


def test_rag_evaluation_split_rejects_raw_content_value() -> None:
    """The sole allowlisted metadata value is an identifier, never a content channel."""
    raw_query = "What is the refund policy for enterprise customers?"
    with pytest.raises(AssessmentSpecError) as caught:
        _request(metadata={"evaluation_split": raw_query})

    assert caught.value.code == "invalid_evaluation_split"
    assert raw_query not in str(caught.value)


def test_rag_system_configuration_id_rejects_raw_content_value() -> None:
    """Managed system identity cannot become a raw query/context side channel."""
    raw_configuration = "Use this retrieved context to answer the customer refund question"
    with pytest.raises(AssessmentSpecError) as caught:
        _request(system_configuration_id=raw_configuration)

    assert caught.value.code == "invalid_system_configuration_id"
    assert raw_configuration not in str(caught.value)
