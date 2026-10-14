"""Deterministic tests for enterprise criterion observation compilation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import runpy
from typing import Any

import pytest

import fast_mlsirm.scoring.enterprise_issue as enterprise
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EvidenceReference,
    EvidenceRole,
    ObservationGranularity,
    ObservationStatus,
    ScoreObservation,
    build_scoring_request,
)
from fast_mlsirm.scoring.enterprise_issue import (
    AtomicIssueRecord,
    CounterevidenceRecord,
    EnterpriseAssertionKind,
    EnterpriseSourceRecord,
    EvidenceSpanRecord,
    build_enterprise_issue_score_observation,
    build_enterprise_issue_scoring_request,
    enterprise_issue_evidence_references,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
rubric = _FIXTURES["rubric"]
automated_engine = _FIXTURES["automated_engine"]

SOURCE_CONTENT_FP = hashlib.sha256(b"enterprise-observation-source").hexdigest()
ISSUE_CONTENT_FP = hashlib.sha256(b"enterprise-observation-issue").hexdigest()
FACT_CONTENT_FP = hashlib.sha256(b"enterprise-observation-fact").hexdigest()
COUNTER_CONTENT_FP = hashlib.sha256(b"enterprise-observation-counter").hexdigest()
TASK_REVISION_FP = hashlib.sha256(b"enterprise-observation-task").hexdigest()
UNKNOWN_CONTENT_FP = hashlib.sha256(b"enterprise-observation-unknown").hexdigest()


def _source() -> EnterpriseSourceRecord:
    """Return one deterministic source identity without source text."""
    return EnterpriseSourceRecord(
        source_id="customer_report",
        source_family_id="customer_feedback",
        source_content_fingerprint=SOURCE_CONTENT_FP,
        source_character_count=240,
        metadata={"source_channel": "support_portal"},
    )


def _span(
    kind: EnterpriseAssertionKind,
    *,
    source_record_fingerprint: str,
    span_id: str,
    span_content_fingerprint: str,
    start_offset: int,
) -> EvidenceSpanRecord:
    """Return one deterministic enterprise evidence span."""
    return EvidenceSpanRecord(
        source_id="customer_report",
        source_record_fingerprint=source_record_fingerprint,
        span_id=span_id,
        span_content_fingerprint=span_content_fingerprint,
        assertion_kind=kind,
        start_offset=start_offset,
        end_offset=start_offset + 20,
        metadata={"parser_family": "offline_fixture"},
    )


def _issue() -> AtomicIssueRecord:
    """Return one issue with both supporting evidence and counterevidence."""
    source = _source()
    fact = _span(
        EnterpriseAssertionKind.DIRECT_FACT,
        source_record_fingerprint=source.source_record_fingerprint,
        span_id="reported_deadline",
        span_content_fingerprint=FACT_CONTENT_FP,
        start_offset=12,
    )
    counter_span = _span(
        EnterpriseAssertionKind.COUNTEREVIDENCE,
        source_record_fingerprint=source.source_record_fingerprint,
        span_id="resolved_incident",
        span_content_fingerprint=COUNTER_CONTENT_FP,
        start_offset=60,
    )
    counter = CounterevidenceRecord(
        counterevidence_id="resolved_incident_record",
        issue_content_fingerprint=ISSUE_CONTENT_FP,
        evidence_span=counter_span,
        metadata={"verification_state": "source_verified"},
    )
    return AtomicIssueRecord(
        issue_id="delivery_deadline_risk",
        issue_family_id="service_delivery_risk",
        issue_content_fingerprint=ISSUE_CONTENT_FP,
        source_record_fingerprints=(source.source_record_fingerprint,),
        evidence_spans=(fact,),
        counterevidence_records=(counter,),
        metadata={"decision_scope": "contract_review"},
    )


def _request(**overrides: Any):
    """Return one exact enterprise criterion-level scoring request."""
    values: dict[str, Any] = {
        "request_id": "enterprise_observation_request",
        "assessment": assessment(),
        "rubric": rubric(),
        "issue": _issue(),
        "response_id": "delivery_issue_response",
        "task_id": "issue_evidence_review",
        "task_revision_fingerprint": TASK_REVISION_FP,
        "task_family_id": "evidence_review",
        "occasion_id": "initial_review",
        "criterion_ids": ("claim_support", "source_alignment"),
        "response_character_count": 128,
        "response_unit_count": 8,
        "metadata": {"deployment_stage": "offline_fixture"},
    }
    values.update(overrides)
    return build_enterprise_issue_scoring_request(**values)


def _references(request=None) -> tuple[EvidenceReference, EvidenceReference]:
    """Return the request's supporting and counter evidence references."""
    issue = _issue()
    references = enterprise_issue_evidence_references(issue)
    supporting = next(
        value for value in references if value.evidence_role is EvidenceRole.SUPPORTING
    )
    counter = next(
        value for value in references if value.evidence_role is EvidenceRole.COUNTER
    )
    return supporting, counter


def _observation(**overrides: Any) -> ScoreObservation:
    """Build one deterministic scored enterprise observation."""
    supporting, counter = _references()
    values: dict[str, Any] = {
        "observation_id": "claim_support_observation",
        "request": _request(),
        "engine": automated_engine(),
        "criterion_id": "claim_support",
        "status": ObservationStatus.SCORED,
        "score_category": 2,
        "evidence_references": (supporting, counter),
        "confidence_metadata": {"confidence_band": "reviewed_high"},
    }
    values.update(overrides)
    return build_enterprise_issue_score_observation(**values)


def _assert_error(code: str, callback) -> None:
    """Assert one stable scoring-contract error code."""
    with pytest.raises(AssessmentSpecError) as captured:
        callback()
    assert captured.value.code == code


def _non_enterprise_request(metadata: dict[str, Any] | None = None):
    """Return one shared request that did not pass through the enterprise adapter."""
    return build_scoring_request(
        request_id="generic_scoring_request",
        assessment=assessment(),
        rubric=rubric(),
        granularity=ObservationGranularity.CRITERION_LEVEL,
        respondent_id="sample_respondent",
        response_id="sample_response",
        task_id="sample_task",
        task_revision_fingerprint=TASK_REVISION_FP,
        task_family_id="evidence_review",
        occasion_id="initial_review",
        criterion_ids=("claim_support", "source_alignment"),
        response_content_fingerprint=ISSUE_CONTENT_FP,
        response_character_count=128,
        response_unit_count=8,
        metadata={} if metadata is None else metadata,
    )


def test_public_observation_surface_is_explicit_and_documented() -> None:
    """The enterprise namespace exports the shared-observation compiler."""
    assert "build_enterprise_issue_score_observation" in enterprise.__all__
    assert build_enterprise_issue_score_observation.__doc__


def test_scored_observation_reuses_shared_contract_and_exact_provenance() -> None:
    """A supported score becomes one canonical shared ScoreObservation."""
    observation = _observation()
    payload = observation.to_dict()
    metadata = payload["confidence_metadata"]

    assert isinstance(observation, ScoreObservation)
    assert observation.status is ObservationStatus.SCORED
    assert observation.score_category == 2
    assert metadata["confidence_band"] == "reviewed_high"
    assert metadata["enterprise_atomic_issue_fingerprint"]
    assert metadata["enterprise_issue_content_fingerprint"] == ISSUE_CONTENT_FP
    assert metadata["enterprise_supporting_evidence_count"] == 1
    assert metadata["enterprise_counter_evidence_count"] == 1
    assert metadata["enterprise_context_evidence_count"] == 0
    assert metadata["enterprise_observation_evidence_fingerprints"] == sorted(
        metadata["enterprise_observation_evidence_fingerprints"]
    )
    assert "source_text" not in repr(payload)


def test_evidence_input_order_does_not_change_observation_identity() -> None:
    """Evidence ordering cannot become a hidden scoring feature."""
    supporting, counter = _references()
    first = _observation(evidence_references=(supporting, counter))
    second = _observation(evidence_references=(counter, supporting))

    assert first.observation_fingerprint == second.observation_fingerprint


def test_abstention_may_preserve_no_evidence() -> None:
    """Insufficient evidence remains an explicit abstention rather than a score."""
    observation = _observation(
        status=ObservationStatus.ABSTAINED,
        score_category=None,
        reason_code="insufficient_evidence",
        evidence_references=(),
    )

    assert observation.status is ObservationStatus.ABSTAINED
    assert observation.evidence_references == ()
    assert observation.to_dict()["confidence_metadata"][
        "enterprise_supporting_evidence_count"
    ] == 0


@pytest.mark.parametrize(
    "status",
    (ObservationStatus.SCORED, ObservationStatus.FAILED, ObservationStatus.EXCLUDED),
)
def test_every_non_abstained_observation_requires_supporting_evidence(
    status: ObservationStatus,
) -> None:
    """Terminal states cannot bypass the enterprise evidence requirement."""
    values: dict[str, Any] = {
        "status": status,
        "evidence_references": (),
    }
    if status is not ObservationStatus.SCORED:
        values.update(score_category=None, reason_code="review_required")
    _assert_error(
        "missing_enterprise_supporting_evidence",
        lambda: _observation(**values),
    )


def test_declared_counterevidence_must_be_represented() -> None:
    """A score cannot silently omit counterevidence declared by the issue."""
    supporting, _ = _references()
    _assert_error(
        "missing_enterprise_counter_evidence",
        lambda: _observation(evidence_references=(supporting,)),
    )


def test_observation_evidence_must_be_exactly_request_bound() -> None:
    """Unknown, malformed, and duplicate evidence fails before observation creation."""
    supporting, _ = _references()
    unknown = EvidenceReference(
        source_id="other_source",
        span_id="other_span",
        content_fingerprint=UNKNOWN_CONTENT_FP,
        evidence_role=EvidenceRole.SUPPORTING,
    )
    _assert_error(
        "unbound_enterprise_observation_evidence",
        lambda: _observation(evidence_references=(unknown,)),
    )
    _assert_error(
        "invalid_evidence_reference",
        lambda: _observation(evidence_references=(object(),)),
    )
    _assert_error(
        "duplicate_evidence_reference",
        lambda: _observation(evidence_references=(supporting, supporting)),
    )


def test_request_provenance_is_verified_before_observation_building() -> None:
    """Only exact enterprise requests can cross the enterprise observation boundary."""
    supporting, counter = _references()
    _assert_error(
        "invalid_scoring_request",
        lambda: _observation(request=object()),
    )
    _assert_error(
        "missing_enterprise_request_provenance",
        lambda: _observation(request=_non_enterprise_request()),
    )
    duplicate_metadata = {
        "enterprise_atomic_issue_fingerprint": "a" * 64,
        "enterprise_issue_content_fingerprint": "b" * 64,
        "enterprise_evidence_reference_fingerprints": [
            supporting.evidence_fingerprint,
            supporting.evidence_fingerprint,
        ],
        "enterprise_counterevidence_fingerprints": [counter.evidence_fingerprint],
    }
    _assert_error(
        "duplicate_enterprise_request_evidence",
        lambda: _observation(request=_non_enterprise_request(duplicate_metadata)),
    )


def test_confidence_metadata_cannot_override_or_smuggle_provenance() -> None:
    """Managed observation provenance and sensitive source text remain fail-closed."""
    _assert_error(
        "reserved_enterprise_confidence_metadata",
        lambda: _observation(
            confidence_metadata={"enterprise_issue_content_fingerprint": "a" * 64}
        ),
    )
    _assert_error(
        "sensitive_metadata_field",
        lambda: _observation(confidence_metadata={"source_text": "private content"}),
    )


def test_shared_observation_invariants_remain_authoritative() -> None:
    """The adapter delegates criterion, score, engine, and terminal-state semantics."""
    supporting, counter = _references()
    _assert_error(
        "invalid_observation_status",
        lambda: _observation(status="unknown_state"),
    )
    _assert_error(
        "invalid_engine_descriptor",
        lambda: _observation(engine=object()),
    )
    _assert_error(
        "unknown_criterion_id",
        lambda: _observation(criterion_id="unknown_criterion"),
    )
    _assert_error(
        "unknown_score_category",
        lambda: _observation(score_category=99),
    )
    _assert_error(
        "missing_reason_code",
        lambda: _observation(
            status=ObservationStatus.FAILED,
            score_category=None,
            reason_code=None,
            evidence_references=(supporting, counter),
        ),
    )
