"""Deterministic tests for enterprise issue scoring-request compilation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import runpy
from typing import Any

import pytest

import fast_mlsirm.scoring.enterprise_issue as enterprise
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EvidenceRole,
    ObservationGranularity,
    ScoringRequest,
)
from fast_mlsirm.scoring.enterprise_issue import (
    AtomicIssueRecord,
    CandidateIntervention,
    CounterevidenceRecord,
    EnterpriseAssertionKind,
    EnterpriseSourceRecord,
    EvidenceSpanRecord,
    StakeholderPerspective,
    build_enterprise_issue_scoring_request,
    enterprise_issue_evidence_references,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
rubric = _FIXTURES["rubric"]

SOURCE_FP = hashlib.sha256(b"enterprise-request-source").hexdigest()
OTHER_SOURCE_FP = hashlib.sha256(b"enterprise-request-other-source").hexdigest()
ISSUE_FP = hashlib.sha256(b"enterprise-request-issue").hexdigest()
OTHER_ISSUE_FP = hashlib.sha256(b"enterprise-request-other-issue").hexdigest()
FACT_FP = hashlib.sha256(b"enterprise-request-fact").hexdigest()
COUNTER_FP = hashlib.sha256(b"enterprise-request-counter").hexdigest()
PERSPECTIVE_FP = hashlib.sha256(b"enterprise-request-perspective").hexdigest()
OTHER_PERSPECTIVE_FP = hashlib.sha256(
    b"enterprise-request-other-perspective"
).hexdigest()
INTERVENTION_FP = hashlib.sha256(b"enterprise-request-intervention").hexdigest()
OTHER_INTERVENTION_FP = hashlib.sha256(
    b"enterprise-request-other-intervention"
).hexdigest()
TASK_FP = hashlib.sha256(b"enterprise-request-task").hexdigest()


def _source(*, content_fingerprint: str = SOURCE_FP) -> EnterpriseSourceRecord:
    """Return one deterministic source record."""
    return EnterpriseSourceRecord(
        source_id="customer_report",
        source_family_id="customer_feedback",
        source_content_fingerprint=content_fingerprint,
        source_character_count=480,
        metadata={"source_channel": "support_portal"},
    )


def _span(
    kind: EnterpriseAssertionKind,
    *,
    source_record_fingerprint: str,
    span_id: str,
    content_fingerprint: str,
    start_offset: int,
) -> EvidenceSpanRecord:
    """Return one deterministic evidence span."""
    return EvidenceSpanRecord(
        source_id="customer_report",
        source_record_fingerprint=source_record_fingerprint,
        span_id=span_id,
        span_content_fingerprint=content_fingerprint,
        assertion_kind=kind,
        start_offset=start_offset,
        end_offset=start_offset + 24,
        metadata={"parser_family": "deterministic_offsets"},
    )


def _issue(
    *,
    issue_content_fingerprint: str = ISSUE_FP,
    evidence_spans: tuple[EvidenceSpanRecord, ...] | None = None,
) -> AtomicIssueRecord:
    """Return one deterministic issue with supporting and counter evidence."""
    source = _source()
    fact = _span(
        EnterpriseAssertionKind.DIRECT_FACT,
        source_record_fingerprint=source.source_record_fingerprint,
        span_id="reported_deadline",
        content_fingerprint=FACT_FP,
        start_offset=10,
    )
    counter_span = _span(
        EnterpriseAssertionKind.COUNTEREVIDENCE,
        source_record_fingerprint=source.source_record_fingerprint,
        span_id="resolved_incident",
        content_fingerprint=COUNTER_FP,
        start_offset=60,
    )
    counter = CounterevidenceRecord(
        counterevidence_id="resolved_incident_record",
        issue_content_fingerprint=issue_content_fingerprint,
        evidence_span=counter_span,
        metadata={"verification_state": "source_verified"},
    )
    return AtomicIssueRecord(
        issue_id="delivery_deadline_risk",
        issue_family_id="service_delivery_risk",
        issue_content_fingerprint=issue_content_fingerprint,
        source_record_fingerprints=(source.source_record_fingerprint,),
        evidence_spans=(fact,) if evidence_spans is None else evidence_spans,
        counterevidence_records=(counter,),
        metadata={"decision_scope": "contract_review"},
    )


def _perspective(
    issue: AtomicIssueRecord,
    *,
    perspective_id: str = "operations_view",
    stakeholder_id: str = "operations_team",
    span_content_fingerprint: str = PERSPECTIVE_FP,
    source_record_fingerprint: str | None = None,
    issue_content_fingerprint: str | None = None,
    value_judgment_span: EvidenceSpanRecord | None = None,
) -> StakeholderPerspective:
    """Return one deterministic stakeholder value judgment."""
    source_fingerprint = (
        issue.source_record_fingerprints[0]
        if source_record_fingerprint is None
        else source_record_fingerprint
    )
    span = value_judgment_span or _span(
        EnterpriseAssertionKind.STAKEHOLDER_VALUE_JUDGMENT,
        source_record_fingerprint=source_fingerprint,
        span_id=f"{perspective_id}_priority",
        content_fingerprint=span_content_fingerprint,
        start_offset=110,
    )
    return StakeholderPerspective(
        perspective_id=perspective_id,
        stakeholder_id=stakeholder_id,
        issue_content_fingerprint=(
            issue.issue_content_fingerprint
            if issue_content_fingerprint is None
            else issue_content_fingerprint
        ),
        value_judgment_span=span,
        metadata={"perspective_scope": "delivery_operations"},
    )


def _intervention(
    issue: AtomicIssueRecord,
    *,
    intervention_id: str = "supplier_escalation",
    content_fingerprint: str = INTERVENTION_FP,
    issue_content_fingerprint: str | None = None,
) -> CandidateIntervention:
    """Return one deterministic candidate intervention."""
    return CandidateIntervention(
        intervention_id=intervention_id,
        intervention_family_id="delivery_mitigation",
        issue_content_fingerprint=(
            issue.issue_content_fingerprint
            if issue_content_fingerprint is None
            else issue_content_fingerprint
        ),
        intervention_content_fingerprint=content_fingerprint,
        stakeholder_ids=("account_team", "operations_team"),
        metadata={"effect_status": "caller_supplied_hypothesis"},
    )


def _request(issue: AtomicIssueRecord | None = None, **overrides: Any) -> ScoringRequest:
    """Compile one deterministic criterion-level enterprise request."""
    issue_record = _issue() if issue is None else issue
    is_valid_issue = isinstance(issue_record, AtomicIssueRecord)
    values: dict[str, Any] = {
        "request_id": "enterprise_issue_request",
        "assessment": assessment(),
        "rubric": rubric(),
        "issue": issue_record,
        "response_id": "delivery_issue_response",
        "task_id": "issue_evidence_review",
        "task_revision_fingerprint": TASK_FP,
        "task_family_id": "evidence_review",
        "occasion_id": "initial_review",
        "criterion_ids": ("claim_support", "source_alignment"),
        "response_character_count": 128,
        "response_unit_count": 8,
        "stakeholder_perspectives": (
            (_perspective(issue_record),) if is_valid_issue else ()
        ),
        "candidate_interventions": (
            (_intervention(issue_record),) if is_valid_issue else ()
        ),
        "metadata": {"deployment_stage": "offline_fixture"},
    }
    values.update(overrides)
    return build_enterprise_issue_scoring_request(**values)


def _assert_error(code: str, callback) -> None:
    """Assert one stable scoring contract error code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code


def test_public_request_surface_is_explicit_and_documented() -> None:
    """The enterprise namespace exports both reviewed request helpers."""
    assert "build_enterprise_issue_scoring_request" in enterprise.__all__
    assert "enterprise_issue_evidence_references" in enterprise.__all__
    assert build_enterprise_issue_scoring_request.__doc__
    assert enterprise_issue_evidence_references.__doc__


def test_evidence_references_preserve_roles_and_deterministic_order() -> None:
    """Issue and stakeholder spans compile to shared role-preserving references."""
    issue = _issue()
    references = enterprise_issue_evidence_references(
        issue,
        stakeholder_perspectives=(_perspective(issue),),
    )

    assert tuple(value.evidence_fingerprint for value in references) == tuple(
        sorted(value.evidence_fingerprint for value in references)
    )
    assert {value.evidence_role for value in references} == {
        EvidenceRole.SUPPORTING,
        EvidenceRole.COUNTER,
        EvidenceRole.CONTEXT,
    }
    assert "source_text" not in repr(
        [value.to_dict() for value in references]
    )


def test_request_compiles_exact_provenance_into_shared_contract() -> None:
    """The adapter returns the authoritative shared criterion-level request."""
    issue = _issue()
    request = _request(issue)

    assert isinstance(request, ScoringRequest)
    assert request.granularity is ObservationGranularity.CRITERION_LEVEL
    assert request.respondent_id == issue.issue_id
    assert request.response_id == "delivery_issue_response"
    assert request.response_content_fingerprint == issue.issue_content_fingerprint
    assert request.response_character_count == 128
    assert request.response_unit_count == 8
    metadata = request.to_dict()["metadata"]
    assert metadata["deployment_stage"] == "offline_fixture"
    assert metadata["enterprise_atomic_issue_fingerprint"] == (
        issue.atomic_issue_fingerprint
    )
    assert metadata["enterprise_issue_content_fingerprint"] == (
        issue.issue_content_fingerprint
    )
    assert metadata["enterprise_source_record_fingerprints"] == list(
        issue.source_record_fingerprints
    )
    assert len(metadata["enterprise_evidence_reference_fingerprints"]) == 3
    assert {value["assertion_kind"] for value in metadata["enterprise_assertion_records"]} == {
        "direct_fact",
        "counterevidence",
        "stakeholder_value_judgment",
    }
    assert metadata["enterprise_perspective_fingerprints"]
    assert metadata["enterprise_intervention_fingerprints"]
    assert metadata["engine_policy_fingerprint"]


def test_input_order_does_not_change_request_identity() -> None:
    """Perspective and intervention input order is not a hidden decision feature."""
    issue = _issue()
    first_perspective = _perspective(issue)
    second_perspective = _perspective(
        issue,
        perspective_id="account_team_view",
        stakeholder_id="account_team",
        span_content_fingerprint=OTHER_PERSPECTIVE_FP,
    )
    first_intervention = _intervention(issue)
    second_intervention = _intervention(
        issue,
        intervention_id="delivery_plan_review",
        content_fingerprint=OTHER_INTERVENTION_FP,
    )

    first = _request(
        issue,
        stakeholder_perspectives=(first_perspective, second_perspective),
        candidate_interventions=(first_intervention, second_intervention),
    )
    second = _request(
        issue,
        stakeholder_perspectives=(second_perspective, first_perspective),
        candidate_interventions=(second_intervention, first_intervention),
    )

    assert first.request_fingerprint == second.request_fingerprint


def test_optional_records_and_metadata_may_be_absent() -> None:
    """A source-grounded issue can be scored before interventions are proposed."""
    request = _request(
        stakeholder_perspectives=(),
        candidate_interventions=(),
        metadata=None,
    )
    metadata = request.to_dict()["metadata"]

    assert metadata["enterprise_perspective_fingerprints"] == []
    assert metadata["enterprise_intervention_fingerprints"] == []
    assert len(metadata["enterprise_evidence_reference_fingerprints"]) == 2


def test_issue_type_is_validated_before_attribute_access() -> None:
    """Both public helpers reject non-issue values with one stable error."""
    _assert_error(
        "invalid_atomic_issue",
        lambda: enterprise_issue_evidence_references(object()),  # type: ignore[arg-type]
    )
    _assert_error(
        "invalid_atomic_issue",
        lambda: _request(object()),  # type: ignore[arg-type]
    )


def test_perspectives_reject_invalid_duplicate_cross_issue_and_unbound_values() -> None:
    """Stakeholder records remain type-safe and bound to exact issue sources."""
    issue = _issue()
    valid = _perspective(issue)
    other_issue = _perspective(
        issue,
        issue_content_fingerprint=OTHER_ISSUE_FP,
    )
    unbound = _perspective(
        issue,
        source_record_fingerprint=_source(
            content_fingerprint=OTHER_SOURCE_FP
        ).source_record_fingerprint,
    )

    _assert_error(
        "invalid_stakeholder_perspectives",
        lambda: _request(issue, stakeholder_perspectives=(object(),)),
    )
    _assert_error(
        "duplicate_stakeholder_perspectives",
        lambda: _request(issue, stakeholder_perspectives=(valid, valid)),
    )
    _assert_error(
        "perspective_issue_mismatch",
        lambda: _request(issue, stakeholder_perspectives=(other_issue,)),
    )
    _assert_error(
        "unbound_perspective_source",
        lambda: _request(issue, stakeholder_perspectives=(unbound,)),
    )


def test_interventions_reject_invalid_duplicate_and_cross_issue_values() -> None:
    """Candidate interventions remain type-safe and issue-revision specific."""
    issue = _issue()
    valid = _intervention(issue)
    other_issue = _intervention(
        issue,
        issue_content_fingerprint=OTHER_ISSUE_FP,
    )

    _assert_error(
        "invalid_candidate_interventions",
        lambda: _request(issue, candidate_interventions=(object(),)),
    )
    _assert_error(
        "duplicate_candidate_interventions",
        lambda: _request(issue, candidate_interventions=(valid, valid)),
    )
    _assert_error(
        "intervention_issue_mismatch",
        lambda: _request(issue, candidate_interventions=(other_issue,)),
    )


def test_duplicate_evidence_reference_is_not_silently_multiplied() -> None:
    """The same value-judgment span cannot count as issue and perspective evidence."""
    source = _source()
    judgment = _span(
        EnterpriseAssertionKind.STAKEHOLDER_VALUE_JUDGMENT,
        source_record_fingerprint=source.source_record_fingerprint,
        span_id="operations_priority",
        content_fingerprint=PERSPECTIVE_FP,
        start_offset=110,
    )
    issue = _issue(evidence_spans=(judgment,))
    perspective = _perspective(issue, value_judgment_span=judgment)

    _assert_error(
        "duplicate_enterprise_evidence_references",
        lambda: enterprise_issue_evidence_references(
            issue,
            stakeholder_perspectives=(perspective,),
        ),
    )


def test_caller_cannot_override_or_smuggle_enterprise_metadata() -> None:
    """Managed provenance and raw source content remain fail-closed."""
    _assert_error(
        "reserved_enterprise_metadata",
        lambda: _request(
            metadata={"enterprise_atomic_issue_fingerprint": OTHER_ISSUE_FP}
        ),
    )
    _assert_error(
        "sensitive_metadata_field",
        lambda: _request(metadata={"source_text": "private customer content"}),
    )


def test_shared_request_validation_remains_authoritative() -> None:
    """The adapter delegates counts and assessment validation to shared contracts."""
    _assert_error(
        "invalid_response_character_count",
        lambda: _request(response_character_count=True),
    )
    _assert_error(
        "invalid_assessment_spec",
        lambda: _request(assessment=object()),
    )
