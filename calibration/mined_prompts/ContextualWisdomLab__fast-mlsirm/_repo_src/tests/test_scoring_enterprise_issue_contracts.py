"""Deterministic contract tests for enterprise-issue evidence adapters."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, EvidenceRole
from fast_mlsirm.scoring.enterprise_issue import (
    AtomicIssueRecord,
    CandidateIntervention,
    CounterevidenceRecord,
    EnterpriseAssertionKind,
    EnterpriseSourceRecord,
    EvidenceSpanRecord,
    StakeholderPerspective,
)

FP_A = hashlib.sha256(b"enterprise-source-a").hexdigest()
FP_B = hashlib.sha256(b"enterprise-source-b").hexdigest()
FP_C = hashlib.sha256(b"enterprise-issue-c").hexdigest()
FP_D = hashlib.sha256(b"enterprise-span-d").hexdigest()
FP_E = hashlib.sha256(b"enterprise-span-e").hexdigest()
FP_F = hashlib.sha256(b"enterprise-intervention-f").hexdigest()


def _source(*, content_fingerprint: str = FP_A) -> EnterpriseSourceRecord:
    """Return one valid source record fixture."""
    return EnterpriseSourceRecord(
        source_id="customer_report",
        source_family_id="customer_feedback",
        source_content_fingerprint=content_fingerprint,
        source_character_count=240,
        metadata={"source_channel": "support_portal"},
    )


def _span(
    kind: EnterpriseAssertionKind = EnterpriseAssertionKind.DIRECT_FACT,
    *,
    source_record_fingerprint: str | None = None,
    span_id: str = "reported_deadline",
    span_content_fingerprint: str = FP_D,
) -> EvidenceSpanRecord:
    """Return one valid evidence span fixture."""
    source = _source()
    return EvidenceSpanRecord(
        source_id=source.source_id,
        source_record_fingerprint=(
            source.source_record_fingerprint
            if source_record_fingerprint is None
            else source_record_fingerprint
        ),
        span_id=span_id,
        span_content_fingerprint=span_content_fingerprint,
        assertion_kind=kind,
        start_offset=10,
        end_offset=34,
        metadata={"parser_family": "deterministic_offsets"},
    )


def _counter(
    *,
    issue_content_fingerprint: str = FP_C,
    source_record_fingerprint: str | None = None,
) -> CounterevidenceRecord:
    """Return one valid counterevidence fixture."""
    return CounterevidenceRecord(
        counterevidence_id="resolved_incident",
        issue_content_fingerprint=issue_content_fingerprint,
        evidence_span=_span(
            EnterpriseAssertionKind.COUNTEREVIDENCE,
            source_record_fingerprint=source_record_fingerprint,
            span_id="resolved_evidence",
            span_content_fingerprint=FP_E,
        ),
        metadata={"verification_state": "source_verified"},
    )


def _issue(
    *,
    evidence_spans: tuple[EvidenceSpanRecord, ...] | None = None,
    counterevidence_records: tuple[CounterevidenceRecord, ...] | None = None,
    source_record_fingerprints: tuple[str, ...] | None = None,
) -> AtomicIssueRecord:
    """Return one valid atomic issue fixture."""
    source = _source()
    evidence = (_span(source_record_fingerprint=source.source_record_fingerprint),)
    counter = (
        _counter(source_record_fingerprint=source.source_record_fingerprint),
    )
    return AtomicIssueRecord(
        issue_id="delivery_deadline_risk",
        issue_family_id="service_delivery_risk",
        issue_content_fingerprint=FP_C,
        source_record_fingerprints=(source.source_record_fingerprint,)
        if source_record_fingerprints is None
        else source_record_fingerprints,
        evidence_spans=evidence if evidence_spans is None else evidence_spans,
        counterevidence_records=(
            counter if counterevidence_records is None else counterevidence_records
        ),
        metadata={"decision_scope": "contract_review"},
    )


def test_contracts_are_content_addressed_immutable_and_source_text_free() -> None:
    """All records expose stable identities without retaining source text."""
    source = _source()
    issue = _issue()
    perspective = StakeholderPerspective(
        perspective_id="operations_view",
        stakeholder_id="operations_team",
        issue_content_fingerprint=FP_C,
        value_judgment_span=_span(
            EnterpriseAssertionKind.STAKEHOLDER_VALUE_JUDGMENT,
            source_record_fingerprint=source.source_record_fingerprint,
            span_id="operations_priority",
            span_content_fingerprint=FP_E,
        ),
        metadata={"perspective_scope": "delivery_operations"},
    )
    intervention = CandidateIntervention(
        intervention_id="supplier_escalation",
        intervention_family_id="delivery_mitigation",
        issue_content_fingerprint=FP_C,
        intervention_content_fingerprint=FP_F,
        stakeholder_ids=("operations_team", "account_team"),
        metadata={"effect_status": "caller_supplied_hypothesis"},
    )

    assert source.source_record_handle.startswith("enterprise_source_")
    assert issue.atomic_issue_handle.startswith("atomic_issue_")
    assert perspective.perspective_handle.startswith("stakeholder_perspective_")
    assert intervention.intervention_handle.startswith("candidate_intervention_")
    assert issue.to_dict()["atomic_issue_fingerprint"] == issue.atomic_issue_fingerprint
    assert perspective.to_dict()["perspective_fingerprint"] == perspective.perspective_fingerprint
    assert intervention.to_dict()["intervention_fingerprint"] == intervention.intervention_fingerprint
    assert "source_text" not in repr(source.to_dict())
    with pytest.raises(FrozenInstanceError):
        source.source_id = "changed_source"  # type: ignore[misc]


def test_ordering_is_deterministic_and_evidence_compiles_to_shared_contract() -> None:
    """Input reordering preserves issue identity and shared evidence references."""
    source_a = _source()
    source_b = _source(content_fingerprint=FP_B)
    fact = _span(
        source_record_fingerprint=source_a.source_record_fingerprint,
        span_id="direct_fact_span",
        span_content_fingerprint=FP_D,
    )
    inference = _span(
        EnterpriseAssertionKind.SUPPORTED_INFERENCE,
        source_record_fingerprint=source_b.source_record_fingerprint,
        span_id="supported_inference_span",
        span_content_fingerprint=FP_E,
    )
    first = AtomicIssueRecord(
        issue_id="renewal_revenue_risk",
        issue_family_id="commercial_retention_risk",
        issue_content_fingerprint=FP_C,
        source_record_fingerprints=(
            source_b.source_record_fingerprint,
            source_a.source_record_fingerprint,
        ),
        evidence_spans=(inference, fact),
        counterevidence_records=(),
        metadata={},
    )
    second = AtomicIssueRecord(
        issue_id="renewal_revenue_risk",
        issue_family_id="commercial_retention_risk",
        issue_content_fingerprint=FP_C,
        source_record_fingerprints=(
            source_a.source_record_fingerprint,
            source_b.source_record_fingerprint,
        ),
        evidence_spans=(fact, inference),
        counterevidence_records=(),
        metadata={},
    )

    assert first.atomic_issue_fingerprint == second.atomic_issue_fingerprint
    references = first.evidence_references()
    assert tuple(reference.evidence_fingerprint for reference in references) == tuple(
        sorted(reference.evidence_fingerprint for reference in references)
    )
    assert all(reference.evidence_role is EvidenceRole.SUPPORTING for reference in references)


@pytest.mark.parametrize(
    ("kind", "expected_role"),
    (
        (EnterpriseAssertionKind.DIRECT_FACT, EvidenceRole.SUPPORTING),
        (EnterpriseAssertionKind.SUPPORTED_INFERENCE, EvidenceRole.SUPPORTING),
        (EnterpriseAssertionKind.COUNTEREVIDENCE, EvidenceRole.COUNTER),
        (EnterpriseAssertionKind.UNRESOLVED_AMBIGUITY, EvidenceRole.CONTEXT),
        (EnterpriseAssertionKind.STAKEHOLDER_VALUE_JUDGMENT, EvidenceRole.CONTEXT),
    ),
)
def test_assertion_kinds_map_to_explicit_shared_evidence_roles(
    kind: EnterpriseAssertionKind,
    expected_role: EvidenceRole,
) -> None:
    """Epistemic assertion kinds never collapse counterevidence into support."""
    span = _span(kind)
    assert span.evidence_role is expected_role
    assert span.to_evidence_reference().evidence_role is expected_role
    assert span.to_dict()["evidence_reference"]["evidence_role"] == expected_role.value


@pytest.mark.parametrize("value", (True, -1, 100_000_001, object()))
def test_bounded_integer_fields_fail_closed(value: object) -> None:
    """Boolean, out-of-range, and non-integer counts fail with redacted errors."""
    with pytest.raises(AssessmentSpecError, match="invalid_source_character_count"):
        EnterpriseSourceRecord(
            source_id="customer_report",
            source_family_id="customer_feedback",
            source_content_fingerprint=FP_A,
            source_character_count=value,  # type: ignore[arg-type]
            metadata={},
        )


def test_offsets_must_define_one_nonempty_bounded_span() -> None:
    """Offsets reject invalid types, negative starts, and empty or reversed spans."""
    with pytest.raises(AssessmentSpecError, match="invalid_start_offset"):
        EvidenceSpanRecord(
            source_id="customer_report",
            source_record_fingerprint=FP_A,
            span_id="reported_deadline",
            span_content_fingerprint=FP_D,
            assertion_kind="direct_fact",  # type: ignore[arg-type]
            start_offset=object(),  # type: ignore[arg-type]
            end_offset=4,
            metadata={},
        )
    with pytest.raises(AssessmentSpecError, match="invalid_start_offset"):
        EvidenceSpanRecord(
            source_id="customer_report",
            source_record_fingerprint=FP_A,
            span_id="reported_deadline",
            span_content_fingerprint=FP_D,
            assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
            start_offset=-1,
            end_offset=4,
            metadata={},
        )
    with pytest.raises(AssessmentSpecError, match="invalid_evidence_offsets"):
        EvidenceSpanRecord(
            source_id="customer_report",
            source_record_fingerprint=FP_A,
            span_id="reported_deadline",
            span_content_fingerprint=FP_D,
            assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
            start_offset=4,
            end_offset=4,
            metadata={},
        )


def test_sensitive_source_text_is_rejected_from_metadata() -> None:
    """Raw source text cannot enter canonical enterprise metadata."""
    with pytest.raises(AssessmentSpecError, match="sensitive_metadata_field"):
        EnterpriseSourceRecord(
            source_id="customer_report",
            source_family_id="customer_feedback",
            source_content_fingerprint=FP_A,
            source_character_count=20,
            metadata={"source_text": "customer secret"},
        )


def test_counterevidence_requires_exact_kind_and_issue_revision() -> None:
    """Counterevidence cannot be mislabeled or attached to another issue revision."""
    with pytest.raises(AssessmentSpecError, match="invalid_evidence_span"):
        CounterevidenceRecord(
            counterevidence_id="resolved_incident",
            issue_content_fingerprint=FP_C,
            evidence_span=object(),  # type: ignore[arg-type]
            metadata={},
        )
    with pytest.raises(AssessmentSpecError, match="invalid_counterevidence_kind"):
        CounterevidenceRecord(
            counterevidence_id="resolved_incident",
            issue_content_fingerprint=FP_C,
            evidence_span=_span(),
            metadata={},
        )
    source = _source()
    with pytest.raises(AssessmentSpecError, match="counterevidence_issue_mismatch"):
        _issue(
            counterevidence_records=(
                _counter(
                    issue_content_fingerprint=FP_B,
                    source_record_fingerprint=source.source_record_fingerprint,
                ),
            )
        )


def test_atomic_issue_rejects_missing_misplaced_duplicate_and_unbound_evidence() -> None:
    """Issue assembly fails closed on incomplete or inconsistent evidence graphs."""
    source = _source()
    with pytest.raises(AssessmentSpecError, match="missing_issue_evidence"):
        _issue(evidence_spans=(), counterevidence_records=())
    with pytest.raises(AssessmentSpecError, match="misplaced_counterevidence_span"):
        _issue(
            evidence_spans=(
                _span(
                    EnterpriseAssertionKind.COUNTEREVIDENCE,
                    source_record_fingerprint=source.source_record_fingerprint,
                ),
            ),
            counterevidence_records=(),
        )
    duplicated = _span(source_record_fingerprint=source.source_record_fingerprint)
    with pytest.raises(AssessmentSpecError, match="duplicate_evidence_spans"):
        _issue(evidence_spans=(duplicated, duplicated), counterevidence_records=())
    counter = _counter(source_record_fingerprint=source.source_record_fingerprint)
    with pytest.raises(AssessmentSpecError, match="duplicate_counterevidence_records"):
        _issue(evidence_spans=(), counterevidence_records=(counter, counter))
    with pytest.raises(AssessmentSpecError, match="duplicate_source_record_fingerprints"):
        _issue(
            source_record_fingerprints=(
                source.source_record_fingerprint,
                source.source_record_fingerprint,
            )
        )
    with pytest.raises(AssessmentSpecError, match="unbound_issue_source"):
        _issue(source_record_fingerprints=(FP_B,))


def test_atomic_issue_rejects_non_contract_collection_members() -> None:
    """Issue collections accept only package-owned evidence record values."""
    with pytest.raises(AssessmentSpecError, match="invalid_evidence_spans"):
        _issue(evidence_spans=(object(),), counterevidence_records=())  # type: ignore[arg-type]
    with pytest.raises(AssessmentSpecError, match="invalid_counterevidence_records"):
        _issue(evidence_spans=(), counterevidence_records=(object(),))  # type: ignore[arg-type]


def test_stakeholder_perspective_requires_value_judgment_evidence() -> None:
    """Stakeholder preferences remain distinct from factual issue evidence."""
    with pytest.raises(AssessmentSpecError, match="invalid_value_judgment_span"):
        StakeholderPerspective(
            perspective_id="operations_view",
            stakeholder_id="operations_team",
            issue_content_fingerprint=FP_C,
            value_judgment_span=object(),  # type: ignore[arg-type]
            metadata={},
        )
    with pytest.raises(AssessmentSpecError, match="invalid_value_judgment_kind"):
        StakeholderPerspective(
            perspective_id="operations_view",
            stakeholder_id="operations_team",
            issue_content_fingerprint=FP_C,
            value_judgment_span=_span(),
            metadata={},
        )


def test_candidate_intervention_is_noncausal_and_stakeholder_order_invariant() -> None:
    """Candidate actions retain hypotheses without asserting learned effects."""
    first = CandidateIntervention(
        intervention_id="supplier_escalation",
        intervention_family_id="delivery_mitigation",
        issue_content_fingerprint=FP_C,
        intervention_content_fingerprint=FP_F,
        stakeholder_ids=("operations_team", "account_team"),
        metadata={"effect_status": "caller_supplied_hypothesis"},
    )
    second = CandidateIntervention(
        intervention_id="supplier_escalation",
        intervention_family_id="delivery_mitigation",
        issue_content_fingerprint=FP_C,
        intervention_content_fingerprint=FP_F,
        stakeholder_ids=("account_team", "operations_team"),
        metadata={"effect_status": "caller_supplied_hypothesis"},
    )
    assert first.intervention_fingerprint == second.intervention_fingerprint
    assert first.stakeholder_ids == ("account_team", "operations_team")
    with pytest.raises(AssessmentSpecError, match="duplicate_stakeholder_ids"):
        CandidateIntervention(
            intervention_id="supplier_escalation",
            intervention_family_id="delivery_mitigation",
            issue_content_fingerprint=FP_C,
            intervention_content_fingerprint=FP_F,
            stakeholder_ids=("operations_team", "operations_team"),
            metadata={},
        )
