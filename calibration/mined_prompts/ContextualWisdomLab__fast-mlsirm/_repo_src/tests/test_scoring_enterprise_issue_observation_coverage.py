"""Defensive branch coverage for the enterprise observation adapter."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, EvidenceRole
from fast_mlsirm.scoring.enterprise_issue import (
    AtomicIssueRecord,
    EnterpriseAssertionKind,
    build_enterprise_issue_score_observation,
    enterprise_issue_evidence_references,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_enterprise_issue_observation.py"))
)
_issue = _FIXTURES["_issue"]
_observation = _FIXTURES["_observation"]
_non_enterprise_request = _FIXTURES["_non_enterprise_request"]
_request = _FIXTURES["_request"]
_references = _FIXTURES["_references"]
_span = _FIXTURES["_span"]
automated_engine = _FIXTURES["automated_engine"]
ISSUE_CONTENT_FP = _FIXTURES["ISSUE_CONTENT_FP"]
UNKNOWN_CONTENT_FP = _FIXTURES["UNKNOWN_CONTENT_FP"]


def test_string_status_none_metadata_and_issue_without_counterevidence() -> None:
    """Valid string status and no-counter branches preserve shared semantics."""
    original = _issue()
    issue = AtomicIssueRecord(
        issue_id=original.issue_id,
        issue_family_id=original.issue_family_id,
        issue_content_fingerprint=original.issue_content_fingerprint,
        source_record_fingerprints=original.source_record_fingerprints,
        evidence_spans=original.evidence_spans,
        counterevidence_records=(),
        metadata=original.metadata,
    )
    request = _request(issue=issue)
    supporting = enterprise_issue_evidence_references(issue)[0]

    observation = build_enterprise_issue_score_observation(
        observation_id="string_status_observation",
        request=request,
        engine=automated_engine(),
        criterion_id="claim_support",
        status="scored",
        score_category=2,
        evidence_references=(supporting,),
        confidence_metadata=None,
    )

    metadata = observation.to_dict()["confidence_metadata"]
    assert metadata["enterprise_supporting_evidence_count"] == 1
    assert metadata["enterprise_counter_evidence_count"] == 0
    assert metadata["enterprise_context_evidence_count"] == 0


def test_context_evidence_count_is_preserved_without_becoming_support() -> None:
    """Contextual ambiguity remains distinct from supporting and counter evidence."""
    original = _issue()
    ambiguity = _span(
        EnterpriseAssertionKind.UNRESOLVED_AMBIGUITY,
        source_record_fingerprint=original.source_record_fingerprints[0],
        span_id="delivery_scope_ambiguity",
        span_content_fingerprint=UNKNOWN_CONTENT_FP,
        start_offset=100,
    )
    issue = AtomicIssueRecord(
        issue_id=original.issue_id,
        issue_family_id=original.issue_family_id,
        issue_content_fingerprint=original.issue_content_fingerprint,
        source_record_fingerprints=original.source_record_fingerprints,
        evidence_spans=original.evidence_spans + (ambiguity,),
        counterevidence_records=original.counterevidence_records,
        metadata=original.metadata,
    )
    references = enterprise_issue_evidence_references(issue)
    assert {value.evidence_role for value in references} == {
        EvidenceRole.SUPPORTING,
        EvidenceRole.COUNTER,
        EvidenceRole.CONTEXT,
    }

    observation = build_enterprise_issue_score_observation(
        observation_id="context_evidence_observation",
        request=_request(issue=issue),
        engine=automated_engine(),
        criterion_id="claim_support",
        status="scored",
        score_category=2,
        evidence_references=references,
    )

    metadata = observation.to_dict()["confidence_metadata"]
    assert metadata["enterprise_supporting_evidence_count"] == 1
    assert metadata["enterprise_counter_evidence_count"] == 1
    assert metadata["enterprise_context_evidence_count"] == 1


def test_duplicate_request_counterevidence_fingerprints_fail_closed() -> None:
    """Spoofed request metadata cannot multiply counterevidence provenance."""
    supporting, counter = _references()
    request = _non_enterprise_request(
        {
            "enterprise_atomic_issue_fingerprint": "a" * 64,
            "enterprise_issue_content_fingerprint": ISSUE_CONTENT_FP,
            "enterprise_evidence_reference_fingerprints": [
                supporting.evidence_fingerprint,
                counter.evidence_fingerprint,
            ],
            "enterprise_counterevidence_fingerprints": [
                counter.evidence_fingerprint,
                counter.evidence_fingerprint,
            ],
        }
    )

    with pytest.raises(AssessmentSpecError) as captured:
        _observation(request=request)
    assert captured.value.code == "duplicate_enterprise_request_counterevidence"
