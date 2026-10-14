"""Adversarial nested-contract tests for enterprise calibration replay."""

from __future__ import annotations

import pytest

from enterprise_issue_calibration_fixtures import (
    _digest,
    _engine,
    _execution,
    _issue,
    _managed_observation_metadata,
    _rebuild_request,
    _request,
    _result,
    _result_with_replacement,
)
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EvidenceReference,
    EvidenceRole,
    ObservationStatus,
    build_score_observation,
)
from fast_mlsirm.scoring.enterprise_issue import (
    build_enterprise_issue_facets_rating_records,
    enterprise_issue_evidence_references,
)


def _assert_provenance_error(callback) -> None:
    """Assert the stable enterprise calibration replay failure contract."""
    with pytest.raises(AssessmentSpecError) as captured:
        callback()
    assert captured.value.code == "enterprise_calibration_provenance_mismatch"


def _forged_reference(label: str, role: EvidenceRole) -> EvidenceReference:
    """Return one well-formed request-declared but issue-unowned reference."""
    return EvidenceReference(
        source_id=f"forged_source_{label}",
        span_id=f"forged_span_{label}",
        content_fingerprint=_digest(f"forged-content:{label}"),
        evidence_role=role,
    )


def _request_with_extra_reference(issue, reference: EvidenceReference):
    """Return one request whose declared evidence includes an unowned reference."""
    request = _request(issue, task_label="forged", engine_label="forged")
    metadata = request.to_dict()["metadata"]
    metadata["enterprise_evidence_reference_fingerprints"] = sorted(
        [
            *metadata["enterprise_evidence_reference_fingerprints"],
            reference.evidence_fingerprint,
        ]
    )
    return _rebuild_request(request, metadata=metadata)


def test_mutated_result_observation_fails_with_structured_error() -> None:
    """An exact result with a counterfeit nested observation never leaks AttributeError."""
    issue, request, result, engine = _execution(
        issue_label="nested_contract",
        task_label="nested_contract",
        engine_label="nested_contract",
        scores=(1, 2),
    )
    object.__setattr__(result, "observations", (object(),))

    with pytest.raises(AssessmentSpecError) as captured:
        build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=result,
            engine=engine,
        )

    assert captured.value.code == "invalid_score_observation"
    assert captured.value.path == "$.result.observations[0]"


@pytest.mark.parametrize(
    "metadata_key",
    (
        "enterprise_source_record_fingerprints",
        "enterprise_evidence_span_fingerprints",
        "enterprise_counterevidence_fingerprints",
    ),
)
def test_issue_owned_request_metadata_must_replay_exactly(metadata_key: str) -> None:
    """Correct top-level fingerprints cannot mask drift in issue-owned metadata."""
    issue = _issue("request_metadata", include_counterevidence=True)
    engine = _engine("request_metadata")
    request = _request(
        issue,
        task_label="request_metadata",
        engine_label="request_metadata",
    )
    result = _result(
        issue=issue,
        request=request,
        engine=engine,
        scores=(1, 2),
    )
    metadata = request.to_dict()["metadata"]
    metadata[metadata_key] = []
    changed_request = _rebuild_request(request, metadata=metadata)

    _assert_provenance_error(
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=changed_request,
            result=result,
            engine=engine,
        )
    )


def test_request_must_retain_every_issue_owned_evidence_reference() -> None:
    """A forged request cannot replace the supplied issue evidence packet."""
    issue = _issue("request_evidence")
    engine = _engine("request_evidence")
    request = _request(
        issue,
        task_label="request_evidence",
        engine_label="request_evidence",
    )
    result = _result(
        issue=issue,
        request=request,
        engine=engine,
        scores=(1, 2),
    )
    metadata = request.to_dict()["metadata"]
    metadata["enterprise_evidence_reference_fingerprints"] = [
        _forged_reference("replacement", EvidenceRole.SUPPORTING).evidence_fingerprint
    ]
    changed_request = _rebuild_request(request, metadata=metadata)

    _assert_provenance_error(
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=changed_request,
            result=result,
            engine=engine,
        )
    )


def test_scored_observation_requires_supporting_evidence_from_supplied_issue() -> None:
    """A request-declared forged supporting span cannot satisfy issue replay."""
    issue = _issue("forged_support")
    engine = _engine("forged")
    forged = _forged_reference("support", EvidenceRole.SUPPORTING)
    request = _request_with_extra_reference(issue, forged)
    replacement = build_score_observation(
        observation_id="forged_supporting_observation",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=(forged,),
        confidence_metadata=_managed_observation_metadata(issue, (forged,)),
    )
    result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )

    _assert_provenance_error(
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=result,
            engine=engine,
        )
    )


def test_scored_observation_requires_exact_supplied_issue_counterevidence() -> None:
    """A forged counter-role span cannot replace declared issue counterevidence."""
    issue = _issue("forged_counter", include_counterevidence=True)
    engine = _engine("forged")
    forged = _forged_reference("counter", EvidenceRole.COUNTER)
    request = _request_with_extra_reference(issue, forged)
    supporting = tuple(
        value
        for value in enterprise_issue_evidence_references(issue)
        if value.evidence_role is EvidenceRole.SUPPORTING
    )
    references = (*supporting, forged)
    replacement = build_score_observation(
        observation_id="forged_counter_observation",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=references,
        confidence_metadata=_managed_observation_metadata(issue, references),
    )
    result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )

    _assert_provenance_error(
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=result,
            engine=engine,
        )
    )
