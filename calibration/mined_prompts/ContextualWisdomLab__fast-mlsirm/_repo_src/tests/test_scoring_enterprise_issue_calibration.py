"""Governed enterprise issue calibration handoff tests."""

from __future__ import annotations

from typing import Any

import pytest

import fast_mlsirm.scoring.enterprise_issue as enterprise
import fast_mlsirm.scoring.enterprise_issue.calibration as calibration_module
from enterprise_issue_calibration_fixtures import (
    CRITERION_IDS,
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
    build_scoring_facets_calibration_bundle,
)
from fast_mlsirm.scoring.enterprise_issue import (
    build_enterprise_issue_facets_rating_records,
    enterprise_issue_evidence_references,
)


def _assert_error(
    code: str,
    callback,
    *,
    path: str | None = None,
) -> None:
    """Assert one stable governed scoring error code and optional path."""
    with pytest.raises(AssessmentSpecError) as captured:
        callback()
    assert captured.value.code == code
    if path is not None:
        assert captured.value.path == path


def test_public_surface_and_shared_bundle_preserve_exact_enterprise_identity() -> None:
    """Enterprise executions become only the existing shared facets contracts."""
    executions = (
        _execution(
            issue_label="alpha",
            task_label="alpha",
            engine_label="alpha",
            scores=(0, 1),
        ),
        _execution(
            issue_label="alpha",
            task_label="beta",
            engine_label="beta",
            scores=(1, 2),
        ),
        _execution(
            issue_label="beta",
            task_label="alpha",
            engine_label="beta",
            scores=(2, 0),
        ),
        _execution(
            issue_label="beta",
            task_label="beta",
            engine_label="alpha",
            scores=(1, 2),
        ),
    )
    records = tuple(
        record
        for issue, request, result, engine in executions
        for record in build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=result,
            engine=engine,
        )
    )
    bundle = build_scoring_facets_calibration_bundle(records)

    assert "build_enterprise_issue_facets_rating_records" in enterprise.__all__
    assert build_enterprise_issue_facets_rating_records.__doc__
    assert bundle.criterion_ids == CRITERION_IDS
    assert all(
        design.respondent_ids == ("issue_alpha", "issue_beta")
        for design in bundle.designs
    )
    expected_task_revisions = {
        ("task_alpha", _digest("task-revision:alpha")),
        ("task_beta", _digest("task-revision:beta")),
    }
    assert all(
        set(zip(design.task_ids, design.task_revision_fingerprints, strict=True))
        == expected_task_revisions
        for design in bundle.designs
    )
    assert all(design.connected for design in bundle.designs)
    assert {record.response_content_fingerprint for record in records} == {
        _digest("issue-content:alpha"),
        _digest("issue-content:beta"),
    }
    serialized = repr(bundle.to_dict())
    for issue_label in ("alpha", "beta"):
        assert _digest(f"source-content:{issue_label}") not in serialized
        assert f"source_{issue_label}" not in serialized
    assert "offline_fixture" not in serialized


def test_execution_order_does_not_change_shared_bundle_identity() -> None:
    """Input execution order cannot become a hidden calibration feature."""
    executions = (
        _execution(
            issue_label="alpha",
            task_label="alpha",
            engine_label="alpha",
            scores=(0, 1),
        ),
        _execution(
            issue_label="alpha",
            task_label="beta",
            engine_label="beta",
            scores=(1, 2),
        ),
        _execution(
            issue_label="beta",
            task_label="alpha",
            engine_label="beta",
            scores=(2, 0),
        ),
        _execution(
            issue_label="beta",
            task_label="beta",
            engine_label="alpha",
            scores=(1, 2),
        ),
    )

    def build(values):
        records = tuple(
            record
            for issue, request, result, engine in values
            for record in build_enterprise_issue_facets_rating_records(
                issue=issue,
                request=request,
                result=result,
                engine=engine,
            )
        )
        return build_scoring_facets_calibration_bundle(records)

    assert build(executions).bundle_fingerprint == build(
        tuple(reversed(executions))
    ).bundle_fingerprint


def test_projection_delegates_to_the_shared_rating_builder(monkeypatch) -> None:
    """The enterprise adapter performs replay and delegates record projection."""
    issue, request, result, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    captured: dict[str, Any] = {}
    sentinel = (object(),)

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        calibration_module,
        "build_scoring_facets_rating_records",
        fake_builder,
    )
    assert build_enterprise_issue_facets_rating_records(
        issue=issue,
        request=request,
        result=result,
        engine=engine,
    ) is sentinel
    assert captured == {
        "request": request,
        "result": result,
        "engine": engine,
    }


@pytest.mark.parametrize(
    ("field_name", "code"),
    (
        ("issue", "invalid_atomic_issue"),
        ("request", "invalid_scoring_request"),
        ("result", "invalid_scoring_result"),
        ("engine", "invalid_engine_descriptor"),
    ),
)
def test_exact_public_contract_types_are_required(
    field_name: str,
    code: str,
) -> None:
    """Counterfeit top-level contract values fail before provenance access."""
    issue, request, result, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    values = {
        "issue": issue,
        "request": request,
        "result": result,
        "engine": engine,
    }
    values[field_name] = object()
    _assert_error(
        code,
        lambda: build_enterprise_issue_facets_rating_records(**values),
    )


def test_request_must_have_enterprise_provenance() -> None:
    """A generic shared request cannot masquerade as an enterprise request."""
    issue, request, result, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    generic_request = _rebuild_request(request, metadata={})
    _assert_error(
        "missing_enterprise_request_provenance",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=generic_request,
            result=result,
            engine=engine,
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    ("atomic", "content", "respondent", "response_revision"),
)
def test_request_issue_binding_replays_every_identity(mutation: str) -> None:
    """Issue, content, respondent, and response-revision drift fail closed."""
    issue, request, result, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    metadata = request.to_dict()["metadata"]
    overrides: dict[str, Any] = {}
    if mutation == "atomic":
        metadata["enterprise_atomic_issue_fingerprint"] = _digest(
            "other-atomic"
        )
        overrides["metadata"] = metadata
    elif mutation == "content":
        metadata["enterprise_issue_content_fingerprint"] = _digest(
            "other-content"
        )
        overrides["metadata"] = metadata
    elif mutation == "respondent":
        overrides["respondent_id"] = "issue_other"
    else:
        overrides["response_content_fingerprint"] = _digest("other-response")
    changed_request = _rebuild_request(request, **overrides)

    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=changed_request,
            result=result,
            engine=engine,
        ),
    )


def test_undeclared_observation_evidence_fails_replay() -> None:
    """Observation evidence must remain a subset of the exact request packet."""
    issue, request, _, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    references = (
        EvidenceReference(
            source_id="unknown_source",
            span_id="unknown_span",
            content_fingerprint=_digest("unknown-evidence"),
            evidence_role=EvidenceRole.SUPPORTING,
        ),
    )
    replacement = build_score_observation(
        observation_id="replacement_unknown_evidence",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=references,
        confidence_metadata=_managed_observation_metadata(issue, references),
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )
    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        ),
        path="$.result.observations[0].evidence_references",
    )


def test_non_abstained_observation_requires_supporting_evidence() -> None:
    """A generic scored observation cannot bypass the enterprise evidence gate."""
    issue, request, _, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    references = ()
    replacement = build_score_observation(
        observation_id="replacement_missing_support",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=references,
        confidence_metadata=_managed_observation_metadata(issue, references),
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )
    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        ),
        path="$.result.observations[0].evidence_references",
    )


def test_declared_counterevidence_must_survive_calibration_replay() -> None:
    """Counterevidence cannot disappear before shared calibration."""
    issue = _issue("counter", include_counterevidence=True)
    engine = _engine("counter")
    request = _request(issue, task_label="counter", engine_label="counter")
    result = _result(
        issue=issue,
        request=request,
        engine=engine,
        scores=(1, 2),
    )
    records = build_enterprise_issue_facets_rating_records(
        issue=issue,
        request=request,
        result=result,
        engine=engine,
    )
    assert len(records) == 2

    supporting = tuple(
        value
        for value in enterprise_issue_evidence_references(issue)
        if value.evidence_role is EvidenceRole.SUPPORTING
    )
    replacement = build_score_observation(
        observation_id="replacement_missing_counter",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=supporting,
        confidence_metadata=_managed_observation_metadata(issue, supporting),
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )
    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        ),
        path="$.result.observations[0].evidence_references",
    )


def test_observation_managed_metadata_must_replay_exactly() -> None:
    """Generic confidence metadata cannot counterfeit enterprise observations."""
    issue, request, _, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    replacement = build_score_observation(
        observation_id="replacement_missing_metadata",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=enterprise_issue_evidence_references(issue),
        confidence_metadata={},
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )
    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        ),
        path="$.result.observations[0].confidence_metadata",
    )


def test_abstention_remains_a_terminal_missing_rating() -> None:
    """Insufficient evidence remains abstention rather than a fabricated score."""
    issue = _issue("abstention")
    engine = _engine("abstention")
    request = _request(
        issue,
        task_label="abstention",
        engine_label="abstention",
    )
    result = _result(
        issue=issue,
        request=request,
        engine=engine,
        scores=(0, 2),
        abstain_first=True,
    )
    records = build_enterprise_issue_facets_rating_records(
        issue=issue,
        request=request,
        result=result,
        engine=engine,
    )

    by_criterion = {record.criterion_id: record for record in records}
    assert by_criterion["claim_support"].status is ObservationStatus.ABSTAINED
    assert by_criterion["claim_support"].score_category is None
    assert by_criterion["source_alignment"].score_category == 2
