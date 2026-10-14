"""Authoritative assessment replay for scoring-engine authorization."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    ObservationStatus,
    artifact_digest,
    build_assessment_spec,
    build_score_observation,
    build_scoring_request,
    build_scoring_result,
)
from fast_mlsirm.scoring.execution import (
    build_scoring_request as build_unprojected_request,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
automated_engine = _FIXTURES["automated_engine"]
rubric = _FIXTURES["rubric"]
_TASK_REVISION_FINGERPRINT = "d" * 64


def _unprojected_request(*, metadata):
    """Return one structurally valid request built below the authorization layer."""
    return build_unprojected_request(
        request_id="scoring_request",
        assessment=assessment(),
        rubric=rubric(),
        granularity="criterion_level",
        respondent_id="sample_respondent",
        response_id="sample_response",
        task_id="sample_task",
        task_revision_fingerprint=_TASK_REVISION_FINGERPRINT,
        task_family_id="evidence_review",
        occasion_id="initial_occasion",
        criterion_ids=("claim_support", "source_alignment"),
        response_content_fingerprint="c" * 64,
        response_character_count=128,
        response_unit_count=8,
        metadata=metadata,
    )


def _complete_observations(request, engine):
    """Return complete criterion observations for one request and engine."""
    return tuple(
        build_score_observation(
            observation_id=f"{criterion_id}_observation",
            request=request,
            engine=engine,
            criterion_id=criterion_id,
            status=ObservationStatus.SCORED,
            score_category=2,
        )
        for criterion_id in request.criterion_ids
    )


def _changed_assessment_version(original):
    """Rebuild the fixture assessment under a distinct governed version."""
    return build_assessment_spec(
        assessment_id=original.assessment_id,
        assessment_version="1.0.1",
        constructs=original.constructs,
        rubrics=(rubric(),),
        response_type=original.response_type,
        engine_policy=original.engine_policy,
        calibration_policy=original.calibration_policy,
        validation_policy=original.validation_policy,
        adjudication_policy=original.adjudication_policy,
        monitoring_policy=original.monitoring_policy,
        reporting_policy=original.reporting_policy,
        metadata=original.metadata,
    )


def test_result_authorization_replays_the_authoritative_assessment() -> None:
    """A caller-forged but well-typed policy projection cannot authorize an engine."""
    selected_assessment = assessment()
    rogue_engine = automated_engine(engine_id="rogue_engine")
    forged_request = _unprojected_request(
        metadata={
            "engine_policy_fingerprint": "a" * 64,
            "allow_human_raters": True,
            "allow_automated_raters": True,
            "permitted_engine_ids": ("rogue_engine",),
        }
    )

    with pytest.raises(AssessmentSpecError) as captured:
        build_scoring_result(
            result_id="scoring_result",
            assessment=selected_assessment,
            request=forged_request,
            engine=rogue_engine,
            observations=_complete_observations(forged_request, rogue_engine),
        )

    assert captured.value.code == "engine_policy_projection_mismatch"
    assert captured.value.path == "$.request.metadata"
    assert artifact_digest(selected_assessment.engine_policy) not in str(captured.value)


def test_result_factory_rejects_an_assessment_request_identity_mismatch() -> None:
    """Result finalization requires the exact assessment named by the request."""
    first_assessment = assessment()
    request = _unprojected_request(
        metadata={
            "engine_policy_fingerprint": artifact_digest(first_assessment.engine_policy),
            "allow_human_raters": first_assessment.engine_policy.allow_human_raters,
            "allow_automated_raters": first_assessment.engine_policy.allow_automated_raters,
            "permitted_engine_ids": first_assessment.engine_policy.engine_ids,
        }
    )
    engine = automated_engine()
    changed_assessment = _changed_assessment_version(first_assessment)

    with pytest.raises(AssessmentSpecError) as captured:
        build_scoring_result(
            result_id="scoring_result",
            assessment=changed_assessment,
            request=request,
            engine=engine,
            observations=_complete_observations(request, engine),
        )

    assert captured.value.code == "assessment_request_mismatch"
    assert captured.value.path == "$.request.assessment_fingerprint"


def test_result_factory_accepts_an_exact_authoritative_assessment_replay() -> None:
    """The authoritative assessment authorizes a faithfully projected request."""
    selected_assessment = assessment()
    engine = automated_engine()
    request = build_scoring_request(
        request_id="scoring_request",
        assessment=selected_assessment,
        rubric=rubric(),
        granularity="criterion_level",
        respondent_id="sample_respondent",
        response_id="sample_response",
        task_id="sample_task",
        task_revision_fingerprint=_TASK_REVISION_FINGERPRINT,
        task_family_id="evidence_review",
        occasion_id="initial_occasion",
        criterion_ids=("claim_support", "source_alignment"),
        response_content_fingerprint="c" * 64,
        response_character_count=128,
        response_unit_count=8,
    )

    result = build_scoring_result(
        result_id="scoring_result",
        assessment=selected_assessment,
        request=request,
        engine=engine,
        observations=_complete_observations(request, engine),
    )

    assert result.request_fingerprint == request.request_fingerprint


def test_result_factory_rejects_a_non_assessment_replay_value() -> None:
    """Only a typed AssessmentSpec may serve as the authoritative replay."""
    engine = automated_engine()
    request = _unprojected_request(metadata=None)

    with pytest.raises(AssessmentSpecError) as captured:
        build_scoring_result(
            result_id="scoring_result",
            assessment=object(),
            request=request,
            engine=engine,
            observations=_complete_observations(request, engine),
        )

    assert captured.value.code == "invalid_assessment_spec"
    assert captured.value.path == "$.assessment"
