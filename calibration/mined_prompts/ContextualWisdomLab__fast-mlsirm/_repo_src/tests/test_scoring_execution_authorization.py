"""Engine-policy authorization contracts for governed scoring execution."""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Any

import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EnginePolicy,
    ObservationStatus,
    StaticFixtureEngine,
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
criterion_request = _FIXTURES["criterion_request"]
fixture_engine = _FIXTURES["fixture_engine"]
human_engine = _FIXTURES["human_engine"]
rubric = _FIXTURES["rubric"]
_TASK_REVISION_FINGERPRINT = "d" * 64


def assessment_with_engine_policy(
    *,
    engine_ids: tuple[str, ...] = ("fixture_engine",),
    allow_human_raters: bool = True,
    allow_automated_raters: bool = True,
):
    """Return the fixture assessment with one selected engine policy."""
    base = assessment()
    return build_assessment_spec(
        assessment_id=base.assessment_id,
        assessment_version=base.assessment_version,
        constructs=base.constructs,
        rubrics=(rubric(),),
        response_type=base.response_type,
        engine_policy=EnginePolicy(
            policy_id="engine_policy",
            engine_ids=engine_ids,
            allow_human_raters=allow_human_raters,
            allow_automated_raters=allow_automated_raters,
            minimum_raters_per_response=1,
        ),
        calibration_policy=base.calibration_policy,
        validation_policy=base.validation_policy,
        adjudication_policy=base.adjudication_policy,
        monitoring_policy=base.monitoring_policy,
        reporting_policy=base.reporting_policy,
        metadata=base.metadata,
    )


def observation_for(request, engine):
    """Return one valid first-criterion observation for authorization tests."""
    return build_score_observation(
        observation_id="claim_observation",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=2,
    )


def test_request_contains_exact_engine_policy_projection() -> None:
    """Serialized requests bind the assessment engine policy transitively."""
    selected_assessment = assessment_with_engine_policy()
    request = criterion_request(assessment=selected_assessment)
    assert request.metadata["engine_policy_fingerprint"] == artifact_digest(
        selected_assessment.engine_policy
    )
    assert request.metadata["allow_human_raters"] is True
    assert request.metadata["allow_automated_raters"] is True
    assert request.metadata["permitted_engine_ids"] == ("fixture_engine",)


def test_policy_mutation_changes_the_request_identity() -> None:
    """Changing engine authorization changes the governed request fingerprint."""
    first = criterion_request(
        assessment=assessment_with_engine_policy(engine_ids=("fixture_engine",))
    )
    second = criterion_request(
        assessment=assessment_with_engine_policy(
            engine_ids=("alternate_engine", "fixture_engine")
        )
    )
    assert first.request_fingerprint != second.request_fingerprint
    assert (
        first.metadata["engine_policy_fingerprint"]
        != second.metadata["engine_policy_fingerprint"]
    )


def test_caller_cannot_override_authorization_metadata() -> None:
    """Package-managed policy fields cannot be injected through caller metadata."""
    for reserved_key in (
        "engine_policy_fingerprint",
        "allow_human_raters",
        "allow_automated_raters",
        "permitted_engine_ids",
    ):
        with pytest.raises(AssessmentSpecError) as captured:
            criterion_request(metadata={reserved_key: "caller_value"})
        assert captured.value.code == "reserved_authorization_metadata"
        assert "caller_value" not in str(captured.value)


def test_automated_result_requires_declared_engine_identity() -> None:
    """Automated engine IDs must be explicitly declared by the assessment policy."""
    request = criterion_request(
        assessment=assessment_with_engine_policy(engine_ids=("fixture_engine",))
    )
    engine = automated_engine(engine_id="alternate_engine")
    with pytest.raises(AssessmentSpecError) as captured:
        build_scoring_result(
            result_id="scoring_result",
            request=request,
            engine=engine,
            observations=(observation_for(request, engine),),
        )
    assert captured.value.code == "unauthorized_engine_id"


def test_engine_kind_permissions_fail_closed() -> None:
    """Human and automated execution honor their independent assessment flags."""
    automated_request = criterion_request(
        assessment=assessment_with_engine_policy(
            engine_ids=(),
            allow_human_raters=True,
            allow_automated_raters=False,
        )
    )
    selected_automated_engine = automated_engine()
    with pytest.raises(AssessmentSpecError) as captured:
        build_scoring_result(
            result_id="scoring_result",
            request=automated_request,
            engine=selected_automated_engine,
            observations=(
                observation_for(automated_request, selected_automated_engine),
            ),
        )
    assert captured.value.code == "automated_engine_forbidden"

    human_request = criterion_request(
        assessment=assessment_with_engine_policy(
            engine_ids=("fixture_engine",),
            allow_human_raters=False,
            allow_automated_raters=True,
        )
    )
    selected_human_engine = human_engine()
    with pytest.raises(AssessmentSpecError) as captured:
        build_scoring_result(
            result_id="scoring_result",
            request=human_request,
            engine=selected_human_engine,
            observations=(observation_for(human_request, selected_human_engine),),
        )
    assert captured.value.code == "human_engine_forbidden"


def test_authorized_human_and_automated_engines_reach_result_validation() -> None:
    """Permitted engine kinds pass authorization before ordinary coverage checks."""
    request = criterion_request(assessment=assessment_with_engine_policy())
    for engine in (automated_engine(), human_engine()):
        with pytest.raises(AssessmentSpecError) as captured:
            build_scoring_result(
                result_id="scoring_result",
                request=request,
                engine=engine,
                observations=(observation_for(request, engine),),
            )
        assert captured.value.code == "incomplete_observation_coverage"


def test_public_result_rejects_unprojected_or_malformed_authorization() -> None:
    """Requests built outside the public boundary cannot finalize a governed result."""
    selected_assessment = assessment_with_engine_policy()
    values: dict[str, Any] = {
        "request_id": "scoring_request",
        "assessment": selected_assessment,
        "rubric": rubric(),
        "granularity": "criterion_level",
        "respondent_id": "sample_respondent",
        "response_id": "sample_response",
        "task_id": "sample_task",
        "task_revision_fingerprint": _TASK_REVISION_FINGERPRINT,
        "task_family_id": "evidence_review",
        "occasion_id": "initial_occasion",
        "criterion_ids": ("claim_support", "source_alignment"),
        "response_content_fingerprint": "c" * 64,
        "response_character_count": 128,
        "response_unit_count": 8,
    }
    request = build_unprojected_request(**values)
    engine = automated_engine()
    with pytest.raises(AssessmentSpecError) as captured:
        build_scoring_result(
            result_id="scoring_result",
            request=request,
            engine=engine,
            observations=(observation_for(request, engine),),
        )
    assert captured.value.code == "missing_engine_authorization"

    for metadata in (
        {
            "engine_policy_fingerprint": "a" * 64,
            "allow_human_raters": "yes",
            "allow_automated_raters": True,
            "permitted_engine_ids": ["fixture_engine"],
        },
        {
            "engine_policy_fingerprint": "a" * 64,
            "allow_human_raters": True,
            "allow_automated_raters": True,
            "permitted_engine_ids": "fixture_engine",
        },
    ):
        malformed = build_unprojected_request(**values, metadata=metadata)
        with pytest.raises(AssessmentSpecError) as captured:
            build_scoring_result(
                result_id="scoring_result",
                request=malformed,
                engine=engine,
                observations=(observation_for(malformed, engine),),
            )
        assert captured.value.code == "invalid_engine_authorization"


def test_public_factories_reject_invalid_request_and_engine_types() -> None:
    """Authorization wrappers retain structured public type validation."""
    with pytest.raises(AssessmentSpecError, match="invalid_assessment_spec"):
        build_scoring_request(
            request_id="scoring_request",
            assessment=object(),  # type: ignore[arg-type]
            rubric=rubric(),
            granularity="criterion_level",
            respondent_id="sample_respondent",
            response_id="sample_response",
            task_id="sample_task",
            task_revision_fingerprint=_TASK_REVISION_FINGERPRINT,
            task_family_id="evidence_review",
            occasion_id="initial_occasion",
            criterion_ids=("claim_support",),
            response_content_fingerprint="c" * 64,
            response_character_count=128,
            response_unit_count=8,
        )
    with pytest.raises(AssessmentSpecError, match="invalid_scoring_request"):
        build_scoring_result(
            result_id="scoring_result",
            request=object(),  # type: ignore[arg-type]
            engine=automated_engine(),
            observations=(),
        )
    with pytest.raises(AssessmentSpecError, match="invalid_engine_descriptor"):
        build_scoring_result(
            result_id="scoring_result",
            request=criterion_request(),
            engine=object(),  # type: ignore[arg-type]
            observations=(),
        )


def test_public_fixture_engine_enforces_authorization_and_request_type() -> None:
    """Offline fixture execution cannot bypass the assessment engine policy."""
    selected_fixture = fixture_engine()
    with pytest.raises(AssessmentSpecError, match="invalid_scoring_request"):
        selected_fixture.score(object())  # type: ignore[arg-type]

    unauthorized_request = criterion_request(
        assessment=assessment_with_engine_policy(
            engine_ids=(),
            allow_human_raters=True,
            allow_automated_raters=False,
        )
    )
    with pytest.raises(AssessmentSpecError) as captured:
        selected_fixture.score(unauthorized_request)
    assert captured.value.code == "automated_engine_forbidden"
