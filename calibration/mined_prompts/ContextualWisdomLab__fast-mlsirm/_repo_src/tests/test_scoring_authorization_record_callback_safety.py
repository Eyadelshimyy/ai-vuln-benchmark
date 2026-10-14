"""Regression tests for callback-free scoring authorization record admission."""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Any, TypeVar

import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    StaticFixtureEngine,
    build_scoring_request,
    build_scoring_result,
)
from fast_mlsirm.scoring.assessment import AssessmentSpec
from fast_mlsirm.scoring.execution import EngineDescriptor, ScoringRequest

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
automated_engine = _FIXTURES["automated_engine"]
criterion_request = _FIXTURES["criterion_request"]
fixture_engine = _FIXTURES["fixture_engine"]
rubric = _FIXTURES["rubric"]
_TASK_REVISION_FINGERPRINT = "d" * 64

_T = TypeVar("_T")


def _hostile_record(base_type: type[_T], watched: set[str]) -> tuple[_T, list[str]]:
    """Return an uninitialized subclass that records watched attribute access."""

    callbacks: list[str] = []

    class HostileRecord(base_type):  # type: ignore[misc, valid-type]
        def __getattribute__(self, name: str) -> Any:
            if name in watched:
                callbacks.append(name)
                raise AssertionError(f"caller attribute callback executed: {name}")
            return object.__getattribute__(self, name)

    return object.__new__(HostileRecord), callbacks


def test_request_builder_rejects_assessment_subclass_before_policy_read() -> None:
    """Assessment subclasses cannot execute callbacks during policy projection."""

    hostile, callbacks = _hostile_record(AssessmentSpec, {"engine_policy"})

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_request(
            request_id="scoring_request",
            assessment=hostile,
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

    assert caught.value.code == "invalid_assessment_spec"
    assert callbacks == []


def test_result_builder_rejects_request_subclass_before_provenance_read() -> None:
    """Request subclasses cannot execute callbacks during authorization replay."""

    hostile, callbacks = _hostile_record(ScoringRequest, {"metadata"})

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_result(
            result_id="scoring_result",
            request=hostile,
            engine=automated_engine(),
            observations=(),
        )

    assert caught.value.code == "invalid_scoring_request"
    assert callbacks == []


def test_result_builder_rejects_engine_subclass_before_identity_read() -> None:
    """Engine subclasses cannot execute callbacks during authorization checks."""

    hostile, callbacks = _hostile_record(
        EngineDescriptor,
        {"engine_kind", "engine_id"},
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_result(
            result_id="scoring_result",
            request=criterion_request(),
            engine=hostile,
            observations=(),
        )

    assert caught.value.code == "invalid_engine_descriptor"
    assert callbacks == []


def test_result_builder_rejects_authoritative_assessment_subclass_before_read() -> None:
    """Authoritative assessment subclasses fail before provenance callbacks."""

    hostile, callbacks = _hostile_record(
        AssessmentSpec,
        {"assessment_fingerprint", "engine_policy"},
    )

    with pytest.raises(AssessmentSpecError) as caught:
        build_scoring_result(
            result_id="scoring_result",
            request=criterion_request(),
            engine=automated_engine(),
            observations=(),
            assessment=hostile,
        )

    assert caught.value.code == "invalid_assessment_spec"
    assert callbacks == []


def test_fixture_engine_rejects_request_subclass_before_authorization_read() -> None:
    """Fixture scoring rejects request subclasses before metadata callbacks."""

    selected_fixture: StaticFixtureEngine = fixture_engine()
    hostile, callbacks = _hostile_record(ScoringRequest, {"metadata"})

    with pytest.raises(AssessmentSpecError) as caught:
        selected_fixture.score(hostile)

    assert caught.value.code == "invalid_scoring_request"
    assert callbacks == []
