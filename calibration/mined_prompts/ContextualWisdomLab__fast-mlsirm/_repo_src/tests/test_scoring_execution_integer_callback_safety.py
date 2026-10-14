"""Regression contracts for inert governed-scoring integer validation."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np
import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    ObservationStatus,
    build_score_observation,
    build_scoring_result,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
automated_engine = _FIXTURES["automated_engine"]
criterion_request = _FIXTURES["criterion_request"]
fixture_engine = _FIXTURES["fixture_engine"]


class _IndexCallback:
    """Integer-like caller value whose numeric hooks are executable code."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.callback_count = 0

    def __int__(self) -> int:
        self.callback_count += 1
        return self.value

    def __index__(self) -> int:
        self.callback_count += 1
        return self.value

    def __repr__(self) -> str:
        self.callback_count += 1
        return str(self.value)

    def __eq__(self, other: object) -> bool:
        self.callback_count += 1
        return False

    def __hash__(self) -> int:
        self.callback_count += 1
        return self.value

    def __lt__(self, other: object) -> bool:
        self.callback_count += 1
        return False

    def __le__(self, other: object) -> bool:
        self.callback_count += 1
        return False

    def __gt__(self, other: object) -> bool:
        self.callback_count += 1
        return False

    def __ge__(self, other: object) -> bool:
        self.callback_count += 1
        return False


class _CallerInt(int):
    """Caller-defined built-in integer subclass with executable numeric hooks."""

    callback_count = 0

    def __int__(self) -> int:
        type(self).callback_count += 1
        return 1

    def __index__(self) -> int:
        type(self).callback_count += 1
        return 1

    def __repr__(self) -> str:
        type(self).callback_count += 1
        return "1"

    def __eq__(self, other: object) -> bool:
        type(self).callback_count += 1
        return False

    def __hash__(self) -> int:
        type(self).callback_count += 1
        return 1

    def __lt__(self, other: object) -> bool:
        type(self).callback_count += 1
        return False

    def __le__(self, other: object) -> bool:
        type(self).callback_count += 1
        return False

    def __gt__(self, other: object) -> bool:
        type(self).callback_count += 1
        return False

    def __ge__(self, other: object) -> bool:
        type(self).callback_count += 1
        return False


class _CallerNumpyInt(np.int64):
    """Caller-defined NumPy integer subclass with executable numeric hooks."""

    callback_count = 0

    def __int__(self) -> int:
        type(self).callback_count += 1
        return 2

    def __index__(self) -> int:
        type(self).callback_count += 1
        return 2

    def __repr__(self) -> str:
        type(self).callback_count += 1
        return "2"

    def __eq__(self, other: object) -> bool:
        type(self).callback_count += 1
        return False

    def __hash__(self) -> int:
        type(self).callback_count += 1
        return 2

    def __lt__(self, other: object) -> bool:
        type(self).callback_count += 1
        return False

    def __le__(self, other: object) -> bool:
        type(self).callback_count += 1
        return False

    def __gt__(self, other: object) -> bool:
        type(self).callback_count += 1
        return False

    def __ge__(self, other: object) -> bool:
        type(self).callback_count += 1
        return False


def _scored_observations():
    """Return a complete trusted observation set for one criterion request."""
    request = criterion_request()
    return request, fixture_engine().score(request).observations


def test_request_integer_controls_reject_index_callbacks_without_execution() -> None:
    """Request controls fail before conversion, comparison, equality, or hashing."""
    hostile = _IndexCallback(128)

    with pytest.raises(AssessmentSpecError) as captured:
        criterion_request(response_character_count=hostile)

    assert captured.value.code == "invalid_response_character_count"
    assert hostile.callback_count == 0


def test_score_category_rejects_numpy_subclasses_without_execution() -> None:
    """Score categories reject NumPy subclasses before any numeric callback."""
    request = criterion_request()
    engine = automated_engine()
    _CallerNumpyInt.callback_count = 0

    with pytest.raises(AssessmentSpecError) as captured:
        build_score_observation(
            observation_id="hostile_score_observation",
            request=request,
            engine=engine,
            criterion_id="claim_support",
            status=ObservationStatus.SCORED,
            score_category=_CallerNumpyInt(2),
        )

    assert captured.value.code == "invalid_score_category"
    assert _CallerNumpyInt.callback_count == 0


def test_execution_attempt_rejects_python_integer_subclasses_without_execution() -> None:
    """Result attempt counters reject Python subclasses before numeric callbacks."""
    request, observations = _scored_observations()
    _CallerInt.callback_count = 0

    with pytest.raises(AssessmentSpecError) as captured:
        build_scoring_result(
            result_id="hostile_attempt_result",
            request=request,
            engine=automated_engine(),
            observations=observations,
            execution_attempt=_CallerInt(1),
        )

    assert captured.value.code == "invalid_execution_attempt"
    assert _CallerInt.callback_count == 0


def test_genuine_numpy_integer_scalars_remain_supported() -> None:
    """Trusted NumPy scalar identities preserve the existing public contract."""
    request = criterion_request(
        response_character_count=np.int64(128),
        response_unit_count=np.uint32(8),
    )
    engine = automated_engine()
    observation = build_score_observation(
        observation_id="numpy_score_observation",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=np.int16(2),
    )
    second = build_score_observation(
        observation_id="numpy_alignment_observation",
        request=request,
        engine=engine,
        criterion_id="source_alignment",
        status=ObservationStatus.SCORED,
        score_category=np.uint8(1),
    )
    result = build_scoring_result(
        result_id="numpy_attempt_result",
        request=request,
        engine=engine,
        observations=(observation, second),
        execution_attempt=np.uint64(1),
    )

    assert request.response_character_count == 128
    assert request.response_unit_count == 8
    assert observation.score_category == 2
    assert result.execution_attempt == 1
