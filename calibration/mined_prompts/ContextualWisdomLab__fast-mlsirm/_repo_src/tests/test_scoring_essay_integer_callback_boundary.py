"""Regression tests for inert essay-adapter integer control validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from fast_mlsirm.scoring import AssessmentSpecError, EvidenceReference, EvidenceRole
from fast_mlsirm.scoring.essay import (
    EssayEvidenceKind,
    build_essay_prompt,
    build_essay_response_evidence,
    build_essay_submission,
)


class _HostileIndex:
    """Integer protocol provider that records any caller callback execution."""

    callbacks: list[str] = []

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        return 10

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "10"

    def __eq__(self, other: object) -> bool:
        type(self).callbacks.append("__eq__")
        return False

    def __hash__(self) -> int:
        type(self).callbacks.append("__hash__")
        return 10

    def __lt__(self, other: object) -> bool:
        type(self).callbacks.append("__lt__")
        return False

    def __le__(self, other: object) -> bool:
        type(self).callbacks.append("__le__")
        return False

    def __gt__(self, other: object) -> bool:
        type(self).callbacks.append("__gt__")
        return False

    def __ge__(self, other: object) -> bool:
        type(self).callbacks.append("__ge__")
        return False


class _HostileInt(int):
    """Caller-defined Python integer subclass with executable numeric hooks."""

    callbacks: list[str] = []

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        return 10

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        return 10

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "10"

    def __eq__(self, other: object) -> bool:
        type(self).callbacks.append("__eq__")
        return False

    def __hash__(self) -> int:
        type(self).callbacks.append("__hash__")
        return 10

    def __lt__(self, other: object) -> bool:
        type(self).callbacks.append("__lt__")
        return False

    def __le__(self, other: object) -> bool:
        type(self).callbacks.append("__le__")
        return False

    def __gt__(self, other: object) -> bool:
        type(self).callbacks.append("__gt__")
        return False

    def __ge__(self, other: object) -> bool:
        type(self).callbacks.append("__ge__")
        return False


class _HostileNumpyInt(np.int64):
    """Caller-defined NumPy integer subclass with executable numeric hooks."""

    callbacks: list[str] = []

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        return 10

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        return 10

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "10"

    def __eq__(self, other: object) -> bool:
        type(self).callbacks.append("__eq__")
        return False

    def __hash__(self) -> int:
        type(self).callbacks.append("__hash__")
        return 10

    def __lt__(self, other: object) -> bool:
        type(self).callbacks.append("__lt__")
        return False

    def __le__(self, other: object) -> bool:
        type(self).callbacks.append("__le__")
        return False

    def __gt__(self, other: object) -> bool:
        type(self).callbacks.append("__gt__")
        return False

    def __ge__(self, other: object) -> bool:
        type(self).callbacks.append("__ge__")
        return False


def _prompt(**overrides: Any):
    """Build one valid prompt while allowing one targeted control override."""
    values: dict[str, Any] = {
        "prompt_id": "argument_prompt",
        "task_family_id": "essay_review",
        "prompt_content_fingerprint": "1" * 64,
        "language_id": "english_language",
        "genre_id": "argument_genre",
        "maximum_response_characters": 5_000,
        "maximum_response_units": 1_000,
    }
    values.update(overrides)
    return build_essay_prompt(**values)


def _submission(prompt_value, **overrides: Any):
    """Build one valid submission while allowing one targeted control override."""
    values: dict[str, Any] = {
        "submission_id": "essay_submission",
        "prompt": prompt_value,
        "respondent_id": "sample_respondent",
        "response_id": "essay_response",
        "response_content_fingerprint": "2" * 64,
        "response_character_count": 800,
        "response_unit_count": 120,
    }
    values.update(overrides)
    return build_essay_submission(**values)


def _evidence_reference() -> EvidenceReference:
    """Return one valid source-text-free evidence reference."""
    return EvidenceReference(
        source_id="essay_response",
        span_id="response_span",
        content_fingerprint="3" * 64,
        evidence_role=EvidenceRole.SUPPORTING,
    )


def _invoke_integer_control(name: str, value: Any) -> None:
    """Exercise one public essay integer control with otherwise valid inputs."""
    if name in {"maximum_response_characters", "maximum_response_units"}:
        _prompt(**{name: value})
        return

    prompt_value = _prompt()
    if name in {"response_character_count", "response_unit_count"}:
        _submission(prompt_value, **{name: value})
        return

    submission_value = _submission(prompt_value)
    offsets = {"start_offset": 10, "end_offset": 30}
    offsets[name] = value
    build_essay_response_evidence(
        prompt=prompt_value,
        submission=submission_value,
        evidence_reference=_evidence_reference(),
        evidence_kind=EssayEvidenceKind.RESPONSE_SPAN,
        **offsets,
    )


_INTEGER_CONTROLS = (
    "maximum_response_characters",
    "maximum_response_units",
    "response_character_count",
    "response_unit_count",
    "start_offset",
    "end_offset",
)

_HOSTILE_FACTORIES: tuple[Callable[[], Any], ...] = (
    _HostileIndex,
    lambda: _HostileInt(10),
    lambda: _HostileNumpyInt(10),
)


@pytest.mark.parametrize("control_name", _INTEGER_CONTROLS)
@pytest.mark.parametrize("hostile_factory", _HOSTILE_FACTORIES)
def test_essay_integer_controls_reject_caller_callbacks(
    control_name: str,
    hostile_factory: Callable[[], Any],
) -> None:
    """Reject controls before caller conversion, comparison, equality, or hashing."""
    hostile = hostile_factory()
    hostile_type = type(hostile)
    hostile_type.callbacks.clear()

    with pytest.raises(AssessmentSpecError) as caught:
        _invoke_integer_control(control_name, hostile)

    assert caught.value.code == f"invalid_{control_name}"
    assert hostile_type.callbacks == []


def test_essay_integer_controls_preserve_genuine_numpy_scalars() -> None:
    """Genuine NumPy integer scalars normalize to built-in immutable integers."""
    prompt_value = _prompt(
        maximum_response_characters=np.int64(5_000),
        maximum_response_units=np.uint32(1_000),
    )
    submission_value = _submission(
        prompt_value,
        response_character_count=np.int32(800),
        response_unit_count=np.uint16(120),
    )
    evidence_value = build_essay_response_evidence(
        prompt=prompt_value,
        submission=submission_value,
        evidence_reference=_evidence_reference(),
        evidence_kind=EssayEvidenceKind.RESPONSE_SPAN,
        start_offset=np.int16(10),
        end_offset=np.uint64(30),
    )

    assert type(prompt_value.maximum_response_characters) is int
    assert type(prompt_value.maximum_response_units) is int
    assert type(submission_value.response_character_count) is int
    assert type(submission_value.response_unit_count) is int
    assert type(evidence_value.start_offset) is int
    assert type(evidence_value.end_offset) is int
