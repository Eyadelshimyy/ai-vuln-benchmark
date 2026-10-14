"""Regression tests for inert enterprise evidence integer validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.enterprise_issue import (
    EnterpriseAssertionKind,
    EnterpriseSourceRecord,
    EvidenceSpanRecord,
)


class _HostileIndex:
    """Integer protocol provider that records any caller callback execution."""

    callbacks: ClassVar[list[str]] = []

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

    callbacks: ClassVar[list[str]] = []

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

    callbacks: ClassVar[list[str]] = []

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


def _source(character_count: Any = 240) -> EnterpriseSourceRecord:
    """Build one valid source record with one targeted count override."""
    return EnterpriseSourceRecord(
        source_id="customer_report",
        source_family_id="customer_feedback",
        source_content_fingerprint="1" * 64,
        source_character_count=character_count,
        metadata={},
    )


def _span(*, start_offset: Any = 10, end_offset: Any = 34) -> EvidenceSpanRecord:
    """Build one valid evidence span with targeted offset overrides."""
    source = _source()
    return EvidenceSpanRecord(
        source_id=source.source_id,
        source_record_fingerprint=source.source_record_fingerprint,
        span_id="reported_deadline",
        span_content_fingerprint="2" * 64,
        assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
        start_offset=start_offset,
        end_offset=end_offset,
        metadata={},
    )


def _invoke_integer_control(name: str, value: Any) -> None:
    """Exercise one public enterprise integer control with valid peer fields."""
    if name == "source_character_count":
        _source(value)
        return
    offsets = {"start_offset": 10, "end_offset": 34}
    offsets[name] = value
    _span(**offsets)


_INTEGER_CONTROLS = ("source_character_count", "start_offset", "end_offset")
_HOSTILE_FACTORIES: tuple[Callable[[], Any], ...] = (
    _HostileIndex,
    lambda: _HostileInt(10),
    lambda: _HostileNumpyInt(10),
)


@pytest.mark.parametrize("control_name", _INTEGER_CONTROLS)
@pytest.mark.parametrize("hostile_factory", _HOSTILE_FACTORIES)
def test_enterprise_integer_controls_reject_caller_callbacks(
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


def test_enterprise_integer_controls_preserve_genuine_numpy_scalars() -> None:
    """Genuine NumPy integer scalars normalize to built-in immutable integers."""
    source = _source(np.uint32(240))
    span = EvidenceSpanRecord(
        source_id=source.source_id,
        source_record_fingerprint=source.source_record_fingerprint,
        span_id="reported_deadline",
        span_content_fingerprint="2" * 64,
        assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
        start_offset=np.int16(10),
        end_offset=np.uint64(34),
        metadata={},
    )

    assert type(source.source_character_count) is int
    assert type(span.start_offset) is int
    assert type(span.end_offset) is int
