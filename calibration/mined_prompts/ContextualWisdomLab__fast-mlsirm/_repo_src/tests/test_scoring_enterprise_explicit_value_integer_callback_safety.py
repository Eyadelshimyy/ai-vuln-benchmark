"""Regression tests for callback-free explicit-value integer admission."""

from __future__ import annotations

import hashlib
from typing import Callable

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.enterprise_issue import (
    DeterministicExplicitValueParser,
    ExplicitValueKind,
    ExplicitValueRecord,
)

_SOURCE_FP = hashlib.sha256(b"explicit-value-source").hexdigest()
_SPAN_FP = hashlib.sha256(b"explicit-value-span").hexdigest()
_PARSER_FP = hashlib.sha256(b"explicit-value-parser").hexdigest()


def _hostile_integer(value: int) -> tuple[int, list[str]]:
    """Return an integer subclass that records comparison/coercion callbacks."""

    callbacks: list[str] = []

    class HostileInteger(int):
        def __lt__(self, other: object) -> bool:
            callbacks.append("lt")
            raise AssertionError("caller less-than callback executed")

        def __le__(self, other: object) -> bool:
            callbacks.append("le")
            raise AssertionError("caller less-equal callback executed")

        def __gt__(self, other: object) -> bool:
            callbacks.append("gt")
            raise AssertionError("caller greater-than callback executed")

        def __ge__(self, other: object) -> bool:
            callbacks.append("ge")
            raise AssertionError("caller greater-equal callback executed")

        def __int__(self) -> int:
            callbacks.append("int")
            raise AssertionError("caller integer callback executed")

        def __index__(self) -> int:
            callbacks.append("index")
            raise AssertionError("caller index callback executed")

    return HostileInteger(value), callbacks


def _record(*, start_offset: object = 2, end_offset: object = 12) -> ExplicitValueRecord:
    """Build one otherwise-valid date record around an offset under test."""

    return ExplicitValueRecord(
        source_id="customer_report",
        source_record_fingerprint=_SOURCE_FP,
        span_content_fingerprint=_SPAN_FP,
        start_offset=start_offset,  # type: ignore[arg-type]
        end_offset=end_offset,  # type: ignore[arg-type]
        value_kind=ExplicitValueKind.CALENDAR_DATE,
        normalized_payload={"calendar_date": "2026-09-30"},
        parser_revision_fingerprint=_PARSER_FP,
        metadata={},
    )


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("start_offset", lambda value: _record(start_offset=value)),
        ("end_offset", lambda value: _record(end_offset=value)),
    ],
)
def test_record_offsets_reject_integer_subclasses_before_callbacks(
    name: str,
    build: Callable[[object], ExplicitValueRecord],
) -> None:
    """Source-span offsets reject caller integer subclasses before comparison."""

    value, callbacks = _hostile_integer(4 if name == "start_offset" else 12)

    with pytest.raises(AssessmentSpecError) as caught:
        build(value)

    assert caught.value.code == f"invalid_{name}"
    assert callbacks == []


def test_parser_record_limit_rejects_integer_subclass_before_callbacks() -> None:
    """Parser record limits reject caller integer subclasses before comparison."""

    value, callbacks = _hostile_integer(8)

    with pytest.raises(AssessmentSpecError) as caught:
        DeterministicExplicitValueParser(maximum_records=value)  # type: ignore[arg-type]

    assert caught.value.code == "invalid_maximum_records"
    assert callbacks == []
