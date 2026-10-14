"""Regression coverage for shared scoring enum admission callback safety."""

from __future__ import annotations

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, EngineKind
from scoring_execution_fixtures import automated_engine


class _HostileEnumText(str):
    """String subclass that records any Enum lookup callback attempt."""

    callback_count = 0

    def __hash__(self) -> int:
        """Fail if enum admission attempts to hash hostile wire text."""
        type(self).callback_count += 1
        raise AssertionError("hostile enum text must not be hashed")

    def __eq__(self, other: object) -> bool:
        """Fail if enum admission attempts to compare hostile wire text."""
        type(self).callback_count += 1
        raise AssertionError("hostile enum text must not be compared")


class _HostileEnumObject:
    """Non-text value that records any Enum lookup callback attempt."""

    callback_count = 0

    def __hash__(self) -> int:
        """Fail if enum admission attempts to hash an arbitrary object."""
        type(self).callback_count += 1
        raise AssertionError("hostile enum object must not be hashed")

    def __eq__(self, other: object) -> bool:
        """Fail if enum admission attempts to compare an arbitrary object."""
        type(self).callback_count += 1
        raise AssertionError("hostile enum object must not be compared")


def test_engine_kind_rejects_string_subclass_before_enum_lookup() -> None:
    """Reject hostile serialized enum text before hashing or equality callbacks."""
    _HostileEnumText.callback_count = 0

    with pytest.raises(AssessmentSpecError) as exc_info:
        automated_engine(engine_kind=_HostileEnumText("automated_engine"))

    assert exc_info.value.code == "invalid_engine_kind"
    assert exc_info.value.path == "$.engine_kind"
    assert _HostileEnumText.callback_count == 0


def test_engine_kind_rejects_non_text_object_before_enum_lookup() -> None:
    """Reject arbitrary objects before Enum lookup can hash or compare them."""
    _HostileEnumObject.callback_count = 0

    with pytest.raises(AssessmentSpecError) as exc_info:
        automated_engine(engine_kind=_HostileEnumObject())

    assert exc_info.value.code == "invalid_engine_kind"
    assert exc_info.value.path == "$.engine_kind"
    assert _HostileEnumObject.callback_count == 0


def test_engine_kind_preserves_builtin_string_and_member_forms() -> None:
    """Keep exact built-in wire text and exact enum members compatible."""
    assert (
        automated_engine(engine_kind="automated_engine").engine_kind
        is EngineKind.AUTOMATED
    )
    assert (
        automated_engine(engine_kind=EngineKind.AUTOMATED).engine_kind
        is EngineKind.AUTOMATED
    )
