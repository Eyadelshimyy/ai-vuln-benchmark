"""Callback-safety regressions for paired rating-range category controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.rating_range as rating_range


_AUTOMATED = np.array([0, 1, 2, 1], dtype=np.uint32)
_REFERENCE = np.array([0, 1, 2, 2], dtype=np.uint32)


def test_category_count_rejects_python_int_subclass_before_callbacks(monkeypatch) -> None:
    """A Python integer subclass must fail before coercion or native dispatch."""

    calls: list[str] = []
    loader_calls = 0

    class HostileInt(int):
        def __int__(self) -> int:
            calls.append("__int__")
            raise AssertionError("integer conversion callback executed")

        def __repr__(self) -> str:
            calls.append("__repr__")
            raise AssertionError("representation callback executed")

    def forbidden_core():
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("RATING_RANGE_CORE_MUST_NOT_RUN")

    monkeypatch.setattr(rating_range, "rating_range_core", forbidden_core)

    with pytest.raises(ValueError, match="category_count"):
        rating_range.paired_rating_range_evidence(
            _AUTOMATED,
            _REFERENCE,
            category_count=HostileInt(3),
        )

    assert calls == []
    assert loader_calls == 0


def test_category_count_rejects_numpy_integer_subclass_before_callbacks(monkeypatch) -> None:
    """A NumPy integer subclass must fail without hashing or coercing its type."""

    calls: list[str] = []
    loader_calls = 0

    class HostileMeta(type):
        def __hash__(cls) -> int:
            calls.append("type-__hash__")
            raise AssertionError("type hash callback executed")

        def __eq__(cls, other) -> bool:
            calls.append("type-__eq__")
            raise AssertionError("type equality callback executed")

    class HostileNumpyInt(np.int64, metaclass=HostileMeta):
        def __int__(self) -> int:
            calls.append("__int__")
            raise AssertionError("integer conversion callback executed")

        def __repr__(self) -> str:
            calls.append("__repr__")
            raise AssertionError("representation callback executed")

    def forbidden_core():
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("RATING_RANGE_CORE_MUST_NOT_RUN")

    monkeypatch.setattr(rating_range, "rating_range_core", forbidden_core)

    with pytest.raises(ValueError, match="category_count"):
        rating_range.paired_rating_range_evidence(
            _AUTOMATED,
            _REFERENCE,
            category_count=HostileNumpyInt(3),
        )

    assert calls == []
    assert loader_calls == 0


def test_category_count_preserves_genuine_numpy_integer_scalar() -> None:
    """A genuine NumPy integer scalar remains a supported public control."""

    result = rating_range.paired_rating_range_evidence(
        _AUTOMATED,
        _REFERENCE,
        category_count=np.int64(3),
    )

    assert result.sample_size == 4
