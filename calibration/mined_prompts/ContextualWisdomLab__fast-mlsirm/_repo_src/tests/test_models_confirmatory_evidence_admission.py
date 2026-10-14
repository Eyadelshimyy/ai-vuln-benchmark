"""Regression coverage for callback-free confirmatory loading-pattern admission."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.models import _resolve_model, confirmatory


def test_confirmatory_rejects_top_level_array_provider_before_callback():
    """Do not execute caller ``__array__`` while admitting model structure."""

    calls: list[str] = []

    class HostileArray:
        def __array__(self, *args, **kwargs):
            calls.append("__array__")
            raise AssertionError("array conversion callback executed")

    with pytest.raises(ValueError, match="confirmatory loading_pattern"):
        confirmatory(HostileArray())

    assert calls == []


def test_confirmatory_rejects_container_subclass_before_array_callback():
    """Reject callback-bearing sequence subclasses before NumPy sees them."""

    calls: list[str] = []

    class HostileList(list):
        def __array__(self, *args, **kwargs):
            calls.append("__array__")
            raise AssertionError("array conversion callback executed")

    value = HostileList([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="confirmatory loading_pattern"):
        confirmatory(value)

    assert calls == []


def test_confirmatory_rejects_nested_numeric_provider_before_callback():
    """Reject nested conversion providers without invoking ``__float__``."""

    calls: list[str] = []

    class HostileFloat:
        def __float__(self) -> float:
            calls.append("__float__")
            raise AssertionError("numeric conversion callback executed")

    with pytest.raises(ValueError, match="numeric 0 or 1"):
        confirmatory([[HostileFloat(), 0]])

    assert calls == []


def test_confirmatory_preserves_trusted_builtin_and_numpy_scalar_rows():
    """Keep ordinary exact rows and concrete NumPy scalar evidence compatible."""

    spec = confirmatory(
        [
            [np.bool_(True), np.int8(0)],
            (np.float32(0.0), np.uint16(1)),
        ]
    )

    assert spec.loading_pattern.dtype == np.int64
    np.testing.assert_array_equal(spec.loading_pattern, np.array([[1, 0], [0, 1]]))
    assert not spec.loading_pattern.flags.writeable


def test_confirmatory_preserves_exact_numpy_rows_inside_builtin_container():
    """Exact numeric ndarray rows are inert and remain supported."""

    spec = confirmatory(
        [
            np.array([1, 0], dtype=np.int16),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )

    np.testing.assert_array_equal(spec.loading_pattern, np.array([[1, 0], [0, 1]]))


def test_confirmatory_giant_integer_uses_stable_binary_error():
    """Do not route arbitrary-size exact integers through NumPy finiteness ufuncs."""

    with pytest.raises(ValueError, match="finite and exactly 0 or 1"):
        confirmatory([[10**400]])


def test_confirmatory_n_dims_replays_structure_before_shape_callback():
    """Reject rebound structure before reading caller-controlled shape metadata."""

    calls: list[str] = []

    class HostilePattern:
        @property
        def shape(self):
            calls.append("shape")
            raise AssertionError("shape callback executed")

    spec = confirmatory([[1, 0], [0, 1]])
    object.__setattr__(spec, "loading_pattern", HostilePattern())

    with pytest.raises(ValueError, match="confirmatory model loading_pattern"):
        _ = spec.n_dims

    assert calls == []


def test_resolve_confirmatory_replays_structure_before_shape_callback():
    """Model resolution must not read shape from rebound caller evidence."""

    calls: list[str] = []

    class HostilePattern:
        @property
        def shape(self):
            calls.append("shape")
            raise AssertionError("shape callback executed")

    spec = confirmatory([[1], [1]])
    object.__setattr__(spec, "loading_pattern", HostilePattern())

    with pytest.raises(ValueError, match="confirmatory model loading_pattern"):
        _resolve_model(spec, 2)

    assert calls == []
