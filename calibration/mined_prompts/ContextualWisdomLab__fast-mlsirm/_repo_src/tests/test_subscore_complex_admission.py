"""Regression contracts for subscore scientific-evidence admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.subscores import subscore_analysis


def _unexpected_core_discovery():
    """Fail if rejected evidence reaches the compiled-core discovery boundary."""

    raise AssertionError("compiled core must not be discovered for invalid evidence")


class _HostileArrayProvider:
    """Array provider whose protocol must never run during admission."""

    def __init__(self, value: object) -> None:
        self.value = value
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        self.calls += 1
        raise AssertionError("caller __array__ must not execute during admission")


class _HostileNumericProvider:
    """Numeric provider whose conversion callback must never run."""

    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("caller __float__ must not execute during admission")


class _CaptureCore:
    """Capture canonical Rust-boundary payloads without doing numeric work."""

    def __init__(self) -> None:
        self.payload: tuple[np.ndarray, int, int, list[int]] | None = None

    def subscore_analysis(self, values, n_persons, n_items, groups):
        self.payload = (values, n_persons, n_items, groups)
        raise RuntimeError("capture subscore payload")


def _valid_responses() -> np.ndarray:
    """Return a complete response matrix satisfying the Python shape boundary."""

    return np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
            [2.0, 3.0, 4.0, 2.0],
        ],
        dtype=np.float64,
    )


def test_subscore_rejects_response_array_provider_before_callback(monkeypatch) -> None:
    """Caller protocols cannot synthesize the person-by-item response evidence."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = _HostileArrayProvider(_valid_responses())

    with pytest.raises(ValueError, match="responses must be real-numeric evidence"):
        subscore_analysis(responses, np.array([0, 0, 1, 1], dtype=np.int64))

    assert responses.calls == 0


def test_subscore_rejects_group_array_provider_before_callback(monkeypatch) -> None:
    """Caller protocols cannot synthesize the item-to-subscale partition."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    groups = _HostileArrayProvider(np.array([0, 0, 1, 1], dtype=np.int64))

    with pytest.raises(ValueError, match="groups must be real-numeric evidence"):
        subscore_analysis(_valid_responses(), groups)

    assert groups.calls == 0


def test_subscore_rejects_nested_numeric_provider_before_callback(monkeypatch) -> None:
    """Built-in containers cannot smuggle caller numeric conversion protocols."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    hostile = _HostileNumericProvider()
    responses = [[0.0, 1.0, 2.0, 0.0], [1.0, hostile, 3.0, 1.0], [2.0, 3.0, 4.0, 2.0]]

    with pytest.raises(ValueError, match="responses must be real-numeric evidence"):
        subscore_analysis(responses, [0, 0, 1, 1])

    assert hostile.calls == 0


def test_subscore_preserves_trusted_builtin_and_numpy_scalar_sequences(monkeypatch) -> None:
    """Trusted plain sequences still reach Rust as canonical numeric payloads."""

    core = _CaptureCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    responses = [
        [np.float32(0), np.int16(1), np.uint8(2), 0.0],
        [1, np.float64(2), np.int32(3), np.float32(1)],
        [np.uint16(2), 3.0, np.float32(4), np.int8(2)],
    ]
    groups = (np.int8(0), np.float32(0), np.uint8(1), np.float64(1.0))

    with pytest.raises(RuntimeError, match="capture subscore payload"):
        subscore_analysis(responses, groups)

    assert core.payload is not None
    values, n_persons, n_items, normalized_groups = core.payload
    assert type(values) is np.ndarray
    assert values.dtype == np.float64
    assert values.shape == (12,)
    assert (n_persons, n_items) == (3, 4)
    assert normalized_groups == [0, 0, 1, 1]
    assert all(type(value) is int for value in normalized_groups)


def test_subscore_preserves_exact_numpy_rows_inside_builtin_container(monkeypatch) -> None:
    """Exact inert NumPy rows remain compatible without reopening array protocols."""

    core = _CaptureCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    responses = [
        np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float32),
        np.array([1.0, 2.0, 3.0, 1.0], dtype=np.float64),
        np.array([2, 3, 4, 2], dtype=np.int16),
    ]

    with pytest.raises(RuntimeError, match="capture subscore payload"):
        subscore_analysis(responses, [0, 0, 1, 1])

    assert core.payload is not None
    values, n_persons, n_items, normalized_groups = core.payload
    assert values.dtype == np.float64
    assert values.shape == (12,)
    assert (n_persons, n_items) == (3, 4)
    assert normalized_groups == [0, 0, 1, 1]


def test_subscore_rejects_complex_responses_before_real_narrowing(monkeypatch) -> None:
    """Imaginary response evidence cannot be projected onto a real score matrix."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = _valid_responses().astype(np.complex128)
    responses[1, 2] += 1.0j

    with pytest.raises(ValueError, match="responses must be real-valued"):
        subscore_analysis(responses, np.array([0, 0, 1, 1], dtype=np.int64))


def test_subscore_rejects_complex_groups_before_real_narrowing(monkeypatch) -> None:
    """Imaginary group evidence cannot be projected onto another partition."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    groups = np.array([0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j, 1.0 + 0.0j])

    with pytest.raises(ValueError, match="groups must be real-valued"):
        subscore_analysis(_valid_responses(), groups)
