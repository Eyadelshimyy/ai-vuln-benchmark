"""Regression tests for validation-before-native-discovery in DETECT adapters."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.detect import _trusted_numeric_array, detect_analysis, dimtest


def _unexpected_core_discovery():
    """Fail if invalid public input reaches compiled-core discovery."""

    raise AssertionError("compiled core must not be discovered for invalid public input")


class _HostileFloat:
    """Sentinel proving DETECT does not numerically coerce object storage."""

    def __float__(self):
        raise AssertionError("caller numeric conversion must not execute")


class _HostileArrayProvider:
    """Sentinel proving DETECT rejects array protocols before NumPy dispatch."""

    def __array__(self, *args, **kwargs):
        raise AssertionError("caller array conversion must not execute")


def test_detect_rejects_invalid_response_shape_before_core_discovery(monkeypatch):
    """Malformed DETECT responses fail locally before touching the native loader."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="2-D persons x items"):
        detect_analysis(np.zeros(6), np.array([0, 1]))


def test_detect_rejects_nonbinary_responses_before_core_discovery(monkeypatch):
    """Out-of-domain DETECT responses fail before native discovery."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="exactly 0 or 1"):
        detect_analysis(responses, np.array([0, 1]))


def test_detect_rejects_complex_responses_before_lossy_coercion(monkeypatch):
    """Imaginary response evidence is never projected onto binary real data."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0.0 + 1.0j, 0.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        detect_analysis(responses, np.array([0, 1]))


def test_detect_rejects_complex_sequence_with_complex_diagnostic(monkeypatch):
    """Complex list evidence reports the same real-valued contract as arrays."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = [[0.0 + 1.0j, 0.0], [1.0, 0.0]]

    with pytest.raises(ValueError, match="responses must be real-valued"):
        detect_analysis(responses, [0, 1])


def test_detect_rejects_object_responses_before_element_coercion(monkeypatch):
    """Object response storage fails before caller numeric conversion callbacks."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array([[_HostileFloat(), 0], [1, 0]], dtype=object)

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        detect_analysis(responses, np.array([0, 1]))


def test_detect_rejects_array_provider_responses_without_callbacks(monkeypatch):
    """Arbitrary response array providers fail before caller conversion callbacks."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        detect_analysis(_HostileArrayProvider(), np.array([0, 1]))


def test_detect_rejects_oversized_exact_ndarray_before_extra_materialization():
    """Oversized exact arrays fail with an explicit resource-limit diagnostic."""
    oversized = np.broadcast_to(
        np.array(0, dtype=np.uint8),
        (1, 20_000_001),
    )

    with pytest.raises(
        ValueError,
        match="responses exceed the 20,000,000-cell resource limit",
    ):
        detect_analysis(oversized, np.array([0], dtype=np.int64))


def test_detect_rejects_complex_cluster_before_lossy_coercion(monkeypatch):
    """Imaginary partition labels cannot be projected onto a real partition."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    cluster = np.array([0.0 + 1.0j, 1.0], dtype=np.complex128)

    with pytest.raises(ValueError, match="cluster labels must be real integers"):
        detect_analysis(responses, cluster)


def test_detect_rejects_object_cluster_before_element_coercion(monkeypatch):
    """Object partition storage fails before caller numeric conversion callbacks."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    cluster = np.array([_HostileFloat(), 1], dtype=object)

    with pytest.raises(ValueError, match="cluster labels must be a numeric array"):
        detect_analysis(responses, cluster)


def test_detect_rejects_array_provider_cluster_without_callbacks(monkeypatch):
    """Arbitrary cluster array providers fail before caller conversion callbacks."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="cluster labels must be a numeric array"):
        detect_analysis(responses, _HostileArrayProvider())


def test_dimtest_rejects_invalid_partition_before_core_discovery(monkeypatch):
    """Malformed DIMTEST item indices fail locally before touching the native loader."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.zeros((40, 8), dtype=np.float64)

    with pytest.raises(ValueError, match="at1 indices must be non-empty integers"):
        dimtest(responses, np.array([], dtype=np.int64), np.array([2, 3]))


def test_dimtest_rejects_array_provider_responses_without_callbacks(monkeypatch):
    """DIMTEST response providers fail before caller array conversion callbacks."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        dimtest(_HostileArrayProvider(), [0, 1, 2, 3], [4, 5, 6, 7])


@pytest.mark.parametrize("slot", ["at1", "at2"])
def test_dimtest_rejects_array_provider_index_sets_without_callbacks(monkeypatch, slot):
    """DIMTEST subtest providers fail before caller array conversion callbacks."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.zeros((40, 10), dtype=np.float64)
    at1 = _HostileArrayProvider() if slot == "at1" else [0, 1, 2, 3]
    at2 = _HostileArrayProvider() if slot == "at2" else [4, 5, 6, 7]

    with pytest.raises(ValueError, match=rf"{slot} indices must be a numeric array"):
        dimtest(responses, at1, at2)


def test_detect_valid_input_discovers_core_only_at_dispatch_boundary(monkeypatch):
    """A valid request still discovers the compiled core exactly when dispatch is needed."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(RuntimeError, match="detect_analysis requires the compiled Rust core"):
        detect_analysis(responses, np.array([0, 1], dtype=np.int64))

    assert calls == 1


def test_detect_accepts_integer_valued_float_cluster_at_dispatch_boundary(monkeypatch):
    """Ordinary integer-valued floating labels retain their historical contract."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(RuntimeError, match="detect_analysis requires the compiled Rust core"):
        detect_analysis(responses, np.array([10.0, 20.0], dtype=np.float64))

    assert calls == 1


def test_detect_accepts_builtin_sequences_at_dispatch_boundary(monkeypatch):
    """Plain response/cluster sequences keep the historical array-like contract."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = [[np.int16(0), np.float32(1.0)], [1, 0]]
    cluster = (np.uint8(10), np.float64(20.0))

    with pytest.raises(RuntimeError, match="detect_analysis requires the compiled Rust core"):
        detect_analysis(responses, cluster)

    assert calls == 1


def test_detect_accepts_platform_numpy_integer_aliases_at_dispatch_boundary(monkeypatch):
    """Concrete NumPy integer aliases retain the historical numeric sequence contract."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = [[np.longlong(0), np.ulonglong(1)], [np.intp(1), np.uintp(0)]]
    cluster = (np.longlong(10), np.ulonglong(20))

    with pytest.raises(RuntimeError, match="detect_analysis requires the compiled Rust core"):
        detect_analysis(responses, cluster)

    assert calls == 1


def test_dimtest_valid_input_discovers_core_only_at_dispatch_boundary(monkeypatch):
    """A valid DIMTEST request still reaches core discovery after validation."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = np.zeros((40, 8), dtype=np.float64)

    with pytest.raises(RuntimeError, match="dimtest requires the compiled Rust core"):
        dimtest(responses, np.array([0, 1]), np.array([2, 3]))

    assert calls == 1


def test_dimtest_accepts_builtin_sequences_at_dispatch_boundary(monkeypatch):
    """Plain DIMTEST evidence sequences remain accepted up to Rust dispatch."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = tuple(tuple(np.uint8(0) for _ in range(10)) for _ in range(40))
    at1 = [np.int16(0), np.int32(1), np.int64(2), 3]
    at2 = (np.uint8(4), np.uint16(5), np.float32(6.0), np.float64(7.0))

    with pytest.raises(RuntimeError, match="dimtest requires the compiled Rust core"):
        dimtest(responses, at1, at2)

    assert calls == 1
