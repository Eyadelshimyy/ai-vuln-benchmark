"""Regression coverage for parallel-analysis public data conversion errors."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.parallel_analysis import parallel_analysis


class _FloatBomb:
    """Object-array element that records any attempted real-number coercion."""

    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("OBJECT_ELEMENT_CONVERSION_MUST_NOT_RUN")


class _ArrayBomb:
    """Top-level array provider that must never execute during admission."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("ARRAY_PROTOCOL_MUST_NOT_RUN")


class _ListBomb(list):
    """Container subclass whose iteration must not run during admission."""

    calls = 0

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("CONTAINER_PROTOCOL_MUST_NOT_RUN")


def _forbid_native_discovery(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace Rust capability discovery with a sentinel and return its call log."""
    core_calls: list[str] = []

    def forbidden_core():
        core_calls.append("_core_module")
        raise AssertionError("RUST_DISCOVERY_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_core)
    return core_calls


def test_top_level_array_provider_is_rejected_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject caller array protocols before observed evidence can be synthesized."""
    core_calls = _forbid_native_discovery(monkeypatch)
    data = _ArrayBomb()

    with pytest.raises(ValueError, match="data must be numeric and convertible to float64"):
        parallel_analysis(data, n_iterations=1)

    assert data.calls == 0
    assert core_calls == []


def test_container_subclass_is_rejected_without_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject callback-bearing container identities before their protocols run."""
    core_calls = _forbid_native_discovery(monkeypatch)
    _ListBomb.calls = 0
    data = _ListBomb([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="data must be numeric and convertible to float64"):
        parallel_analysis(data, n_iterations=1)

    assert _ListBomb.calls == 0
    assert core_calls == []


def test_over_rank_builtin_tree_is_rejected_without_recursive_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known 2-D evidence must not recurse through arbitrarily deep containers."""
    core_calls = _forbid_native_discovery(monkeypatch)
    data: object = 0.0
    for _ in range(1500):
        data = [data]

    with pytest.raises(ValueError, match="data must be a 2-D persons x items array"):
        parallel_analysis(data, n_iterations=1)

    assert core_calls == []


def test_lossy_integer_matrix_is_rejected_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observed integer identities cannot change during Rust-f64 marshalling."""
    core_calls = _forbid_native_discovery(monkeypatch)
    data = np.array([[2**53 + 1, 0], [0, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="data must be exactly representable as float64"):
        parallel_analysis(data, n_iterations=1)

    assert core_calls == []


def test_lossy_longdouble_matrix_is_rejected_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extended-precision observations cannot be silently rounded to Rust f64."""
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform longdouble has no precision beyond float64")

    core_calls = _forbid_native_discovery(monkeypatch)
    wider = np.nextafter(np.longdouble(1.0), np.longdouble(2.0))
    assert np.longdouble(float(wider)) != wider
    data = np.array([[wider, np.longdouble(0.0)], [0.0, 1.0]], dtype=np.longdouble)

    with pytest.raises(ValueError, match="data must be exactly representable as float64"):
        parallel_analysis(data, n_iterations=1)

    assert core_calls == []


def test_mixed_builtin_matrix_rejects_lossy_scalar_before_numpy_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed built-in evidence cannot lose integer identity during dtype promotion."""
    core_calls = _forbid_native_discovery(monkeypatch)
    data = [
        [np.uint64(2**53 + 1), np.float64(0.0)],
        [np.int16(0), np.float64(1.0)],
    ]

    with pytest.raises(ValueError, match="data must be exactly representable as float64"):
        parallel_analysis(data, n_iterations=1)

    assert core_calls == []


def test_builtin_matrix_with_numpy_scalars_reaches_rust_as_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve inert built-in/NumPy scalar evidence at the Rust boundary."""
    captured: dict[str, object] = {}

    class _Core:
        def parallel_analysis(
            self,
            data,
            n_persons,
            n_items,
            n_iterations,
            centile,
            seed,
        ):
            captured["data"] = np.array(data, copy=True)
            captured["shape"] = (n_persons, n_items)
            captured["controls"] = (n_iterations, centile, seed)
            return {
                "retained": 1,
                "eigenvalues": [1.5, 0.5],
                "random_eigenvalues": [1.1, 0.9],
                "bias": [0.1, -0.1],
                "adjusted_eigenvalues": [1.4, 0.6],
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())
    data = [
        [np.float32(1.0), np.int16(0)],
        [np.uint8(0), np.float64(1.0)],
    ]

    result = parallel_analysis(data, n_iterations=np.int16(1), seed=np.uint8(2))

    assert result.retained == 1
    assert captured["shape"] == (2, 2)
    assert captured["controls"] == (1, 0, 2)
    assert isinstance(captured["data"], np.ndarray)
    assert captured["data"].dtype == np.float64
    np.testing.assert_array_equal(
        captured["data"], np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )


def test_non_numeric_data_conversion_fails_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize invalid storage to a package ValueError before Rust lookup."""
    core_calls = _forbid_native_discovery(monkeypatch)
    data = np.array([[object()]], dtype=object)

    with pytest.raises(ValueError, match="data must be numeric and convertible to float64"):
        parallel_analysis(data, n_iterations=1)

    assert core_calls == []


def test_complex_data_is_rejected_before_lossy_projection_or_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not discard imaginary observed evidence before Horn retention."""
    core_calls = _forbid_native_discovery(monkeypatch)
    data = np.array([[1.0 + 2.0j, 0.0], [0.0, 1.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="data must be real-valued"):
        parallel_analysis(data, n_iterations=1)

    assert core_calls == []


def test_object_storage_is_rejected_without_element_numeric_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject object storage before caller-defined ``__float__`` can execute."""
    core_calls = _forbid_native_discovery(monkeypatch)
    bomb = _FloatBomb()
    data = np.array([[bomb, bomb], [bomb, bomb]], dtype=object)

    with pytest.raises(ValueError, match="data must be numeric and convertible to float64"):
        parallel_analysis(data, n_iterations=1)

    assert bomb.calls == 0
    assert core_calls == []
