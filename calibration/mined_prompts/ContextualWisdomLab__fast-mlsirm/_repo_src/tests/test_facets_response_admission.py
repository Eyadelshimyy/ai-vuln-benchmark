"""Trust-boundary regressions for many-facet response evidence admission."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import fast_mlsirm.facets as facets_module
from fast_mlsirm import fitstats
from fast_mlsirm.facets import fit_facets


@dataclass
class _CallbackCounter:
    """Record caller-controlled evidence conversion attempts."""

    calls: int = 0

    def hit(self) -> None:
        """Record and fail if package validation executes caller conversion."""

        self.calls += 1
        raise AssertionError("caller evidence conversion executed")


class _HostileFloat:
    """Object-array element whose real conversion must never execute."""

    def __init__(self, counter: _CallbackCounter) -> None:
        self._counter = counter

    def __float__(self) -> float:
        self._counter.hit()
        return 0.0


class _HostileArrayProvider:
    """Top-level provider whose NumPy array callback must never execute."""

    def __init__(self, counter: _CallbackCounter) -> None:
        self._counter = counter

    def __array__(self, dtype=None, copy=None):
        self._counter.hit()
        return np.zeros((1, 1, 1), dtype=np.float64)


def _unexpected_core_discovery():
    """Fail if invalid evidence reaches compiled-core discovery."""

    raise AssertionError("compiled core must not be discovered for invalid evidence")


def test_fit_facets_rejects_complex_responses_before_lossy_coercion(monkeypatch):
    """Imaginary rating evidence is not projected onto real response categories."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array(
        [[[0.0 + 1.0j], [1.0]], [[1.0], [0.0]]], dtype=np.complex128
    )

    with pytest.raises(ValueError, match="responses must be real-valued"):
        fit_facets(responses, n_cat=2, q_theta=41, max_iter=10, tol=1e-6)


def test_fit_facets_rejects_object_storage_before_element_coercion(monkeypatch):
    """Object ratings fail before caller-provided per-element conversions."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    counter = _CallbackCounter()
    responses = np.array(
        [[[_HostileFloat(counter)], [1]], [[1], [0]]], dtype=object
    )

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        fit_facets(responses, n_cat=2, q_theta=41, max_iter=10, tol=1e-6)

    assert counter.calls == 0


def test_fit_facets_rejects_array_provider_before_numpy_callback(monkeypatch):
    """Arbitrary array providers cannot synthesize the observed rating cube."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    counter = _CallbackCounter()

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        fit_facets(
            _HostileArrayProvider(counter),
            n_cat=2,
            q_theta=41,
            max_iter=10,
            tol=1e-6,
        )

    assert counter.calls == 0


def test_fit_facets_rejects_oversized_exact_array_before_value_work(monkeypatch):
    """Logical size is bounded before value-wise NumPy work on broadcast ratings."""

    responses = np.broadcast_to(
        np.array([[[0.0]]], dtype=np.float64),
        (20_000_001, 1, 1),
    )

    def unexpected_value_work(*args, **kwargs):
        del args, kwargs
        raise AssertionError("value-wise NumPy work ran before facets resource admission")

    monkeypatch.setattr(facets_module.np, "iscomplexobj", unexpected_value_work)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="20,000,000"):
        fit_facets(responses, n_cat=2, q_theta=41, max_iter=10, tol=1e-6)


def test_fit_facets_bounds_builtin_tree_before_numpy_materialization(monkeypatch):
    """Built-in rating leaves are counted before sequence materialization."""

    monkeypatch.setattr(facets_module, "_MAX_FACETS_RESPONSE_CELLS", 3, raising=False)

    def unexpected_asarray(*args, **kwargs):
        del args, kwargs
        raise AssertionError("NumPy sequence materialization ran before facets resource admission")

    monkeypatch.setattr(facets_module.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = [[[0, 1]], [[1, 0]]]

    with pytest.raises(ValueError, match="3"):
        fit_facets(responses, n_cat=2, q_theta=41, max_iter=10, tol=1e-6)


def test_fit_facets_bounds_empty_container_fanout_before_numpy_materialization(monkeypatch):
    """Malformed empty-container fan-out cannot bypass bounded traversal work."""

    monkeypatch.setattr(facets_module, "_MAX_FACETS_RESPONSE_CELLS", 2, raising=False)

    def unexpected_asarray(*args, **kwargs):
        del args, kwargs
        raise AssertionError("NumPy materialization ran after unbounded container fan-out")

    monkeypatch.setattr(facets_module.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = [[[]] for _ in range(7)]

    with pytest.raises(ValueError, match="structural traversal budget"):
        fit_facets(responses, n_cat=2, q_theta=41, max_iter=10, tol=1e-6)


def test_fit_facets_valid_real_evidence_reaches_dispatch_boundary(monkeypatch):
    """Ordinary real ratings retain their historical native-dispatch contract."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = np.array(
        [[[0.0], [1.0]], [[1.0], [np.nan]]], dtype=np.float64
    )

    with pytest.raises(RuntimeError, match="fit_facets requires the compiled Rust core"):
        fit_facets(responses, n_cat=2, q_theta=41, max_iter=10, tol=1e-6)

    assert calls == 1


def test_fit_facets_trusted_builtin_sequence_evidence_reaches_dispatch(monkeypatch):
    """Exact list/tuple trees with trusted real scalars remain compatible."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = (
        ([np.int8(0)], [np.float32(1.0)]),
        ([np.uint8(1)], [np.nan]),
    )

    with pytest.raises(RuntimeError, match="fit_facets requires the compiled Rust core"):
        fit_facets(responses, n_cat=np.int8(2), q_theta=np.int16(41), max_iter=10, tol=1e-6)

    assert calls == 1
