"""Trust-boundary regressions for graded-response public admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.grm import fit_grm


class _ArraySentinel:
    """Caller array provider that invalid controls must not materialize."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __array__(self, *args, **kwargs):
        type(self).calls += 1
        raise AssertionError("caller response materialization executed")


class _HostileFloat:
    """Numeric protocol provider that package admission must never execute."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("caller numeric conversion executed")


class _HostileInt(int):
    """Integer subclass whose coercion hooks must remain unreachable."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        return int.__int__(self)

    def __float__(self) -> float:
        type(self).calls += 1
        return float(int.__int__(self))


def _unexpected_core_discovery():
    """Fail if rejected public input reaches compiled-core discovery."""

    raise AssertionError("compiled core must not be discovered for invalid GRM input")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n_cat": 1}, "n_cat must be between"),
        # #1929: no node-count cap; q=13 is now accepted, only < 1 is not.
        ({"q": 0}, "q must be >= 1"),
        ({"max_iter": 0}, "max_iter must be between"),
        ({"tol": 0.0}, "tol must be finite and > 0"),
        ({"xi_points": 0}, "xi_points must be between"),
        ({"xi_seed": -1}, "xi_seed must be in"),
    ],
)
def test_invalid_semantic_controls_fail_before_response_materialization(
    monkeypatch, override, message
):
    """Resource/model controls are rejected before caller response array protocols."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _ArraySentinel.reset()
    kwargs = {"n_cat": 3, "model": 1, "max_iter": 500, "tol": 1e-6, "xi_points": 4000, "xi_seed": 0x9E3779B97F4A7C15}
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        fit_grm(_ArraySentinel(), **kwargs)

    assert _ArraySentinel.calls == 0


@pytest.mark.parametrize("name", ["n_cat", "q", "max_iter", "xi_points", "xi_seed"])
def test_hostile_integer_subclasses_fail_without_callbacks(monkeypatch, name):
    """Integer-like controls do not invoke caller subclass coercion hooks."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _HostileInt.reset()
    kwargs = {"n_cat": 3, "model": 1, "max_iter": 500, "tol": 1e-6, "xi_points": 4000, "xi_seed": 0x9E3779B97F4A7C15}
    value = 21 if name == "q" else 3
    if name == "max_iter":
        value = 5
    elif name == "xi_points":
        value = 100
    elif name == "xi_seed":
        value = 7
    kwargs[name] = _HostileInt(value)

    with pytest.raises(ValueError):
        fit_grm(_ArraySentinel(), **kwargs)

    assert _HostileInt.calls == 0


def test_hostile_tolerance_fails_without_numeric_callback(monkeypatch):
    """Arbitrary real protocol providers are outside the trusted tolerance domain."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _HostileFloat.reset()

    with pytest.raises(ValueError, match="tol must be a real number"):
        fit_grm(_ArraySentinel(), n_cat=3, tol=_HostileFloat(), model=1, max_iter=500, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)

    assert _HostileFloat.calls == 0


def test_complex_responses_fail_before_lossy_coercion_or_native_discovery(monkeypatch):
    """Imaginary category evidence is never projected onto real GRM responses."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0.0 + 1.0j, 1.0], [1.0, 2.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        fit_grm(responses, n_cat=3, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_object_responses_fail_before_element_numeric_coercion(monkeypatch):
    """Object storage is rejected without executing caller numeric conversion."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _HostileFloat.reset()
    responses = np.array([[_HostileFloat(), 1], [1, 2]], dtype=object)

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        fit_grm(responses, n_cat=3, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)

    assert _HostileFloat.calls == 0


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinite_responses_are_not_silently_reclassified_as_missing(monkeypatch, bad):
    """Only NaN/negative values use the documented missingness convention."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0.0, bad], [1.0, 2.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="responses must not contain infinity"):
        fit_grm(responses, n_cat=3, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_exact_numpy_controls_and_real_responses_reach_dispatch_boundary(monkeypatch):
    """Supported concrete NumPy scalars retain compatibility after normalization."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [np.nan, 0.0]],
        dtype=np.float64,
    )

    with pytest.raises(RuntimeError, match="fit_grm requires the compiled Rust core"):
        fit_grm(
            responses,
            n_cat=np.int64(3),
            q=np.int64(21),
            max_iter=np.int64(5),
            tol=np.float64(1e-6),
            xi_points=np.int64(100),
            xi_seed=np.uint64(7), model=1
        )

    assert calls == 1
