"""Callback-safety regressions for public RSM semantic controls."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.rsm import fit_rsm


class _ResponseSentinel:
    """Fail if rejected controls reach response materialization."""

    def __array__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("response materialization must not run for rejected controls")


class _HostileInt(int):
    """Integer subclass that fails if validation invokes callbacks."""

    callbacks = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter before one admission test."""
        cls.callbacks = 0

    @classmethod
    def _boom(cls, *args, **kwargs):  # noqa: ANN002, ANN003
        """Raise when package validation dispatches to caller behavior."""
        cls.callbacks += 1
        raise AssertionError("caller integer callback executed")

    __ge__ = _boom
    __le__ = _boom
    __gt__ = _boom
    __lt__ = _boom
    __hash__ = _boom
    __int__ = _boom
    __index__ = _boom


class _HostileFloat(float):
    """Floating-point subclass that fails if validation invokes callbacks."""

    callbacks = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter before one admission test."""
        cls.callbacks = 0

    @classmethod
    def _boom(cls, *args, **kwargs):  # noqa: ANN002, ANN003
        """Raise when package validation dispatches to caller behavior."""
        cls.callbacks += 1
        raise AssertionError("caller floating callback executed")

    __ge__ = _boom
    __le__ = _boom
    __gt__ = _boom
    __lt__ = _boom
    __float__ = _boom


class _HashTrap:
    """Object that fails if validation hashes or compares it."""

    callbacks = 0

    def __hash__(self) -> int:
        type(self).callbacks += 1
        raise AssertionError("caller hash callback executed")

    def __eq__(self, other: object) -> bool:
        type(self).callbacks += 1
        raise AssertionError("caller equality callback executed")


class _FakeCore:
    """Minimal Rust-core sentinel that records normalized RSM controls."""

    def __init__(self) -> None:
        """Initialize the call-recording slot."""
        self.controls: tuple[int, int, int, float] | None = None

    def fit_rsm(
        self,
        yy,
        observed,
        n_persons,
        n_items,
        n_cat,
        q_theta,
        max_iter,
        tol,
    ):
        """Record controls and return the smallest valid fit result."""
        self.controls = (n_cat, q_theta, max_iter, tol)
        return {
            "item_location": np.zeros(n_items),
            "thresholds": np.zeros(n_cat - 1),
            "theta": np.zeros(n_persons),
            "loglik_trace": np.zeros(1),
            "n_iter": 1,
            "converged": True,
            "n_parameters": n_items + n_cat - 2,
        }


def _responses() -> np.ndarray:
    """Return a small valid ordinal response matrix."""
    return np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
        ]
    )


def test_n_cat_subclass_rejected_without_callbacks_or_data_work() -> None:
    """Reject hostile category counts before callbacks or response access."""
    _HostileInt.reset()
    with pytest.raises(TypeError, match="n_cat must be an integer >= 2"):
        fit_rsm(_ResponseSentinel(), n_cat=_HostileInt(3), q_theta=41, max_iter=500, tol=1e-6)
    assert _HostileInt.callbacks == 0


def test_q_theta_untrusted_values_rejected_without_hash_or_numeric_callbacks() -> None:
    """Reject hostile quadrature controls before hashing or numeric callbacks."""
    _HostileInt.reset()
    _HashTrap.callbacks = 0
    with pytest.raises(ValueError, match="q_theta must be an integer"):
        fit_rsm(_ResponseSentinel(), n_cat=3, q_theta=_HostileInt(7), max_iter=500, tol=1e-6)
    assert _HostileInt.callbacks == 0
    with pytest.raises(ValueError, match="q_theta must be an integer"):
        fit_rsm(_ResponseSentinel(), n_cat=3, q_theta=_HashTrap(), max_iter=500, tol=1e-6)
    assert _HashTrap.callbacks == 0


def test_max_iter_subclass_rejected_without_callbacks_or_data_work() -> None:
    """Reject hostile iteration limits before callbacks or response access."""
    _HostileInt.reset()
    with pytest.raises(ValueError, match="max_iter must be an integer"):
        fit_rsm(_ResponseSentinel(), n_cat=3, max_iter=_HostileInt(10), q_theta=41, tol=1e-6)
    assert _HostileInt.callbacks == 0


def test_tol_subclass_rejected_without_callbacks_or_data_work() -> None:
    """Reject hostile tolerances before callbacks or response access."""
    _HostileFloat.reset()
    with pytest.raises(ValueError, match="tol must be finite and > 0"):
        fit_rsm(_ResponseSentinel(), n_cat=3, tol=_HostileFloat(1e-6), q_theta=41, max_iter=500)
    assert _HostileFloat.callbacks == 0


def test_existing_numpy_scalar_compatibility_is_preserved() -> None:
    """Preserve supported NumPy scalar controls after normalization."""
    core = _FakeCore()
    with patch("fast_mlsirm.fitstats._core_module", return_value=core):
        fit_rsm(
            _responses(),
            n_cat=3,
            q_theta=np.int64(7),
            max_iter=10,
            tol=np.float32(1e-5),
        )
    assert core.controls is not None
    n_cat, q_theta, max_iter, tol = core.controls
    assert type(n_cat) is int
    assert type(q_theta) is int
    assert type(max_iter) is int
    assert type(tol) is float
    assert (n_cat, q_theta, max_iter) == (3, 7, 10)
    assert tol == pytest.approx(float(np.float32(1e-5)))


def test_numpy_integer_contract_remains_narrow_for_n_cat_and_max_iter() -> None:
    """Keep the deliberate NumPy integer admission contract narrow."""
    with pytest.raises(TypeError, match="n_cat must be an integer >= 2"):
        fit_rsm(_ResponseSentinel(), n_cat=np.int64(3), q_theta=41, max_iter=500, tol=1e-6)
    with pytest.raises(ValueError, match="max_iter must be an integer"):
        fit_rsm(_ResponseSentinel(), n_cat=3, max_iter=np.int64(10), q_theta=41, tol=1e-6)
