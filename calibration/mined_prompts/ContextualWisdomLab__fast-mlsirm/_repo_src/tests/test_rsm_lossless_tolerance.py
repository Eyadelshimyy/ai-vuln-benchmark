"""Lossless Python-to-Rust tolerance regressions for the rating scale model."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.rsm import fit_rsm


class _ResponseSentinel:
    """Fail if a rejected tolerance reaches caller response materialization."""

    calls = 0

    def __array__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        type(self).calls += 1
        raise AssertionError("response materialization must not run for rejected tol")


class _FakeCore:
    """Minimal Rust-core sentinel recording the normalized tolerance."""

    def __init__(self) -> None:
        self.tol: float | None = None

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
        self.tol = tol
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
    return np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _lossy_longdouble() -> np.longdouble:
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform long double is not wider than float64")
    value = np.nextafter(np.longdouble(1.0), np.longdouble(2.0))
    assert np.longdouble(float(value)) != value
    return value


def test_rsm_rejects_lossy_longdouble_tolerance_before_response_work() -> None:
    """Extended-precision tolerance identity cannot change at the Rust f64 boundary."""

    _ResponseSentinel.calls = 0
    with pytest.raises(ValueError, match="tol must be finite and > 0"):
        fit_rsm(_ResponseSentinel(), n_cat=3, tol=_lossy_longdouble(), q_theta=41, max_iter=500)
    assert _ResponseSentinel.calls == 0


def test_rsm_rejects_lossy_integer_tolerance_before_response_work() -> None:
    """Integer-valued tolerance must also preserve exact binary64 identity."""

    _ResponseSentinel.calls = 0
    with pytest.raises(ValueError, match="tol must be finite and > 0"):
        fit_rsm(_ResponseSentinel(), n_cat=3, tol=2**53 + 1, q_theta=41, max_iter=500)
    assert _ResponseSentinel.calls == 0


def test_rsm_exact_longdouble_tolerance_reaches_rust_as_builtin_float() -> None:
    """Exactly representable extended-precision controls remain supported."""

    core = _FakeCore()
    with patch("fast_mlsirm.fitstats._core_module", return_value=core):
        fit_rsm(_responses(), n_cat=3, tol=np.longdouble(0.5), q_theta=41, max_iter=500)

    assert core.tol == 0.5
    assert type(core.tol) is float
