"""Trust-boundary regressions for GPCM admission."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fast_mlsirm.config import MAX_MAX_ITER, MAX_POLYTOMOUS_CATEGORIES
from fast_mlsirm.gpcm import fit_gpcm

_REQUIRED_DEFAULTS: dict[str, object] = {
    "model": 1,
    "max_iter": 500,
    "tol": 1e-6,
    "xi_points": 4000,
    "xi_seed": 0x9E37_79B9_7F4A_7C15,
}


class _ExplosiveResponses:
    """Response sentinel that records forbidden materialization."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        self.calls.append("array")
        raise AssertionError("responses were materialized before control rejection")


class _HostileInt(int):
    """Integer subclass whose conversion hooks must never execute."""

    def __new__(cls, value: int) -> _HostileInt:
        instance = super().__new__(cls, value)
        instance.calls = []
        return instance

    def __int__(self) -> int:
        self.calls.append("int")
        raise AssertionError("integer conversion callback executed")

    def __float__(self) -> float:
        self.calls.append("float")
        raise AssertionError("float conversion callback executed")

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        self.calls.append("array")
        raise AssertionError("array conversion callback executed")


class _HostileFloat(float):
    """Floating subclass whose conversion hooks must never execute."""

    def __new__(cls, value: float) -> _HostileFloat:
        instance = super().__new__(cls, value)
        instance.calls = []
        return instance

    def __float__(self) -> float:
        self.calls.append("float")
        raise AssertionError("float conversion callback executed")

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        self.calls.append("array")
        raise AssertionError("array conversion callback executed")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("n_cat", _HostileInt(3), "n_cat must be a finite integer"),
        ("q", _HostileInt(21), "q must be a finite integer"),
        ("max_iter", _HostileInt(10), "max_iter must be a finite integer"),
        ("xi_points", _HostileInt(100), "xi_points must be a finite integer"),
        ("xi_seed", _HostileInt(7), "xi_seed must be a non-negative integer"),
        ("tol", _HostileFloat(1e-6), "tol must be finite and > 0"),
    ],
)
def test_fit_gpcm_rejects_controls_before_callbacks_or_data(
    name: str,
    value: Any,
    message: str,
) -> None:
    """Rejected controls execute no caller callback or response work."""

    responses = _ExplosiveResponses()
    call_kwargs: dict[str, object] = {**_REQUIRED_DEFAULTS, "n_cat": 3, name: value}

    with pytest.raises(ValueError, match=message):
        fit_gpcm(responses, **call_kwargs)

    assert value.calls == []
    assert responses.calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_cat": 1}, "n_cat must be between"),
        ({"n_cat": MAX_POLYTOMOUS_CATEGORIES + 1}, "n_cat must be between"),
        # #1929: no node-count cap; q=13 is now accepted, only < 1 is not.
        ({"q": 0}, "q must be >= 1"),
        ({"max_iter": 0}, "max_iter must be between"),
        ({"max_iter": MAX_MAX_ITER + 1}, "max_iter must be between"),
        ({"tol": 0.0}, "tol must be finite and > 0"),
        ({"tol": float("nan")}, "tol must be finite and > 0"),
        ({"tol": float("inf")}, "tol must be finite and > 0"),
        ({"xi_points": 0}, "xi_points must be between"),
        ({"xi_points": 200_001}, "xi_points must be between"),
        ({"xi_seed": -1}, "xi_seed must be in"),
        ({"xi_seed": 2**64}, "xi_seed must be in"),
    ],
)
def test_fit_gpcm_rejects_control_domains_before_data(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Semantic-domain failures remain pre-data and pre-native."""

    responses = _ExplosiveResponses()
    call_kwargs: dict[str, object] = {**_REQUIRED_DEFAULTS, "n_cat": 3, **kwargs}

    with pytest.raises(ValueError, match=message):
        fit_gpcm(responses, **call_kwargs)

    assert responses.calls == []


def test_fit_gpcm_rejects_complex_responses_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complex observed evidence cannot be silently projected onto real categories."""

    import fast_mlsirm.fitstats as fitstats

    core_calls: list[str] = []
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: core_calls.append("core") or None,
    )
    responses = np.array([[0.0 + 1.0j, 1.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        fit_gpcm(responses, n_cat=2, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)

    assert core_calls == []


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf")])
def test_fit_gpcm_rejects_infinite_responses_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
) -> None:
    """Infinity is invalid evidence rather than an implicit missing category."""

    import fast_mlsirm.fitstats as fitstats

    core_calls: list[str] = []
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: core_calls.append("core") or None,
    )
    responses = np.array([[0.0, 0.0], [1.0, 1.0], [bad_value, 0.0]])

    with pytest.raises(ValueError, match="responses must be finite where not missing"):
        fit_gpcm(responses, n_cat=2, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)

    assert core_calls == []


def test_fit_gpcm_normalizes_supported_numpy_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concrete supported NumPy scalars reach Rust only as built-in primitives."""

    import fast_mlsirm.fitstats as fitstats

    captured: list[tuple[object, ...]] = []

    class _Core:
        @staticmethod
        def fit_gpcm(*args: object) -> dict[str, object]:
            captured.append(args)
            return {
                "slope": np.ones(2, dtype=np.float64),
                "step": np.zeros(2, dtype=np.float64),
                "theta": np.zeros(2, dtype=np.float64),
                "n_cat": 2,
                "loglik_trace": np.array([-2.0, -1.0]),
                "n_iter": 2,
                "converged": True,
                "termination_reason": "tolerance_met",
                "final_loglik_change": 1.0,
                "n_parameters": 4,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())
    responses = np.array([[0, 0], [1, 1]], dtype=np.int64)

    result = fit_gpcm(
        responses,
        n_cat=np.int64(2),
        q=np.int32(7),
        max_iter=np.uint16(5),
        tol=np.float32(1e-4),
        node_rule="qmc",
        xi_points=np.int64(16),
        xi_seed=np.uint64(7), model=1
    )

    assert result.n_cat == 2
    assert len(captured) == 1
    args = captured[0]
    assert type(args[6]) is int
    assert type(args[7]) is int
    assert type(args[8]) is int
    assert type(args[9]) is float
    assert type(args[11]) is int
    assert type(args[12]) is int
