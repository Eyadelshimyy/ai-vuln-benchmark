"""Trust-boundary regressions for compensatory 2PL semantic controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fast_mlsirm.config import MAX_MAX_ITER, MAX_XI_POINTS
from fast_mlsirm.twopl import fit_2pl

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

    def __index__(self) -> int:
        self.calls.append("index")
        raise AssertionError("index conversion callback executed")

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        self.calls.append("array")
        raise AssertionError("array conversion callback executed")


class _HostileFloat(float):
    """Floating subclass whose conversion hook must never execute."""

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


class _HostileBoolProtocol:
    """Arbitrary truth-value provider that must not cross admission."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __bool__(self) -> bool:
        self.calls.append("bool")
        raise AssertionError("boolean conversion callback executed")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("q", _HostileInt(21), "q must be a finite integer"),
        ("max_iter", _HostileInt(10), "max_iter must be a finite integer"),
        ("xi_points", _HostileInt(100), "xi_points must be a finite integer"),
        ("xi_seed", _HostileInt(7), "xi_seed must be a non-negative integer"),
        ("tol", _HostileFloat(1e-6), "tol must be finite and > 0"),
        ("estimate_corr", _HostileBoolProtocol(), "estimate_corr must be a boolean"),
    ],
)
def test_fit_2pl_rejects_controls_before_callbacks_or_data(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: Any,
    message: str,
) -> None:
    """Rejected controls execute no caller callback, data work, or core discovery."""

    responses = _ExplosiveResponses()
    core_calls: list[str] = []

    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: core_calls.append("core") or None,
    )

    with pytest.raises(ValueError, match=message):
        fit_2pl(responses, **{**_REQUIRED_DEFAULTS, name: value})

    assert value.calls == []
    assert responses.calls == []
    assert core_calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        # #1929: no node-count cap; q=13 is now accepted, only < 1 is not.
        ({"q": 0}, "q must be >= 1"),
        ({"max_iter": 0}, "max_iter must be in"),
        ({"max_iter": MAX_MAX_ITER + 1}, "max_iter must be in"),
        ({"tol": 0.0}, "tol must be finite and > 0"),
        ({"tol": float("nan")}, "tol must be finite and > 0"),
        ({"tol": float("inf")}, "tol must be finite and > 0"),
        ({"estimate_corr": 1}, "estimate_corr must be a boolean"),
        ({"xi_seed": -1}, "xi_seed must be in"),
        ({"xi_seed": 2**64}, "xi_seed must be in"),
        ({"node_rule": "qmc", "xi_points": 0}, "xi_points must be in"),
        (
            {"node_rule": "mc", "xi_points": MAX_XI_POINTS + 1},
            "xi_points must be in",
        ),
    ],
)
def test_fit_2pl_rejects_control_boundaries_before_data(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Domain failures remain pre-data and pre-native, not only type failures."""

    responses = _ExplosiveResponses()
    core_calls: list[str] = []

    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: core_calls.append("core") or None,
    )

    with pytest.raises(ValueError, match=message):
        fit_2pl(responses, **{**_REQUIRED_DEFAULTS, **kwargs})

    assert responses.calls == []
    assert core_calls == []


@pytest.mark.parametrize(
    "tol",
    [
        pytest.param(2**53 + 1, id="python-int-beyond-exact-binary64"),
    ],
)
def test_fit_2pl_rejects_lossy_tolerance_before_response_work(
    monkeypatch: pytest.MonkeyPatch,
    tol: object,
) -> None:
    """A convergence tolerance may not change identity at the Rust f64 boundary."""

    responses = _ExplosiveResponses()
    core_calls: list[str] = []
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: core_calls.append("core") or None,
    )

    with pytest.raises(ValueError, match="tol must be finite and > 0"):
        fit_2pl(responses, tol=tol, model=1, max_iter=500, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)  # type: ignore[arg-type]

    assert responses.calls == []
    assert core_calls == []


def test_fit_2pl_rejects_lossy_longdouble_tolerance_before_response_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extended-precision tolerance evidence cannot be silently rounded to f64."""

    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("np.longdouble has no additional precision on this platform")

    responses = _ExplosiveResponses()
    core_calls: list[str] = []
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: core_calls.append("core") or None,
    )
    one = np.longdouble(1)
    tol = np.nextafter(one, np.longdouble(2), dtype=np.longdouble)

    with pytest.raises(ValueError, match="tol must be finite and > 0"):
        fit_2pl(responses, tol=tol, model=1, max_iter=500, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)

    assert responses.calls == []
    assert core_calls == []


def test_fit_2pl_normalizes_supported_numpy_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concrete NumPy scalar controls remain compatible and reach Rust as built-ins."""

    captured: dict[str, object] = {}

    class _Core:
        def fit_2pl(
            self,
            _yy: np.ndarray,
            _observed: np.ndarray,
            _pattern: np.ndarray,
            n_persons: int,
            n_items: int,
            n_dims: int,
            q: int,
            estimate_corr: bool,
            max_iter: int,
            tol: float,
            node_rule: str,
            xi_points: int,
            xi_seed: int,
        ) -> dict[str, object]:
            captured.update(
                q=q,
                estimate_corr=estimate_corr,
                max_iter=max_iter,
                tol=tol,
                node_rule=node_rule,
                xi_points=xi_points,
                xi_seed=xi_seed,
            )
            return {
                "loading": np.ones(n_items * n_dims),
                "intercept": np.zeros(n_items),
                "theta": np.zeros(n_persons * n_dims),
                "corr": np.eye(n_dims),
                "loglik_trace": np.array([-2.0, -1.0]),
                "n_iter": 2,
                "converged": True,
                "n_parameters": n_items * (n_dims + 1),
                "termination_reason": "converged",
                "final_loglik_change": 1.0,
            }

    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    fit_2pl(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        q=np.int64(21),
        estimate_corr=np.bool_(False),
        max_iter=np.int64(2),
        tol=np.longdouble(0.5),
        xi_points=np.int64(100),
        xi_seed=np.uint64(7), model=1
    )

    assert type(captured["q"]) is int
    assert type(captured["estimate_corr"]) is bool
    assert type(captured["max_iter"]) is int
    assert type(captured["tol"]) is float
    assert captured["tol"] == 0.5
    assert type(captured["node_rule"]) is str
    assert type(captured["xi_points"]) is int
    assert type(captured["xi_seed"]) is int