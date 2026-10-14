"""Fail-closed trust-boundary tests for public many-facet controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

from fast_mlsirm import fitstats
from fast_mlsirm.facets import fit_facets


_RESPONSES = np.array([[[0.0], [1.0]], [[1.0], [0.0]]], dtype=np.float64)


@dataclass
class _CallbackCounter:
    """Count caller-controlled callbacks attempted during validation."""

    calls: int = 0

    def hit(self) -> None:
        """Record one callback and fail immediately."""
        self.calls += 1
        raise AssertionError("caller callback executed")


def _hostile_int(counter: _CallbackCounter) -> int:
    """Return an int subclass whose conversion/comparison/hash hooks fail."""

    class HostileInt(int):
        def __int__(self):
            counter.hit()

        def __index__(self):
            counter.hit()
            return 0

        def __repr__(self):
            counter.hit()

        def __eq__(self, other):
            counter.hit()

        def __lt__(self, other):
            counter.hit()

        def __le__(self, other):
            counter.hit()

        def __hash__(self):
            counter.hit()
            return 0

    return HostileInt(41)


def _hostile_numpy_int(counter: _CallbackCounter) -> np.int64:
    """Return a NumPy integer subclass that must never be normalized."""

    class HostileNumpyInt(np.int64):
        def __int__(self):
            counter.hit()

        def __index__(self):
            counter.hit()
            return 0

        def __repr__(self):
            counter.hit()

        def __eq__(self, other):
            counter.hit()

        def __lt__(self, other):
            counter.hit()

        def __le__(self, other):
            counter.hit()

        def __hash__(self):
            counter.hit()
            return 0

    return HostileNumpyInt(41)


def _hostile_float(counter: _CallbackCounter) -> float:
    """Return a float subclass whose numeric hooks fail."""

    class HostileFloat(float):
        def __float__(self):
            counter.hit()

        def __repr__(self):
            counter.hit()

        def __le__(self, other):
            counter.hit()

        def __lt__(self, other):
            counter.hit()

    return HostileFloat(1e-6)


def _hostile_numpy_float(counter: _CallbackCounter) -> np.float64:
    """Return a NumPy floating subclass whose conversion hooks fail."""

    class HostileNumpyFloat(np.float64):
        def __float__(self):
            counter.hit()

        def __repr__(self):
            counter.hit()

        def __array_ufunc__(self, *args, **kwargs):
            counter.hit()

    return HostileNumpyFloat(1e-6)


class _IndexProvider:
    """Arbitrary integer-protocol provider that is never trusted."""

    def __init__(self, counter: _CallbackCounter) -> None:
        self._counter = counter

    def __index__(self) -> int:
        self._counter.hit()
        return 0

    def __int__(self) -> int:
        self._counter.hit()

    def __repr__(self) -> str:
        self._counter.hit()


class _FloatProvider:
    """Arbitrary float-protocol provider that is never trusted."""

    def __init__(self, counter: _CallbackCounter) -> None:
        self._counter = counter

    def __float__(self) -> float:
        self._counter.hit()

    def __repr__(self) -> str:
        self._counter.hit()


class _FakeCore:
    """Minimal Rust-boundary stand-in recording normalized controls."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, float]] = []

    def fit_facets(
        self,
        yy: np.ndarray,
        observed: np.ndarray,
        n_persons: int,
        n_items: int,
        n_raters: int,
        n_cat: int,
        q_theta: int,
        max_iter: int,
        tol: float,
    ) -> dict[str, object]:
        assert yy.dtype == np.int64
        assert observed.dtype == np.bool_
        assert all(type(value) is int for value in (n_cat, q_theta, max_iter))
        assert type(tol) is float
        self.calls.append((n_cat, q_theta, max_iter, tol))
        return {
            "item_difficulty": np.zeros(n_items),
            "rater_severity": np.zeros(n_raters),
            "thresholds": np.zeros(n_cat - 1),
            "theta": np.zeros(n_persons),
            "loglik_trace": np.array([0.0]),
            "n_iter": 1,
            "converged": True,
            "connected": True,
            "n_parameters": 1,
        }


@pytest.mark.parametrize("factory", [_hostile_int, _hostile_numpy_int])
@pytest.mark.parametrize("field", ["n_cat", "q_theta", "max_iter"])
def test_fit_facets_rejects_integer_subclasses_before_callbacks_or_core(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[_CallbackCounter], object],
    field: str,
) -> None:
    """Untrusted integer subclasses cannot execute code or trigger discovery."""
    counter = _CallbackCounter()
    discovery_calls = 0

    def discover_core():
        nonlocal discovery_calls
        discovery_calls += 1
        raise AssertionError("Rust core discovered before control validation")

    monkeypatch.setattr(fitstats, "_core_module", discover_core)
    values: dict[str, object] = {
        "n_cat": 2,
        "q_theta": 41,
        "max_iter": 10,
        "tol": 1e-6,
    }
    values[field] = factory(counter)

    with pytest.raises(ValueError, match=rf"{field} must"):
        fit_facets(_RESPONSES, **values)

    assert counter.calls == 0
    assert discovery_calls == 0


@pytest.mark.parametrize("factory", [_hostile_float, _hostile_numpy_float])
def test_fit_facets_rejects_float_subclasses_before_callbacks_or_core(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[_CallbackCounter], object],
) -> None:
    """Untrusted tolerance subclasses fail without ufunc/coercion callbacks."""
    counter = _CallbackCounter()
    discovery_calls = 0

    def discover_core():
        nonlocal discovery_calls
        discovery_calls += 1
        raise AssertionError("Rust core discovered before control validation")

    monkeypatch.setattr(fitstats, "_core_module", discover_core)
    with pytest.raises(ValueError, match="tol must be a real number"):
        fit_facets(_RESPONSES, n_cat=2, q_theta=41, max_iter=10, tol=factory(counter))
    assert counter.calls == 0
    assert discovery_calls == 0


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("n_cat", True, "n_cat must be an integer"),
        ("q_theta", np.bool_(True), "q_theta must be an integer"),
        ("max_iter", False, "max_iter must be an integer"),
        ("q_theta", _IndexProvider, "q_theta must be an integer"),
        ("tol", _FloatProvider, "tol must be a real number"),
    ],
)
def test_fit_facets_rejects_protocol_and_boolean_controls_before_core(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    """Booleans and protocol-only controls fail at exact trusted identities."""
    counter = _CallbackCounter()
    value = bad_value(counter) if isinstance(bad_value, type) else bad_value
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected core discovery")),
    )
    values: dict[str, object] = {
        "n_cat": 2,
        "q_theta": 41,
        "max_iter": 10,
        "tol": 1e-6,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        fit_facets(_RESPONSES, **values)
    assert counter.calls == 0


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("n_cat", 1, "n_cat must be an integer in"),
        # #1929: no node-count cap; q_theta=9 is now accepted, only < 1 is not.
        ("q_theta", 0, "q_theta must be >= 1"),
        ("max_iter", 0, "max_iter must be an integer in"),
        ("tol", 0.0, "tol must be finite and > 0"),
        ("tol", float("nan"), "tol must be finite and > 0"),
        ("tol", float("inf"), "tol must be finite and > 0"),
        ("tol", 10**1000, "tol must be finite and > 0"),
    ],
)
def test_fit_facets_rejects_control_domains_before_core(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    """Control-domain failures occur before native discovery."""
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected core discovery")),
    )
    values: dict[str, object] = {
        "n_cat": 2,
        "q_theta": 41,
        "max_iter": 10,
        "tol": 1e-6,
    }
    values[field] = bad_value
    with pytest.raises(ValueError, match=message):
        fit_facets(_RESPONSES, **values)


def test_fit_facets_validates_responses_before_core_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed response arrays fail before native capability is discovered."""
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected core discovery")),
    )
    with pytest.raises(ValueError, match="responses must be a 3-D"):
        fit_facets(np.array([0.0, 1.0]), n_cat=2, q_theta=41, max_iter=10, tol=1e-6)


def test_fit_facets_accepts_genuine_numpy_scalars_and_marshals_builtins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Genuine NumPy scalars retain compatibility at the Rust boundary."""
    core = _FakeCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)

    result = fit_facets(
        _RESPONSES,
        n_cat=np.int32(2),
        q_theta=np.uint8(41),
        max_iter=np.int64(10),
        tol=np.float32(1e-5),
    )

    assert core.calls == [(2, 41, 10, pytest.approx(float(np.float32(1e-5))))]
    assert result.converged is True
    assert result.connected is True
