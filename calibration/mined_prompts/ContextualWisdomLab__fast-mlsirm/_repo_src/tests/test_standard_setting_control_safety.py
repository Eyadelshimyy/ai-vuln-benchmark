"""Fail-closed trust-boundary tests for Hofstee scalar controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import pytest

from fast_mlsirm import fitstats
from fast_mlsirm.standard_setting import hofstee


_SCORES = np.array([45.0, 55.0, 65.0, 75.0], dtype=np.float64)


@dataclass
class _CallbackCounter:
    """Count every caller-controlled scalar callback attempted by validation."""

    calls: int = 0

    def hit(self) -> NoReturn:
        """Record one callback and fail immediately."""
        self.calls += 1
        raise AssertionError("caller callback executed")


def _hostile_float(counter: _CallbackCounter) -> float:
    """Return a float subclass whose common conversion/comparison hooks fail."""

    class HostileFloat(float):
        def __float__(self):
            counter.hit()

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

    return HostileFloat(40.0)


def _hostile_int(counter: _CallbackCounter) -> int:
    """Return an int subclass whose common conversion/comparison hooks fail."""

    class HostileInt(int):
        def __float__(self):
            counter.hit()

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

    return HostileInt(40)


def _hostile_numpy_float(counter: _CallbackCounter) -> np.float64:
    """Return a NumPy floating subclass that must not be normalized."""

    class HostileNumpyFloat(np.float64):
        def __float__(self):
            counter.hit()

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

    return HostileNumpyFloat(40.0)


class _FloatProvider:
    """Arbitrary float protocol provider that is never an accepted control type."""

    def __init__(self, counter: _CallbackCounter) -> None:
        self._counter = counter

    def __float__(self) -> float:
        self._counter.hit()

    def __repr__(self) -> str:
        self._counter.hit()


class _FakeCore:
    """Minimal Rust-boundary stand-in that records normalized controls."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, float, float, float]] = []

    def py_hofstee(
        self,
        scores: np.ndarray,
        min_cut: float,
        max_cut: float,
        min_fail: float,
        max_fail: float,
    ) -> dict[str, object]:
        assert scores.dtype == np.float64
        assert all(
            type(value) is float
            for value in (min_cut, max_cut, min_fail, max_fail)
        )
        self.calls.append((min_cut, max_cut, min_fail, max_fail))
        return {
            "cut_score": 55.0,
            "fail_rate": 20.0,
            "failed": False,
            "cum_freq_percent": np.array([0.0, 100.0]),
        }


@pytest.mark.parametrize("factory", [_hostile_float, _hostile_int, _hostile_numpy_float])
@pytest.mark.parametrize("field", ["min_cut", "max_cut", "min_fail", "max_fail"])
def test_hofstee_rejects_scalar_subclasses_before_callbacks_or_core(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    field: str,
) -> None:
    """Caller scalar subclasses cannot execute code or trigger core discovery."""
    counter = _CallbackCounter()
    discovery_calls = 0

    def discover_core():
        nonlocal discovery_calls
        discovery_calls += 1
        raise AssertionError("Rust core discovered before control validation")

    monkeypatch.setattr(fitstats, "_core_module", discover_core)
    values = {
        "min_cut": 40.0,
        "max_cut": 70.0,
        "min_fail": 10.0,
        "max_fail": 30.0,
    }
    values[field] = factory(counter)

    with pytest.raises(ValueError, match=rf"{field} must be a real number"):
        hofstee(_SCORES, **values)

    assert counter.calls == 0
    assert discovery_calls == 0


@pytest.mark.parametrize("kind", ["bool", "numpy_bool", "provider"])
def test_hofstee_rejects_non_real_controls_before_core(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """Boolean and protocol-only values fail at the trusted type boundary."""
    counter = _CallbackCounter()
    bad_value: object
    if kind == "bool":
        bad_value = True
    elif kind == "numpy_bool":
        bad_value = np.bool_(True)
    else:
        bad_value = _FloatProvider(counter)
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected core discovery")),
    )
    with pytest.raises(ValueError, match="min_cut must be a real number"):
        hofstee(_SCORES, bad_value, 70.0, 10.0, 30.0)
    assert counter.calls == 0


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("min_cut", -0.1, "min_cut must be finite and in [0, 100]"),
        ("max_cut", 100.1, "max_cut must be finite and in [0, 100]"),
        ("min_fail", float("nan"), "min_fail must be finite and in [0, 100]"),
        ("max_fail", float("inf"), "max_fail must be finite and in [0, 100]"),
        ("max_fail", 10**1000, "max_fail must be finite and in [0, 100]"),
    ],
)
def test_hofstee_rejects_scalar_domains_before_core(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    """Range, finiteness, and overflow failures occur before native discovery."""
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected core discovery")),
    )
    values = {
        "min_cut": 40.0,
        "max_cut": 70.0,
        "min_fail": 10.0,
        "max_fail": 30.0,
    }
    values[field] = bad_value
    with pytest.raises(
        ValueError,
        match=message.replace("[", r"\[").replace("]", r"\]"),
    ):
        hofstee(_SCORES, **values)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {"min_cut": 70.0, "max_cut": 40.0, "min_fail": 10.0, "max_fail": 30.0},
            "min_cut must not exceed max_cut",
        ),
        (
            {"min_cut": 40.0, "max_cut": 70.0, "min_fail": 30.0, "max_fail": 10.0},
            "min_fail must not exceed max_fail",
        ),
    ],
)
def test_hofstee_rejects_inverted_bounds_before_core(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, float],
    message: str,
) -> None:
    """Cross-field ordering is established on trusted built-in floats only."""
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected core discovery")),
    )
    with pytest.raises(ValueError, match=message):
        hofstee(_SCORES, **values)


def test_hofstee_accepts_genuine_numpy_scalars_and_marshals_builtin_floats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supported NumPy scalar identities retain compatibility at the Rust boundary."""
    core = _FakeCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)

    result = hofstee(
        _SCORES,
        np.int32(40),
        np.float32(70.0),
        np.uint8(10),
        np.float64(30.0),
    )

    assert core.calls == [(40.0, 70.0, 10.0, 30.0)]
    assert result.cut_score == 55.0
    assert result.fail_rate == 20.0
    assert result.failed is False
