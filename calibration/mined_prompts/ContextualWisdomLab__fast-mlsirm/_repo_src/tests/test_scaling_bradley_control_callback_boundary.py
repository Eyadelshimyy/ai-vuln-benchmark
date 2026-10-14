"""Trust-boundary regressions for Bradley-Terry public controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fast_mlsirm
import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.config import MAX_MAX_ITER
from fast_mlsirm.scaling import bradley_terry_mm


class _HostileFloat(float):
    """Float subclass whose conversion/comparison callbacks must stay inert."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __float__(self):
        type(self).calls += 1
        return float.__float__(self)

    def __lt__(self, other):
        type(self).calls += 1
        return float.__lt__(self, other)

    def __le__(self, other):
        type(self).calls += 1
        return float.__le__(self, other)


class _FloatProvider:
    """Arbitrary float protocol provider that must never be coerced."""

    calls = 0

    def __init__(self, value: float):
        self.value = value

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __float__(self):
        type(self).calls += 1
        return self.value


class _HostileInt(int):
    """Integer subclass whose conversion/comparison callbacks must stay inert."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __int__(self):
        type(self).calls += 1
        return int.__int__(self)

    def __index__(self):
        type(self).calls += 1
        return int.__index__(self)

    def __lt__(self, other):
        type(self).calls += 1
        return int.__lt__(self, other)

    def __le__(self, other):
        type(self).calls += 1
        return int.__le__(self, other)


class _IndexProvider:
    """Arbitrary integer protocol provider that must never be coerced."""

    calls = 0

    def __init__(self, value: int):
        self.value = value

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __int__(self):
        type(self).calls += 1
        return self.value

    def __index__(self):
        type(self).calls += 1
        return self.value


class _WinsSentinel:
    """Fail if rejected controls allow win-matrix materialization."""

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Raise when NumPy is asked to materialize this sentinel."""
        raise AssertionError("wins must not be materialized for invalid controls")


def _unexpected_core_discovery():
    """Fail if rejected controls reach compiled-core discovery."""
    raise AssertionError("compiled core must not be discovered for invalid controls")


@pytest.mark.parametrize("factory", (_HostileFloat, _FloatProvider))
@pytest.mark.parametrize("field", ("alpha", "tol"))
def test_rejects_untrusted_real_controls_before_callbacks_or_data(
    monkeypatch, factory, field
):
    """Real-valued controls reject executable providers before any side effect."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    factory.reset()
    kwargs = {"alpha": 0.0, "tol": 1e-8, field: factory(0.5)}

    with pytest.raises(ValueError, match=f"{field} must be a real number"):
        bradley_terry_mm(_WinsSentinel(), **kwargs)

    assert factory.calls == 0


@pytest.mark.parametrize("factory", (_HostileInt, _IndexProvider))
def test_rejects_untrusted_max_iter_before_callbacks_or_data(monkeypatch, factory):
    """Iteration controls reject subclasses/protocol providers before callbacks."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    factory.reset()

    with pytest.raises(ValueError, match="max_iter must be an integer"):
        bradley_terry_mm(_WinsSentinel(), max_iter=factory(10))

    assert factory.calls == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"alpha": True}, "alpha must be a real number"),
        ({"alpha": np.bool_(True)}, "alpha must be a real number"),
        ({"tol": False}, "tol must be a real number"),
        ({"tol": np.bool_(False)}, "tol must be a real number"),
        ({"max_iter": True}, "max_iter must be an integer"),
        ({"max_iter": np.bool_(True)}, "max_iter must be an integer"),
        ({"alpha": -1.0}, "alpha must be finite and nonnegative"),
        ({"alpha": float("inf")}, "alpha must be finite and nonnegative"),
        ({"alpha": 10**10000}, "alpha must be finite"),
        ({"tol": 0.0}, "tol must be finite and positive"),
        ({"tol": float("nan")}, "tol must be finite and positive"),
        ({"tol": 10**10000}, "tol must be finite"),
        ({"max_iter": 0}, f"max_iter must be between 1 and {MAX_MAX_ITER}"),
        (
            {"max_iter": MAX_MAX_ITER + 1},
            f"max_iter must be between 1 and {MAX_MAX_ITER}",
        ),
    ),
)
def test_invalid_exact_controls_fail_before_data_and_core(monkeypatch, kwargs, message):
    """Type-correct but invalid controls fail before data/native boundaries."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=message):
        bradley_terry_mm(_WinsSentinel(), **kwargs)


class _FakeScalingCore:
    """Capture the trusted Bradley-Terry PyO3 payload."""

    def __init__(self) -> None:
        """Initialize an empty dispatch ledger."""
        self.calls: list[tuple[Any, ...]] = []

    def bradley_terry_mm(self, *args: Any) -> dict[str, Any]:
        """Record one dispatch and return a structurally valid Rust-like result."""
        self.calls.append(args)
        return {
            "params": [0.0, 0.0],
            "weights": [1.0, 1.0],
            "iterations": 2,
        }


def _wins() -> np.ndarray:
    """Return a connected two-object Bradley-Terry win matrix."""
    return np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float64)


def test_genuine_numpy_controls_dispatch_as_exact_builtins(monkeypatch):
    """Supported NumPy scalars normalize once before the PyO3 call."""
    core = _FakeScalingCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)

    result = bradley_terry_mm(
        _wins(),
        alpha=np.float64(0.25),
        max_iter=np.int64(50),
        tol=np.float32(1e-6),
    )

    assert result.iterations == 2
    assert len(core.calls) == 1
    _flat_wins, n, alpha, max_iter, tol = core.calls[0]
    assert n == 2
    assert type(alpha) is float
    assert type(max_iter) is int
    assert type(tol) is float
    assert alpha == pytest.approx(0.25)
    assert max_iter == 50
    assert tol == pytest.approx(float(np.float32(1e-6)))


def test_top_level_export_uses_the_hardened_scaling_wrapper():
    """The public package export must not retain the pre-install legacy callable."""
    assert fast_mlsirm.bradley_terry_mm is bradley_terry_mm
    assert getattr(bradley_terry_mm, "__fast_mlsirm_control_hardened__", False)
