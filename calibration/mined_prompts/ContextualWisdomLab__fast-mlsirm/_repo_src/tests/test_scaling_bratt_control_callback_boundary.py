"""Trust-boundary regressions for Bradley-Terry-with-ties controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fast_mlsirm
import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.config import MAX_MAX_ITER
from fast_mlsirm.scaling import bratt_mm


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


class _MatrixSentinel:
    """Fail if rejected controls allow comparison-data materialization."""

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Raise when NumPy is asked to materialize this sentinel."""
        raise AssertionError("comparison data must not be materialized for invalid controls")


def _unexpected_core_discovery():
    """Fail if rejected controls reach compiled-core discovery."""
    raise AssertionError("compiled core must not be discovered for invalid controls")


@pytest.mark.parametrize("factory", (_HostileInt, _IndexProvider))
@pytest.mark.parametrize("field", ("ref_index", "max_iter"))
def test_rejects_untrusted_integer_controls_before_callbacks_or_data(
    monkeypatch, factory, field
):
    """Integer controls reject executable providers before any side effect."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    factory.reset()
    kwargs = {"ref_index": 0, "max_iter": 100, field: factory(10)}

    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        bratt_mm(_MatrixSentinel(), _MatrixSentinel(), **kwargs)

    assert factory.calls == 0


@pytest.mark.parametrize("factory", (_HostileFloat, _FloatProvider))
@pytest.mark.parametrize("field", ("ref_value", "tol"))
def test_rejects_untrusted_real_controls_before_callbacks_or_data(
    monkeypatch, factory, field
):
    """Real controls reject executable providers before any side effect."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    factory.reset()
    kwargs = {"ref_value": 1.0, "tol": 1e-10, field: factory(0.5)}

    with pytest.raises(ValueError, match=f"{field} must be a real number"):
        bratt_mm(_MatrixSentinel(), _MatrixSentinel(), **kwargs)

    assert factory.calls == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"ref_index": True}, "ref_index must be an integer"),
        ({"ref_index": np.bool_(True)}, "ref_index must be an integer"),
        ({"ref_index": -1}, "ref_index must be nonnegative"),
        ({"ref_value": False}, "ref_value must be a real number"),
        ({"ref_value": np.bool_(False)}, "ref_value must be a real number"),
        ({"ref_value": 0.0}, "ref_value must be finite and positive"),
        ({"ref_value": float("inf")}, "ref_value must be finite and positive"),
        ({"ref_value": 10**10000}, "ref_value must be finite"),
        ({"max_iter": True}, "max_iter must be an integer"),
        ({"max_iter": np.bool_(True)}, "max_iter must be an integer"),
        ({"max_iter": 0}, f"max_iter must be between 1 and {MAX_MAX_ITER}"),
        (
            {"max_iter": MAX_MAX_ITER + 1},
            f"max_iter must be between 1 and {MAX_MAX_ITER}",
        ),
        ({"tol": False}, "tol must be a real number"),
        ({"tol": np.bool_(False)}, "tol must be a real number"),
        ({"tol": 0.0}, "tol must be finite and positive"),
        ({"tol": float("nan")}, "tol must be finite and positive"),
        ({"tol": 10**10000}, "tol must be finite"),
    ),
)
def test_invalid_exact_controls_fail_before_data_and_core(monkeypatch, kwargs, message):
    """Type-correct but invalid controls fail before data/native boundaries."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=message):
        bratt_mm(_MatrixSentinel(), _MatrixSentinel(), **kwargs)


class _FakeScalingCore:
    """Capture the trusted BRATT PyO3 payload."""

    def __init__(self) -> None:
        """Initialize an empty dispatch ledger."""
        self.calls: list[tuple[Any, ...]] = []

    def bratt_mm(self, *args: Any) -> dict[str, Any]:
        """Record one dispatch and return a structurally valid Rust-like result."""
        self.calls.append(args)
        return {
            "alpha": [1.0, 1.0],
            "alpha0": 0.5,
            "iterations": 2,
            "log_likelihood": -1.0,
        }


def _comparison_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a connected two-object BRATT fixture with observed ties."""
    wins = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float64)
    ties = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    return wins, ties


def test_genuine_numpy_controls_dispatch_as_exact_builtins(monkeypatch):
    """Supported NumPy scalars normalize once before the PyO3 call."""
    core = _FakeScalingCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    wins, ties = _comparison_data()

    result = bratt_mm(
        wins,
        ties,
        ref_index=np.int64(1),
        ref_value=np.float64(2.0),
        max_iter=np.int64(50),
        tol=np.float32(1e-6),
    )

    assert result.iterations == 2
    assert len(core.calls) == 1
    _flat_wins, _flat_ties, n, ref_index, ref_value, max_iter, tol = core.calls[0]
    assert n == 2
    assert type(ref_index) is int
    assert type(ref_value) is float
    assert type(max_iter) is int
    assert type(tol) is float
    assert ref_index == 1
    assert ref_value == pytest.approx(2.0)
    assert max_iter == 50
    assert tol == pytest.approx(float(np.float32(1e-6)))


def test_top_level_export_uses_the_hardened_scaling_wrapper():
    """The package export must not retain the pre-install legacy callable."""
    assert fast_mlsirm.bratt_mm is bratt_mm
    assert getattr(bratt_mm, "__fast_mlsirm_control_hardened__", False)
