"""Trust-boundary regressions for Sympson-Hetter scalar controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm import exposure


class _UnexpectedCore:
    """Fail if rejected controls reach native Sympson-Hetter dispatch."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"native core reached before Sympson-Hetter admission: {name}")


class _UnexpectedArray:
    """Fail if rejected controls allow caller array materialization."""

    callbacks = 0

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        type(self).callbacks += 1
        raise AssertionError("caller array materialized before Sympson-Hetter controls")


class _HostileFloat(float):
    """Float subclass whose conversion callback must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller __float__ executed during Sympson-Hetter admission")


@pytest.mark.parametrize("field", ["r_max", "tol"])
def test_sympson_hetter_rejects_hostile_real_controls_before_data_or_native(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Caller scalar subclasses fail before callbacks, data work, or native work."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileFloat.callbacks = 0
    _UnexpectedArray.callbacks = 0
    controls: dict[str, object] = {"r_max": 0.25, "tol": 0.02}
    controls[field] = _HostileFloat(controls[field])

    with pytest.raises(ValueError, match=rf"{field} must be a real scalar"):
        exposure.sympson_hetter(
            _UnexpectedArray(),
            _UnexpectedArray(),
            r_max=controls["r_max"],
            tol=controls["tol"],
        )

    assert _HostileFloat.callbacks == 0
    assert _UnexpectedArray.callbacks == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("r_max", 0.0, r"r_max must be finite and in \(0, 1\]"),
        ("r_max", -0.1, r"r_max must be finite and in \(0, 1\]"),
        ("r_max", 1.01, r"r_max must be finite and in \(0, 1\]"),
        ("r_max", float("nan"), r"r_max must be finite and in \(0, 1\]"),
        ("r_max", float("inf"), r"r_max must be finite and in \(0, 1\]"),
        ("r_max", float("-inf"), r"r_max must be finite and in \(0, 1\]"),
        ("tol", -0.01, r"tol must be finite and non-negative"),
        ("tol", float("nan"), r"tol must be finite and non-negative"),
        ("tol", float("inf"), r"tol must be finite and non-negative"),
        ("tol", float("-inf"), r"tol must be finite and non-negative"),
    ],
)
def test_sympson_hetter_rejects_invalid_real_domains_before_data_or_native(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: float,
    message: str,
) -> None:
    """Semantic scalar domains are established before caller-owned data work."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _UnexpectedArray.callbacks = 0
    controls: dict[str, object] = {"r_max": 0.25, "tol": 0.02}
    controls[field] = value

    with pytest.raises(ValueError, match=message):
        exposure.sympson_hetter(
            _UnexpectedArray(),
            _UnexpectedArray(),
            r_max=controls["r_max"],
            tol=controls["tol"],
        )

    assert _UnexpectedArray.callbacks == 0


def test_sympson_hetter_accepts_zero_tolerance_at_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero tolerance preserves the Rust finite non-negative contract exactly."""

    captured: dict[str, tuple[object, ...]] = {}

    class _Core:
        def py_sympson_hetter(self, *args: object) -> dict[str, object]:
            captured["args"] = args
            return {
                "k": np.array([1.0]),
                "exposure": np.array([0.25]),
                "selection": np.array([0.5]),
                "max_exposure": 0.25,
                "n_iter": 1,
                "converged": False,
                "history_max_exposure": np.array([0.25]),
            }

    monkeypatch.setattr(fast_mlsirm, "_core", _Core(), raising=False)

    result = exposure.sympson_hetter(
        np.array([1.0]),
        np.array([0.0]),
        r_max=1.0,
        test_length=1,
        n_simulees=1,
        max_iter=1,
        tol=0.0,
        q_theta=3,
    )

    args = captured["args"]
    assert type(args[7]) is float
    assert args[7] == 0.0
    assert result.converged is False


def test_sympson_hetter_normalizes_supported_numpy_real_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concrete NumPy real scalars reach Rust as package-owned built-in floats."""

    captured: dict[str, tuple[object, ...]] = {}

    class _Core:
        def py_sympson_hetter(self, *args: object) -> dict[str, object]:
            captured["args"] = args
            return {
                "k": np.array([1.0]),
                "exposure": np.array([0.25]),
                "selection": np.array([0.5]),
                "max_exposure": 0.25,
                "n_iter": 1,
                "converged": True,
                "history_max_exposure": np.array([0.25]),
            }

    monkeypatch.setattr(fast_mlsirm, "_core", _Core(), raising=False)

    result = exposure.sympson_hetter(
        np.array([1], dtype=np.int16),
        np.array([0.0], dtype=np.float32),
        r_max=np.float32(0.25),
        test_length=1,
        n_simulees=10,
        max_iter=1,
        tol=np.float64(0.02),
        seed=np.uint64(7),
        q_theta=3,
    )

    args = captured["args"]
    assert type(args[3]) is float
    assert args[3] == pytest.approx(0.25)
    assert type(args[7]) is float
    assert args[7] == pytest.approx(0.02)
    assert result.converged is True
