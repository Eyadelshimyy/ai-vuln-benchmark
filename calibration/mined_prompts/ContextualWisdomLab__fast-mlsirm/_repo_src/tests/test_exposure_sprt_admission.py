"""Trust-boundary regressions for Wald SPRT classification admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm import exposure


class _UnexpectedCore:
    """Fail if rejected evidence reaches native SPRT dispatch."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"native core reached before SPRT admission: {name}")


class _UnexpectedArray:
    """Fail if invalid controls allow caller array materialization."""

    callbacks = 0

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        type(self).callbacks += 1
        raise AssertionError("caller array materialized before SPRT controls")


class _HostileFloat(float):
    """Float subclass whose conversion callback must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller __float__ executed during SPRT admission")


class _HostileCell:
    """Object-array cell whose numeric conversion must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("object-array cell coerced during SPRT admission")


def _valid_inputs() -> dict[str, object]:
    return {
        "a": np.array([1.2, 0.9], dtype=np.float32),
        "b": np.array([-0.4, 0.5], dtype=np.float32),
        "c": np.array([0.1, 0.2], dtype=np.float32),
        "responses": np.array([1, 0], dtype=np.int16),
        "theta_cut": 0.0,
        "delta": 0.5,
        "alpha": 0.05,
        "beta": 0.05,
    }


@pytest.mark.parametrize("field", ["theta_cut", "delta", "alpha", "beta"])
def test_sprt_rejects_hostile_scalar_subclasses_before_callbacks_or_native(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Caller scalar subclasses fail before conversion hooks or dispatch."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    args = _valid_inputs()
    _HostileFloat.callbacks = 0
    args[field] = _HostileFloat(args[field])

    with pytest.raises(ValueError, match=rf"{field} must be a real scalar"):
        exposure.sprt_classify(**args)

    assert _HostileFloat.callbacks == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("theta_cut", float("nan"), "theta_cut must be finite"),
        ("theta_cut", float("inf"), "theta_cut must be finite"),
        ("delta", 0.0, "delta must be finite and > 0"),
        ("delta", -0.1, "delta must be finite and > 0"),
        ("delta", float("nan"), "delta must be finite and > 0"),
        ("alpha", 0.0, r"alpha must be finite and in \(0, 1\)"),
        ("alpha", 1.0, r"alpha must be finite and in \(0, 1\)"),
        ("alpha", float("nan"), r"alpha must be finite and in \(0, 1\)"),
        ("beta", 0.0, r"beta must be finite and in \(0, 1\)"),
        ("beta", 1.0, r"beta must be finite and in \(0, 1\)"),
        ("beta", float("inf"), r"beta must be finite and in \(0, 1\)"),
    ],
)
def test_sprt_rejects_invalid_scalar_domains_before_caller_data(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: float,
    message: str,
) -> None:
    """Rust scalar domains are established before caller-owned data work."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _UnexpectedArray.callbacks = 0
    args = {
        "a": _UnexpectedArray(),
        "b": _UnexpectedArray(),
        "c": _UnexpectedArray(),
        "responses": _UnexpectedArray(),
        "theta_cut": 0.0,
        "delta": 0.5,
        "alpha": 0.05,
        "beta": 0.05,
    }
    args[field] = value

    with pytest.raises(ValueError, match=message):
        exposure.sprt_classify(**args)

    assert _UnexpectedArray.callbacks == 0


def test_sprt_rejects_invalid_combined_error_budget_before_caller_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The alpha+beta contract fails before any item or response evidence."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _UnexpectedArray.callbacks = 0

    with pytest.raises(ValueError, match=r"alpha \+ beta must be < 1"):
        exposure.sprt_classify(
            _UnexpectedArray(),
            _UnexpectedArray(),
            _UnexpectedArray(),
            responses=_UnexpectedArray(),
            theta_cut=0.0,
            delta=0.5,
            alpha=0.6,
            beta=0.4,
        )

    assert _UnexpectedArray.callbacks == 0


@pytest.mark.parametrize("field", ["a", "b", "c", "responses"])
@pytest.mark.parametrize(
    "bad",
    [
        np.array([1.0 + 1.0j, 0.0]),
        np.array([_HostileCell(), _HostileCell()], dtype=object),
        np.array(["1", "0"]),
    ],
)
def test_sprt_rejects_lossy_or_coercive_evidence_before_element_conversion(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad: np.ndarray,
) -> None:
    """Scientific evidence is never silently projected or text/object-coerced."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileCell.callbacks = 0
    args = _valid_inputs()
    args[field] = bad

    expected = (
        "responses must be a real numeric array"
        if field == "responses"
        else rf"{field} must be a real numeric array"
    )
    with pytest.raises(ValueError, match=expected):
        exposure.sprt_classify(**args)

    assert _HostileCell.callbacks == 0


def test_sprt_normalizes_supported_numpy_evidence_and_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supported NumPy inputs reach Rust as contiguous package-owned payloads."""

    captured: dict[str, tuple[object, ...]] = {}

    class _Core:
        def py_sprt_classify(self, *args: object) -> dict[str, object]:
            captured["args"] = args
            return {
                "decision": "continue",
                "n_used": 2,
                "llr": 0.1,
                "llr_trace": np.array([0.05, 0.1]),
            }

    monkeypatch.setattr(fast_mlsirm, "_core", _Core(), raising=False)

    result = exposure.sprt_classify(
        np.array([1.2, 0.9], dtype=np.float32),
        np.array([-0.4, 0.5], dtype=np.int16),
        np.array([0.1, 0.2], dtype=np.float32),
        responses=np.array([True, False], dtype=np.bool_),
        theta_cut=np.float32(0.0),
        delta=np.float64(0.5),
        alpha=np.float32(0.05),
        beta=np.float64(0.05),
    )

    args = captured["args"]
    for array in args[:3]:
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
    assert isinstance(args[3], np.ndarray)
    assert args[3].dtype == np.uint8
    assert args[3].flags.c_contiguous
    for scalar in args[4:]:
        assert type(scalar) is float
    assert result["decision"] == "continue"
