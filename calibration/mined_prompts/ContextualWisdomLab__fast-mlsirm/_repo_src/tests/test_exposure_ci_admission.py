"""Trust-boundary regressions for confidence-interval CAT classification."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm import exposure


class _UnexpectedCore:
    """Fail if rejected CI evidence reaches native dispatch."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"native core reached before CI admission: {name}")


class _UnexpectedArray:
    """Fail if invalid controls allow caller array materialization."""

    callbacks = 0

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        type(self).callbacks += 1
        raise AssertionError("caller array materialized before CI controls")


class _HostileFloat(float):
    """Float subclass whose conversion callback must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller __float__ executed during CI admission")


class _HostileCell:
    """Object-array cell whose numeric conversion must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("object-array cell coerced during CI admission")


def _valid_inputs() -> dict[str, object]:
    return {
        "a": np.array([1.2, 0.9], dtype=np.float32),
        "b": np.array([-0.4, 0.5], dtype=np.float32),
        "c": np.array([0.1, 0.2], dtype=np.float32),
        "responses": np.array([1, 0], dtype=np.int16),
        "theta_cut": 0.0,
        "z_crit": 1.6448536269514722,
    }


@pytest.mark.parametrize("field", ["theta_cut", "z_crit"])
def test_ci_rejects_hostile_scalar_subclasses_before_callbacks_or_native(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Caller scalar subclasses fail before conversion hooks or dispatch."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    args = _valid_inputs()
    _HostileFloat.callbacks = 0
    args[field] = _HostileFloat(args[field])

    with pytest.raises(ValueError, match=rf"{field} must be a real scalar"):
        exposure.ci_classify(**args)

    assert _HostileFloat.callbacks == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("theta_cut", float("nan"), "theta_cut must be finite"),
        ("theta_cut", float("inf"), "theta_cut must be finite"),
        ("theta_cut", float("-inf"), "theta_cut must be finite"),
        ("z_crit", 0.0, "z_crit must be finite and > 0"),
        ("z_crit", -0.1, "z_crit must be finite and > 0"),
        ("z_crit", float("nan"), "z_crit must be finite and > 0"),
        ("z_crit", float("inf"), "z_crit must be finite and > 0"),
    ],
)
def test_ci_rejects_invalid_scalar_domains_before_caller_data(
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
        "z_crit": 1.6448536269514722,
    }
    args[field] = value

    with pytest.raises(ValueError, match=message):
        exposure.ci_classify(**args)

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
def test_ci_rejects_lossy_or_coercive_evidence_before_element_conversion(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad: np.ndarray,
) -> None:
    """Observed evidence is never silently projected or text/object-coerced."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileCell.callbacks = 0
    args = _valid_inputs()
    args[field] = bad

    with pytest.raises(ValueError, match=rf"{field} must be a real numeric array"):
        exposure.ci_classify(**args)

    assert _HostileCell.callbacks == 0


def test_ci_normalizes_supported_numpy_evidence_and_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supported NumPy inputs reach Rust as contiguous package-owned payloads."""

    captured: dict[str, tuple[object, ...]] = {}

    class _Core:
        def py_ci_classify(self, *args: object) -> dict[str, object]:
            captured["args"] = args
            return {
                "decision": "continue",
                "n_used": 2,
                "theta_trace": np.array([0.0, 0.1]),
                "se_trace": np.array([0.8, 0.7]),
                "lower_trace": np.array([-1.3, -1.05]),
                "upper_trace": np.array([1.3, 1.25]),
            }

    monkeypatch.setattr(fast_mlsirm, "_core", _Core(), raising=False)

    result = exposure.ci_classify(
        np.array([1.2, 0.9], dtype=np.float32),
        np.array([-1, 1], dtype=np.int16),
        np.array([0.1, 0.2], dtype=np.float32),
        responses=np.array([True, False], dtype=np.bool_),
        theta_cut=np.float32(0.0),
        z_crit=np.float64(1.6448536269514722),
    )

    args = captured["args"]
    for array in args[:3]:
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
    assert isinstance(args[3], np.ndarray)
    assert args[3].dtype == np.uint8
    assert args[3].flags.c_contiguous
    assert type(args[4]) is float
    assert type(args[5]) is float
    assert result["decision"] == "continue"
