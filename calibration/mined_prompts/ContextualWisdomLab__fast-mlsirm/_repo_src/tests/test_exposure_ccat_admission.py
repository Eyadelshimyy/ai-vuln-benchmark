"""Trust-boundary regressions for constrained-CAT evidence admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm import exposure


class _UnexpectedCore:
    """Fail if rejected CCAT evidence reaches native dispatch."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"native core reached before CCAT admission: {name}")


class _UnexpectedArray:
    """Fail if invalid controls allow caller array materialization."""

    callbacks = 0

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        type(self).callbacks += 1
        raise AssertionError("caller array materialized before CCAT controls")


class _HostileFloat(float):
    """Float subclass whose conversion callback must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller __float__ executed during CCAT admission")


class _HostileCell:
    """Object-array cell whose numeric conversion must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("object-array cell coerced during CCAT admission")


def _valid_inputs() -> dict[str, object]:
    """Return a bounded valid CCAT request using supported NumPy storage."""

    return {
        "a": np.array([1.0, 1.5, 0.8, 2.0, 1.2, 0.9], dtype=np.float32),
        "b": np.array([-0.5, 0.2, 0.0, 0.8, -0.2, 0.4], dtype=np.float32),
        "c": np.array([0.0, 0.1, 0.0, 0.2, 0.0, 0.0], dtype=np.float32),
        "groups": np.array([0, 0, 1, 1, 0, 1], dtype=np.int16),
        "targets": np.array([0.6, 0.4], dtype=np.float32),
        "administered": np.array([True, False, False, True, False, False]),
        "theta0": np.float32(0.1),
    }


def test_ccat_rejects_hostile_theta_before_caller_data_or_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semantic controls fail before caller arrays and native discovery."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _UnexpectedArray.callbacks = 0
    _HostileFloat.callbacks = 0

    with pytest.raises(ValueError, match="theta0 must be a real scalar"):
        exposure.ccat_select(
            _UnexpectedArray(),
            _UnexpectedArray(),
            _UnexpectedArray(),
            groups=_UnexpectedArray(),
            targets=_UnexpectedArray(),
            administered=_UnexpectedArray(),
            theta0=_HostileFloat(0.1),
        )

    assert _HostileFloat.callbacks == 0
    assert _UnexpectedArray.callbacks == 0


@pytest.mark.parametrize("field", ["a", "b", "c", "targets"])
@pytest.mark.parametrize("storage", ["complex", "object", "text"])
def test_ccat_rejects_lossy_item_and_target_evidence_before_native(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    storage: str,
) -> None:
    """Item and content-target evidence is admitted before real marshalling."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    args = _valid_inputs()
    size = 2 if field == "targets" else 6
    _HostileCell.callbacks = 0
    if storage == "complex":
        bad = np.zeros(size, dtype=np.complex128)
        bad[0] = 1.0 + 1.0j
    elif storage == "object":
        bad = np.array([_HostileCell() for _ in range(size)], dtype=object)
    else:
        bad = np.array(["1" for _ in range(size)])
    args[field] = bad

    with pytest.raises(ValueError, match=rf"{field} must be a real numeric array"):
        exposure.ccat_select(**args)

    assert _HostileCell.callbacks == 0


@pytest.mark.parametrize("storage", ["object", "text"])
def test_ccat_rejects_coercive_group_storage_before_native(
    monkeypatch: pytest.MonkeyPatch,
    storage: str,
) -> None:
    """Content-group identity is not inferred through caller numeric coercion."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    args = _valid_inputs()
    _HostileCell.callbacks = 0
    if storage == "object":
        args["groups"] = np.array([_HostileCell() for _ in range(6)], dtype=object)
    else:
        args["groups"] = np.array(["0", "0", "1", "1", "0", "1"])

    with pytest.raises(ValueError, match="groups must be a real numeric array"):
        exposure.ccat_select(**args)

    assert _HostileCell.callbacks == 0


def test_ccat_normalizes_supported_numpy_evidence_before_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supported evidence reaches Rust as contiguous package-owned payloads."""

    captured: dict[str, tuple[object, ...]] = {}

    class _Core:
        def py_ccat_select(self, *args: object) -> dict[str, object]:
            captured["args"] = args
            return {
                "selected": 1,
                "group": 0,
                "discrepancy": np.array([0.1, -0.1]),
                "info": np.arange(6, dtype=np.float64),
            }

    monkeypatch.setattr(fast_mlsirm, "_core", _Core(), raising=False)
    result = exposure.ccat_select(**_valid_inputs())

    args = captured["args"]
    for array in (args[0], args[1], args[2], args[4]):
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
    assert isinstance(args[3], np.ndarray)
    assert args[3].dtype == np.uintp
    assert args[3].flags.c_contiguous
    assert isinstance(args[5], np.ndarray)
    assert args[5].dtype == np.bool_
    assert args[5].flags.c_contiguous
    assert type(args[6]) is float
    assert result["selected"] == 1
