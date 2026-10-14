"""Trust-boundary regressions for Owen CAT wrappers."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm import exposure


class _UnexpectedCore:
    """Fail if rejected caller evidence reaches native Owen dispatch."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"native core reached before Owen admission: {name}")


class _HostileFloat(float):
    """Float subclass whose conversion callback must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller __float__ executed during Owen admission")


class _HostileBool:
    """Truth provider whose callback must never execute."""

    callbacks = 0

    def __bool__(self) -> bool:
        type(self).callbacks += 1
        raise AssertionError("caller __bool__ executed during Owen admission")


class _HostileElement:
    """Object-array cell whose numeric conversion must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller element __float__ executed during Owen admission")


@pytest.mark.parametrize("field", ["a", "b", "c", "mu", "sig2"])
def test_owen_update_rejects_hostile_real_scalar_before_callback_or_native(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Owen scalar controls require package-trusted identities before dispatch."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileFloat.callbacks = 0
    values: dict[str, object] = {
        "a": 1.0,
        "b": 0.0,
        "c": 0.0,
        "mu": 0.0,
        "sig2": 1.0,
    }
    values[field] = _HostileFloat(1.0)

    with pytest.raises(ValueError, match=rf"{field} must be a real scalar"):
        exposure.owen_update(
            values["a"],
            values["b"],
            values["c"],
            correct=True,
            mu=values["mu"],
            sig2=values["sig2"],
        )

    assert _HostileFloat.callbacks == 0


def test_owen_update_rejects_truth_provider_before_callback_or_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Response truth must not be synthesized from caller-defined objects."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileBool.callbacks = 0

    with pytest.raises(ValueError, match="correct must be a boolean scalar"):
        exposure.owen_update(1.0, 0.0, correct=_HostileBool(), mu=0.0, sig2=1.0)

    assert _HostileBool.callbacks == 0


@pytest.mark.parametrize("field", ["mu0", "sig2_0", "sig2_stop"])
def test_owen_cat_rejects_hostile_real_control_before_data_or_native(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """CAT controls must be admitted before caller item/response materialization."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileFloat.callbacks = 0
    values: dict[str, object] = {"mu0": 0.0, "sig2_0": 1.0, "sig2_stop": None}
    values[field] = _HostileFloat(0.5)

    with pytest.raises(ValueError, match=rf"{field} must be a real scalar"):
        exposure.owen_cat(
            np.array([1.0]),
            np.array([0.0]),
            responses=np.array([1]),
            test_length=1,
            mu0=values["mu0"],
            sig2_0=values["sig2_0"],
            sig2_stop=values["sig2_stop"],
        )

    assert _HostileFloat.callbacks == 0


@pytest.mark.parametrize("field", ["a", "b", "c"])
def test_owen_cat_rejects_complex_item_evidence_before_lossy_cast_or_native(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Imaginary item evidence must never be projected onto the real axis."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    values: dict[str, object] = {
        "a": np.array([1.0]),
        "b": np.array([0.0]),
        "c": np.array([0.0]),
    }
    values[field] = np.array([1.0 + 0.25j])

    with pytest.raises(ValueError, match=rf"{field} must be a real numeric array"):
        exposure.owen_cat(
            values["a"],
            values["b"],
            values["c"],
            responses=np.array([1]),
            test_length=1,
        )


def test_owen_cat_rejects_object_item_storage_without_element_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Object item storage must fail before per-element numeric conversion."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileElement.callbacks = 0
    hostile = np.array([_HostileElement()], dtype=object)

    with pytest.raises(ValueError, match="a must be a real numeric array"):
        exposure.owen_cat(
            hostile,
            np.array([0.0]),
            responses=np.array([1]),
            test_length=1,
        )

    assert _HostileElement.callbacks == 0


@pytest.mark.parametrize(
    "responses",
    [
        np.array([1.0 + 0.5j]),
        np.array(["1"], dtype="U1"),
    ],
)
def test_owen_cat_rejects_non_numeric_or_complex_response_storage_before_native(
    monkeypatch: pytest.MonkeyPatch,
    responses: np.ndarray,
) -> None:
    """Binary response evidence must already have real numeric storage."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)

    with pytest.raises(ValueError, match="responses must be a real numeric array"):
        exposure.owen_cat(
            np.array([1.0]),
            np.array([0.0]),
            responses=responses,
            test_length=1,
        )


def test_owen_cat_rejects_object_responses_without_element_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Object response storage must fail before caller numeric callbacks execute."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileElement.callbacks = 0
    responses = np.array([_HostileElement()], dtype=object)

    with pytest.raises(ValueError, match="responses must be a real numeric array"):
        exposure.owen_cat(
            np.array([1.0]),
            np.array([0.0]),
            responses=responses,
            test_length=1,
        )

    assert _HostileElement.callbacks == 0


def test_valid_owen_inputs_preserve_normalized_native_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted scalars and arrays reach Rust as package-normalized primitives."""

    captured: dict[str, tuple[object, ...]] = {}

    class _Core:
        def py_owen_update(self, *args: object) -> tuple[float, float]:
            captured["update"] = args
            return (0.1, 0.9)

        def py_owen_cat(self, *args: object) -> dict[str, object]:
            captured["cat"] = args
            return {
                "administered": [0],
                "mu_trace": np.array([0.0, 0.1]),
                "sig2_trace": np.array([1.0, 0.9]),
                "mu": 0.1,
                "sig2": 0.9,
            }

    monkeypatch.setattr(fast_mlsirm, "_core", _Core(), raising=False)

    exposure.owen_update(
        np.float32(1.0),
        np.int16(0),
        np.float64(0.0),
        correct=np.bool_(True),
        mu=np.float32(0.0),
        sig2=np.float64(1.0),
    )
    exposure.owen_cat(
        np.array([1], dtype=np.int16),
        np.array([0.0], dtype=np.float32),
        responses=np.array([True], dtype=np.bool_),
        test_length=np.int32(1),
        mu0=np.float32(0.0),
        sig2_0=np.float64(1.0),
        sig2_stop=np.float32(0.5),
    )

    update_args = captured["update"]
    assert all(type(value) is float for value in (*update_args[:3], *update_args[4:]))
    assert type(update_args[3]) is bool

    cat_args = captured["cat"]
    for array in cat_args[:3]:
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
    assert isinstance(cat_args[3], np.ndarray)
    assert cat_args[3].dtype == np.uint8
    assert cat_args[3].flags.c_contiguous
    assert type(cat_args[4]) is float
    assert type(cat_args[5]) is float
    assert type(cat_args[6]) is int
    assert type(cat_args[7]) is float
