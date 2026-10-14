"""Data-integrity regressions for CAT exposure item-parameter admission."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm import exposure


class _UnexpectedCore:
    """Fail if invalid caller evidence reaches native CAT dispatch."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"native core reached before item admission: {name}")


class _HostileReal:
    """Object-storage element whose numeric conversion must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller __float__ executed during item admission")


def _sympson(a: object, b: object, c: object | None) -> object:
    return exposure.sympson_hetter(
        a,
        b,
        c,
        test_length=1,
        n_simulees=4,
        max_iter=1,
        q_theta=5,
    )


def _a_stratified(a: object, b: object, c: object | None) -> object:
    return exposure.a_stratified(
        a,
        b,
        c,
        n_strata=1,
        test_length=1,
        n_simulees=4,
        q_theta=5,
    )


@pytest.mark.parametrize("invoke", [_sympson, _a_stratified])
@pytest.mark.parametrize("field", ["a", "b", "c"])
def test_complex_item_parameters_fail_before_lossy_cast_or_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    invoke: Callable[[object, object, object | None], object],
    field: str,
) -> None:
    """Imaginary item evidence must never be projected onto the real axis."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    values: dict[str, object] = {
        "a": np.array([1.0], dtype=np.float64),
        "b": np.array([0.0], dtype=np.float64),
        "c": np.array([0.0], dtype=np.float64),
    }
    values[field] = np.array([1.0 + 0.25j], dtype=np.complex128)

    with pytest.raises(ValueError, match=rf"{field} must be a real numeric array"):
        invoke(values["a"], values["b"], values["c"])


@pytest.mark.parametrize("invoke", [_sympson, _a_stratified])
def test_object_item_storage_fails_without_element_conversion(
    monkeypatch: pytest.MonkeyPatch,
    invoke: Callable[[object, object, object | None], object],
) -> None:
    """Object arrays must fail before caller-defined numeric protocols execute."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileReal.callbacks = 0
    hostile = np.array([_HostileReal()], dtype=object)

    with pytest.raises(ValueError, match="a must be a real numeric array"):
        invoke(hostile, np.array([0.0]), np.array([0.0]))

    assert _HostileReal.callbacks == 0


def test_real_item_parameters_preserve_float64_marshalling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary real arrays retain the existing Rust dispatch dtypes."""

    captured: dict[str, tuple[object, ...]] = {}

    class _Core:
        def py_sympson_hetter(self, *args: object) -> dict[str, object]:
            captured["sympson"] = args
            return {
                "k": [1.0],
                "exposure": [0.25],
                "selection": [0.25],
                "max_exposure": 0.25,
                "n_iter": 1,
                "converged": True,
                "history_max_exposure": [0.25],
            }

        def py_a_stratified(self, *args: object) -> dict[str, object]:
            captured["stratified"] = args
            return {
                "exposure": [0.25],
                "max_exposure": 0.25,
                "stratum": [0],
                "stage_lengths": [1],
                "theta_rmse": 0.0,
                "theta_bias": 0.0,
            }

    monkeypatch.setattr(fast_mlsirm, "_core", _Core(), raising=False)
    a = np.array([1], dtype=np.int16)
    b = np.array([0.5], dtype=np.float32)

    _sympson(a, b, None)
    _a_stratified(a, b, None)

    for key in ("sympson", "stratified"):
        args = captured[key]
        for array in args[:3]:
            assert isinstance(array, np.ndarray)
            assert array.dtype == np.float64
            assert array.flags.c_contiguous
