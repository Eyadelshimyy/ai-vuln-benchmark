"""Trust-boundary regressions for Chang-Ying KL CAT wrappers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm import exposure


class _UnexpectedCore:
    """Fail if rejected caller evidence reaches native KL dispatch."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"native core reached before KL admission: {name}")


class _HostileReal:
    """Object-array element whose numeric conversion must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller __float__ executed during KL admission")


class _HostileBool:
    """Object-array element whose truth conversion must never execute."""

    callbacks = 0

    def __bool__(self) -> bool:
        type(self).callbacks += 1
        raise AssertionError("caller __bool__ executed during KL admission")


class _FloatProvider:
    """Semantic-control provider that must be rejected without conversion."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller __float__ executed during control admission")


def _information(a: object, b: object, c: object | None) -> object:
    return exposure.kl_information(a, b, c, theta0=0.0, delta=0.5)


def _select(a: object, b: object, c: object | None) -> object:
    return exposure.kl_select(
        a,
        b,
        c,
        administered=np.array([False]),
        theta0=0.0,
        n_administered=1,
    )


@pytest.mark.parametrize("invoke", [_information, _select])
@pytest.mark.parametrize("field", ["a", "b", "c"])
def test_complex_item_evidence_fails_before_lossy_cast_or_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    invoke: Callable[[object, object, object | None], object],
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
        invoke(values["a"], values["b"], values["c"])


@pytest.mark.parametrize("invoke", [_information, _select])
def test_object_item_storage_fails_without_element_conversion(
    monkeypatch: pytest.MonkeyPatch,
    invoke: Callable[[object, object, object | None], object],
) -> None:
    """Object item storage must fail before per-element numeric conversion."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileReal.callbacks = 0
    hostile = np.array([_HostileReal()], dtype=object)

    with pytest.raises(ValueError, match="a must be a real numeric array"):
        invoke(hostile, np.array([0.0]), np.array([0.0]))

    assert _HostileReal.callbacks == 0


def test_kl_select_requires_boolean_administered_storage_before_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection masks must not be synthesized through numeric truth coercion."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    with pytest.raises(ValueError, match="administered must be a boolean array"):
        exposure.kl_select(
            np.array([1.0]),
            np.array([0.0]),
            administered=np.array([0], dtype=np.int8),
            theta0=0.0,
            n_administered=1,
        )


def test_kl_select_rejects_object_mask_without_truth_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Object mask storage must fail before caller truth callbacks execute."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileBool.callbacks = 0
    mask = np.array([_HostileBool()], dtype=object)

    with pytest.raises(ValueError, match="administered must be a boolean array"):
        exposure.kl_select(
            np.array([1.0]),
            np.array([0.0]),
            administered=mask,
            theta0=0.0,
            n_administered=1,
        )

    assert _HostileBool.callbacks == 0


@pytest.mark.parametrize(
    ("entrypoint", "control"),
    [
        ("information", "theta0"),
        ("information", "delta"),
        ("select", "theta0"),
        ("select", "r"),
    ],
)
def test_real_controls_fail_before_callbacks_data_or_native(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    control: str,
) -> None:
    """KL real controls require trusted scalar identities before data work."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _FloatProvider.callbacks = 0
    hostile = _FloatProvider()

    if entrypoint == "information":
        kwargs: dict[str, object] = {"theta0": 0.0, "delta": 0.5}
        kwargs[control] = hostile
        with pytest.raises(ValueError, match=rf"{control} must be a real scalar"):
            exposure.kl_information(np.array([1.0]), np.array([0.0]), **kwargs)
    else:
        kwargs = {"theta0": 0.0, "r": 3.0}
        kwargs[control] = hostile
        with pytest.raises(ValueError, match=rf"{control} must be a real scalar"):
            exposure.kl_select(
                np.array([1.0]),
                np.array([0.0]),
                administered=np.array([False]),
                n_administered=1,
                **kwargs,
            )

    assert _FloatProvider.callbacks == 0


def test_valid_kl_inputs_preserve_native_marshalling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accepted arrays and controls reach Rust as normalized built-in payloads."""

    captured: dict[str, tuple[object, ...]] = {}

    class _Core:
        def py_kl_information(self, *args: object) -> np.ndarray:
            captured["information"] = args
            return np.array([0.25])

        def py_kl_select(self, *args: object) -> dict[str, object]:
            captured["select"] = args
            return {"index": np.array([0.25]), "selected": 0, "delta": 3.0}

    monkeypatch.setattr(fast_mlsirm, "_core", _Core(), raising=False)
    a = np.array([1], dtype=np.int16)
    b = np.array([0.5], dtype=np.float32)

    exposure.kl_information(a, b, theta0=np.float32(0.0), delta=np.float64(0.5))
    exposure.kl_select(
        a,
        b,
        administered=np.array([False], dtype=np.bool_),
        theta0=np.float32(0.0),
        n_administered=np.int32(1),
        r=np.float64(3.0),
    )

    info_args = captured["information"]
    for array in info_args[:3]:
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
    assert type(info_args[3]) is float
    assert type(info_args[4]) is float

    select_args = captured["select"]
    for array in select_args[:3]:
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
    assert isinstance(select_args[3], np.ndarray)
    assert select_args[3].dtype == np.bool_
    assert select_args[3].flags.c_contiguous
    assert type(select_args[4]) is float
    assert type(select_args[5]) is int
    assert type(select_args[6]) is float
