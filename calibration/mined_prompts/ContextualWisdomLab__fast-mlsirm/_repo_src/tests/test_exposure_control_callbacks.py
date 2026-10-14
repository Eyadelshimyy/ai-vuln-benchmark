"""Callback-safety regressions for exposure-control scalar marshalling."""

from __future__ import annotations

import builtins
from collections.abc import Callable

import numpy as np
import pytest

from fast_mlsirm import exposure


class _HostileInt(int):
    """Integer subclass that records forbidden coercion callbacks."""

    callbacks: list[str] = []

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        raise AssertionError("integer conversion callback executed")

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        raise AssertionError("representation callback executed")


class _HostileNumpyInt(np.int64):
    """NumPy integer subclass that must never be normalized."""

    callbacks: list[str] = []

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        raise AssertionError("NumPy integer conversion callback executed")

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        raise AssertionError("representation callback executed")


class _HostileFloat(float):
    """Floating subclass that must be rejected before integer coercion."""

    callbacks: list[str] = []

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        raise AssertionError("float integer conversion callback executed")

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        raise AssertionError("representation callback executed")


@pytest.mark.parametrize("hostile_type", [_HostileInt, _HostileNumpyInt, _HostileFloat])
def test_as_int_rejects_scalar_subclasses_without_callbacks(hostile_type) -> None:
    """Shared integer validation must establish exact scalar identity first."""

    hostile_type.callbacks.clear()
    with pytest.raises(ValueError, match="test_length must be an integer"):
        exposure._as_int("test_length", hostile_type(4), minimum=1, maximum=8)

    assert hostile_type.callbacks == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.int8(2), 2),
        (np.uint64(3), 3),
        (np.float32(4.0), 4),
        (np.float64(5.0), 5),
        (np.longdouble(6.0), 6),
        (7.0, 7),
    ],
)
def test_as_int_preserves_genuine_supported_numpy_and_builtin_scalars(
    value: object,
    expected: int,
) -> None:
    """Exact genuine numeric scalars preserve the established integral contract."""

    assert exposure._as_int("test_length", value, minimum=1, maximum=8) == expected


@pytest.mark.parametrize(
    "value",
    [True, 1.5, np.float32(2.5), np.inf, np.float64(np.nan), object()],
)
def test_as_int_rejects_non_integral_or_unsupported_controls(value: object) -> None:
    """Malformed controls retain package-owned validation errors."""

    with pytest.raises(ValueError, match="test_length must be an integer"):
        exposure._as_int("test_length", value, minimum=1, maximum=8)


def test_as_int_preserves_package_owned_range_errors() -> None:
    """Normalized trusted values still use the established package bounds."""

    with pytest.raises(ValueError, match=r"test_length out of range \[1, 8\]"):
        exposure._as_int("test_length", np.int64(9), minimum=1, maximum=8)


def test_sympson_hetter_rejects_hostile_control_before_core_use() -> None:
    """A public exposure boundary must fail closed before result-affecting dispatch."""

    _HostileInt.callbacks.clear()
    with pytest.raises(ValueError, match="test_length must be an integer"):
        exposure.sympson_hetter(
            np.ones(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            test_length=_HostileInt(2),
            n_simulees=4,
            max_iter=1,
            q_theta=5,
        )

    assert _HostileInt.callbacks == []


def test_a_stratified_rejects_hostile_control_without_callbacks() -> None:
    """Shared validation semantics also govern a second CAT public boundary."""

    _HostileNumpyInt.callbacks.clear()
    with pytest.raises(ValueError, match="n_strata must be an integer"):
        exposure.a_stratified(
            np.ones(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            n_strata=_HostileNumpyInt(2),
            test_length=2,
            n_simulees=4,
            q_theta=5,
        )

    assert _HostileNumpyInt.callbacks == []


def _sympson_hetter_with_hostile_control(value: object) -> object:
    return exposure.sympson_hetter(
        np.ones(4), np.zeros(4), test_length=value, n_simulees=4, max_iter=1
    )


def _a_stratified_with_hostile_control(value: object) -> object:
    return exposure.a_stratified(
        np.ones(4), np.zeros(4), n_strata=value, test_length=2, n_simulees=4
    )


def _kl_select_with_hostile_control(value: object) -> object:
    return exposure.kl_select(
        np.ones(1),
        np.zeros(1),
        administered=np.array([False]),
        theta0=0.0,
        n_administered=value,
    )


def _owen_cat_with_hostile_control(value: object) -> object:
    return exposure.owen_cat(
        np.ones(1), np.zeros(1), responses=np.array([1]), test_length=value
    )


def _flexilevel_with_hostile_control(value: object) -> object:
    return exposure.flexilevel_administer(
        np.zeros(3, dtype=np.uint8), n_persons=value, n_items=3
    )


def _stradaptive_with_hostile_control(value: object) -> object:
    return exposure.stradaptive_administer(
        np.array([0, 1]),
        np.array([-1.0, 1.0]),
        np.array([0, 1]),
        entry_stratum=value,
        chance=0.25,
    )


def _pyramidal_with_hostile_control(value: object) -> object:
    return exposure.pyramidal_administer(
        np.array([0.0]), value, np.array([1], dtype=np.uint8)
    )


def _two_stage_route_with_hostile_control(value: object) -> object:
    return exposure.two_stage_route(value, 1, 1.0, 0.0, np.array([0.0]), 0.0)


def _two_stage_score_with_hostile_control(value: object) -> object:
    return exposure.two_stage_score(
        value,
        1,
        1.0,
        0.0,
        1,
        1,
        0,
        np.array([1.0]),
        np.array([0.0]),
        0.0,
    )


@pytest.mark.parametrize(
    ("invoke", "field"),
    [
        (_sympson_hetter_with_hostile_control, "test_length"),
        (_a_stratified_with_hostile_control, "n_strata"),
        (_kl_select_with_hostile_control, "n_administered"),
        (_owen_cat_with_hostile_control, "test_length"),
        (_flexilevel_with_hostile_control, "n_persons"),
        (_stradaptive_with_hostile_control, "entry_stratum"),
        (_pyramidal_with_hostile_control, "n_stages"),
        (_two_stage_route_with_hostile_control, "x1"),
        (_two_stage_score_with_hostile_control, "x1"),
    ],
)
def test_public_integer_controls_fail_before_native_core_discovery(
    monkeypatch: pytest.MonkeyPatch,
    invoke: Callable[[object], object],
    field: str,
) -> None:
    """Invalid controls must be rejected before attempting ``_core`` discovery."""

    _HostileInt.callbacks.clear()
    core_discoveries: list[tuple[str, tuple[str, ...]]] = []
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if level == 1 and "_core" in fromlist:
            core_discoveries.append((name, tuple(fromlist)))
            raise AssertionError("native core discovery preceded control validation")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        invoke(_HostileInt(1))

    assert core_discoveries == []
    assert _HostileInt.callbacks == []
