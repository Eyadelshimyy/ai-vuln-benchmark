"""Callback-safety regressions for public S-X² scalar controls."""

from __future__ import annotations

from types import SimpleNamespace

import fast_mlsirm.fitstats as fitstats_module
import numpy as np
import pytest

from fast_mlsirm.fitstats import s_x2


class _HostileInt(int):
    """Integer subclass that records caller-dispatchable conversion."""

    def __new__(cls, value: int, calls: list[str]) -> "_HostileInt":
        """Attach the callback log to one hostile integer scalar."""
        instance = super().__new__(cls, value)
        instance.calls = calls
        return instance

    def __int__(self) -> int:
        """Fail if S-X² control validation dispatches caller conversion."""
        self.calls.append("int")
        raise AssertionError("caller integer callback executed")


class _HostileFloat(float):
    """Floating subclass that records caller-dispatchable conversion."""

    def __new__(cls, value: float, calls: list[str]) -> "_HostileFloat":
        """Attach the callback log to one hostile floating scalar."""
        instance = super().__new__(cls, value)
        instance.calls = calls
        return instance

    def __float__(self) -> float:
        """Fail if S-X² control validation dispatches caller conversion."""
        self.calls.append("float")
        raise AssertionError("caller floating callback executed")


def _params() -> SimpleNamespace:
    """Return minimal valid item parameters for pre-native control tests."""
    return SimpleNamespace(
        alpha=np.zeros(4),
        b=np.zeros(4),
        zeta=np.zeros((4, 1)),
        tau=-30.0,
    )


@pytest.mark.parametrize(
    ("field", "kind", "value"),
    [
        ("q_theta", "int", 21),
        ("q_xi", "int", 11),
        ("min_expected", "float", 5.0),
        ("fdr_q", "float", 0.05),
        ("min_effect", "float", 0.1),
    ],
)
def test_sx2_rejects_numeric_subclasses_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    kind: str,
    value: float,
) -> None:
    """Every S-X² scalar class fails closed before caller conversion or Rust."""

    class BombCore:
        def s_x2_stat(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("unsafe S-X² control reached native core")

    calls: list[str] = []
    hostile = (
        _HostileInt(int(value), calls)
        if kind == "int"
        else _HostileFloat(float(value), calls)
    )
    y = np.zeros((3, 4))
    factor_id = np.zeros(4, dtype=np.int64)
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: BombCore())

    with pytest.raises(ValueError, match=field):
        s_x2(y, factor_id, _params(), "MIRT", **{field: hostile})

    assert calls == []


def test_sx2_trusted_numpy_controls_remain_supported() -> None:
    """Concrete NumPy scalar identities preserve the established API contract."""
    normalized = fitstats_module._validate_sx2_controls(
        np.int64(21),
        np.uint8(11),
        np.int16(5),
        np.float64(0.05),
        np.float16(0.1),
    )

    assert normalized[:2] == (21, 11)
    assert all(type(value) is int for value in normalized[:2])
    assert all(type(value) is float for value in normalized[2:])


def test_sx2_rejects_unrepresentable_builtin_integer_as_value_error() -> None:
    """An exact but non-float-representable integer fails with the stable contract."""
    with pytest.raises(ValueError, match="min_expected must be a finite number"):
        fitstats_module._validate_sx2_controls(7, 7, 10**400, 0.05, 0.1)


@pytest.mark.parametrize("value", [2**53 + 1, np.uint64(2**53 + 1)])
def test_sx2_rejects_lossy_integer_real_controls(value: object) -> None:
    """Integer-valued real controls must not change identity during float64 normalization."""
    with pytest.raises(ValueError, match="min_expected must be exactly representable as float64"):
        fitstats_module._validate_sx2_controls(7, 7, value, 0.05, 0.1)


def test_sx2_rejects_lossy_longdouble_real_control() -> None:
    """Extended-precision controls cannot be silently rounded to Rust f64."""
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform longdouble has no precision beyond float64")

    value = np.nextafter(np.longdouble(1.0), np.longdouble(2.0))
    assert np.longdouble(float(value)) != value

    with pytest.raises(
        ValueError,
        match="min_expected must be exactly representable as float64",
    ):
        fitstats_module._validate_sx2_controls(7, 7, value, 0.05, 0.1)


def test_sx2_preserves_lossless_longdouble_real_control() -> None:
    """An exact long-double value inside the binary64 lattice remains supported."""
    normalized = fitstats_module._validate_sx2_controls(
        7,
        7,
        np.longdouble(0.5),
        0.05,
        0.1,
    )

    assert normalized[2] == 0.5
    assert type(normalized[2]) is float
