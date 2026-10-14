"""Regressions for caller-defined array providers at exposure trust boundaries."""

from __future__ import annotations

import builtins
from collections.abc import Callable

import numpy as np
import pytest

from fast_mlsirm import exposure


class _ProviderBomb:
    """An array provider whose protocol callback must never be executed."""

    callbacks = 0

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        type(self).callbacks += 1
        raise AssertionError("caller __array__ callback executed")


class _FloatSubclassBomb(float):
    """A numeric subclass whose conversion callback must never be executed."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller numeric callback executed")


def _guard_native_discovery(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record any compiled-core discovery attempted during rejected admission."""

    discoveries: list[str] = []
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if level == 1 and "_core" in fromlist:
            discoveries.append(name)
            raise AssertionError("native core discovered before array admission")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return discoveries


@pytest.mark.parametrize(
    ("invoke", "message"),
    [
        (
            lambda value: exposure._as_real_numeric_array("a", value),
            "a must be a real numeric array",
        ),
        (
            lambda value: exposure._as_boolean_array("administered", value),
            "administered must be a boolean array",
        ),
        (
            lambda value: exposure._as_binary_response_array("responses", value),
            "responses must be a real numeric array",
        ),
    ],
)
def test_shared_array_admission_rejects_protocol_providers_without_callbacks(
    invoke: Callable[[object], object],
    message: str,
) -> None:
    """Shared helpers establish inert container identity before ``np.asarray``."""

    _ProviderBomb.callbacks = 0
    with pytest.raises(ValueError, match=message):
        invoke(_ProviderBomb())
    assert _ProviderBomb.callbacks == 0


def test_shared_array_admission_preserves_plain_builtin_sequences() -> None:
    """Plain built-in sequences retain their historical array-like compatibility."""

    real = exposure._as_real_numeric_array("a", [1, 2.5])
    mask = exposure._as_boolean_array("administered", [True, False])
    binary = exposure._as_binary_response_array("responses", (1, 0.0))

    assert np.array_equal(real, np.array([1.0, 2.5]))
    assert real.dtype == np.float64 and real.flags.c_contiguous
    assert np.array_equal(mask, np.array([True, False]))
    assert mask.dtype == np.bool_ and mask.flags.c_contiguous
    assert np.array_equal(binary, np.array([1, 0], dtype=np.uint8))
    assert binary.flags.c_contiguous


def test_shared_array_admission_preserves_concrete_numpy_scalars_in_sequences() -> None:
    """Concrete NumPy scalar identities remain safe array-like compatibility."""

    real = exposure._as_real_numeric_array(
        "a", [np.int16(1), np.float32(2.5)]
    )
    mask = exposure._as_boolean_array(
        "administered", (np.bool_(True), np.bool_(False))
    )
    binary = exposure._as_binary_response_array(
        "responses", [np.uint8(1), np.float64(0.0)]
    )

    assert np.array_equal(real, np.array([1.0, 2.5]))
    assert real.dtype == np.float64 and real.flags.c_contiguous
    assert np.array_equal(mask, np.array([True, False]))
    assert mask.dtype == np.bool_ and mask.flags.c_contiguous
    assert np.array_equal(binary, np.array([1, 0], dtype=np.uint8))
    assert binary.flags.c_contiguous


def test_plain_sequence_admission_rejects_numeric_subclasses_without_callbacks() -> None:
    """Sequence compatibility must not reopen caller-defined scalar conversion."""

    _FloatSubclassBomb.callbacks = 0
    with pytest.raises(ValueError, match="a must be a real numeric array"):
        exposure._as_real_numeric_array("a", [1.0, _FloatSubclassBomb(2.0)])
    assert _FloatSubclassBomb.callbacks == 0


def test_ccat_groups_reject_array_provider_before_callback_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inline CCAT group admission cannot execute caller ``__array__`` hooks."""

    discoveries = _guard_native_discovery(monkeypatch)
    _ProviderBomb.callbacks = 0

    with pytest.raises(ValueError, match="groups must be a real numeric array"):
        exposure.ccat_select(
            np.array([1.0]),
            np.array([0.0]),
            groups=_ProviderBomb(),
            targets=np.array([1.0]),
            administered=np.array([False]),
            theta0=0.0,
        )

    assert _ProviderBomb.callbacks == 0
    assert discoveries == []


def test_flexilevel_rejects_array_provider_before_callback_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flexilevel valid controls still require inert response container identity."""

    discoveries = _guard_native_discovery(monkeypatch)
    _ProviderBomb.callbacks = 0

    with pytest.raises(ValueError, match="responses must be a real numeric array"):
        exposure.flexilevel_administer(_ProviderBomb(), n_persons=1, n_items=3)

    assert _ProviderBomb.callbacks == 0
    assert discoveries == []


def test_shared_array_admission_preserves_exact_ndarray_inputs() -> None:
    """Ordinary package-supported NumPy storage retains normalized marshalling."""

    real = exposure._as_real_numeric_array("a", np.array([1], dtype=np.int16))
    mask = exposure._as_boolean_array("administered", np.array([True, False]))
    binary = exposure._as_binary_response_array(
        "responses", np.array([1.0, 0.0], dtype=np.float32)
    )

    assert real.dtype == np.float64 and real.flags.c_contiguous
    assert mask.dtype == np.bool_ and mask.flags.c_contiguous
    assert binary.dtype == np.uint8 and binary.flags.c_contiguous
