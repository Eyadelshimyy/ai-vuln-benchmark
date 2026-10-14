"""Regression contracts for caller-defined NumPy scalar subclasses."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.gtheory as gtheory


class _NoNumericCore:
    """Core stub proving rejected controls never reach numerical dispatch."""

    def gtheory_pi(self, *args, **kwargs):
        raise AssertionError("invalid G-theory control reached Rust")

    def phi_lambda(self, *args, **kwargs):
        raise AssertionError("invalid Phi(lambda) control reached Rust")


def _pi_data() -> np.ndarray:
    """Return a minimal one-facet score matrix for marshalling tests."""

    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def test_pi_rejects_spoofed_numpy_integer_subclass_before_callbacks(monkeypatch) -> None:
    """NumPy-looking subclass metadata must not authorize integer coercion."""

    calls: list[str] = []

    class SpoofedNumpyInt(np.int64):
        __module__ = "numpy"

        def __int__(self) -> int:
            calls.append("__int__")
            raise AssertionError("caller integer conversion executed")

        def __repr__(self) -> str:
            calls.append("__repr__")
            raise AssertionError("caller representation executed")

    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"n_i_prime entries must be positive integers"):
        gtheory.gtheory_pi(_pi_data(), n_i_prime=[SpoofedNumpyInt(2)])

    assert calls == []


def test_pi_rejects_hostile_scalar_metaclass_hash_before_callback(monkeypatch) -> None:
    """Integer type admission must not hash a caller-controlled metaclass."""

    calls: list[str] = []

    class HostileMeta(type):
        def __hash__(cls) -> int:
            calls.append("type-__hash__")
            raise AssertionError("type hash callback executed")

    class HostileNumpyInt(np.int64, metaclass=HostileMeta):
        pass

    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"n_i_prime entries must be positive integers"):
        gtheory.gtheory_pi(_pi_data(), n_i_prime=[HostileNumpyInt(2)])

    assert calls == []


def test_phi_lambda_rejects_spoofed_numpy_float_subclass_before_callbacks(
    monkeypatch,
) -> None:
    """NumPy-looking subclass metadata must not authorize float coercion."""

    calls: list[str] = []

    class SpoofedNumpyFloat(np.float64):
        __module__ = "numpy"

        def __float__(self) -> float:
            calls.append("__float__")
            raise AssertionError("caller float conversion executed")

        def __repr__(self) -> str:
            calls.append("__repr__")
            raise AssertionError("caller representation executed")

    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"cut must be a finite real scalar"):
        gtheory.phi_lambda(_pi_data(), SpoofedNumpyFloat(0.5), n_i_prime=[2])

    assert calls == []


def test_phi_lambda_rejects_hostile_scalar_metaclass_hash_before_callback(
    monkeypatch,
) -> None:
    """Float type admission must not hash a caller-controlled metaclass."""

    calls: list[str] = []

    class HostileMeta(type):
        def __hash__(cls) -> int:
            calls.append("type-__hash__")
            raise AssertionError("type hash callback executed")

    class HostileNumpyFloat(np.float64, metaclass=HostileMeta):
        pass

    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"cut must be a finite real scalar"):
        gtheory.phi_lambda(_pi_data(), HostileNumpyFloat(0.5), n_i_prime=[2])

    assert calls == []


def test_phi_lambda_rejects_hostile_scalar_metaclass_equality_before_callback(
    monkeypatch,
) -> None:
    """Float admission must not compare a caller-controlled metaclass for equality."""

    calls: list[str] = []

    class HostileMeta(type):
        __hash__ = type.__hash__

        def __eq__(cls, other: object) -> bool:
            calls.append("type-__eq__")
            raise AssertionError("type equality callback executed")

    class HostileNumpyFloat(np.float64, metaclass=HostileMeta):
        pass

    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"cut must be a finite real scalar"):
        gtheory.phi_lambda(_pi_data(), HostileNumpyFloat(0.5), n_i_prime=[2])

    assert calls == []
