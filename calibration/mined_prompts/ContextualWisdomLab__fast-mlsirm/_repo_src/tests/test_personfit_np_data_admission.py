"""Regression coverage for nonparametric person-fit evidence admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.personfit_np import person_fit_np


class _HostileArrayProvider:
    """Array-protocol provider that must not execute during admission."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, dtype=None, copy=None):  # noqa: ANN001, ANN201, ARG002
        """Fail if package validation invokes caller-owned array conversion."""
        self.calls += 1
        raise AssertionError("caller __array__ must not execute")


class _HostileFloat(float):
    """Numeric subclass that must not enter NumPy materialization."""

    def __new__(cls, value: float):
        obj = super().__new__(cls, value)
        obj.calls = 0
        return obj

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("caller __float__ must not execute")


def _unexpected_core():
    raise AssertionError("compiled core must not be discovered")


def test_person_fit_np_rejects_array_provider_before_callback_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject arbitrary array providers before protocol or native discovery."""
    provider = _HostileArrayProvider()
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="x must be a numeric 2-D array"):
        person_fit_np(provider)

    assert provider.calls == 0


def test_person_fit_np_rejects_numeric_subclass_cells_before_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject callback-bearing numeric cells before NumPy can coerce them."""
    hostile = _HostileFloat(1.0)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="x must be a numeric array"):
        person_fit_np([[hostile, 0.0], [0.0, 1.0]])

    assert hostile.calls == 0


def test_person_fit_np_preserves_plain_sequence_numpy_scalar_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusted built-in matrices retain their contiguous float64 Rust payload."""
    captured: dict[str, object] = {}

    class _Core:
        """Minimal Rust-boundary stand-in for marshalling assertions."""

        @staticmethod
        def py_person_fit_np(x: np.ndarray, n: int, ni: int) -> dict[str, np.ndarray]:
            captured["x"] = x.copy()
            captured["n"] = n
            captured["ni"] = ni
            return {
                "g": np.zeros(n),
                "gnormed": np.zeros(n),
                "nci": np.zeros(n),
                "u3": np.zeros(n),
                "zu3": np.zeros(n),
                "c_sato": np.zeros(n),
                "cstar": np.zeros(n),
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    result = person_fit_np(
        [
            (np.uint8(0), np.int16(1)),
            (np.float32(1.0), np.float64(0.0)),
        ]
    )

    payload = captured["x"]
    assert isinstance(payload, np.ndarray)
    assert payload.dtype == np.float64
    assert payload.flags.c_contiguous
    assert np.array_equal(payload, np.array([0.0, 1.0, 1.0, 0.0]))
    assert captured["n"] == 2
    assert captured["ni"] == 2
    assert result.g.shape == (2,)
