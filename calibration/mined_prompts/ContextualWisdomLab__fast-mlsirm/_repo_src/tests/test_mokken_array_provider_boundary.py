"""Callback-free response-container regressions for Mokken analysis."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import mokken


class _HostileArrayProvider:
    """Array provider whose NumPy callback must never execute during admission."""

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("response array callback executed")


def test_mokken_rejects_array_provider_before_protocol_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted response providers fail before NumPy or Rust discovery."""

    def _unexpected_core() -> object:
        raise AssertionError("compiled core was discovered")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        mokken.mokken_analysis(_HostileArrayProvider())


def test_mokken_preserves_inert_builtin_and_numpy_row_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary historical array-likes still marshal as built-in Rust payloads."""
    captured: list[np.ndarray] = []

    class _Core:
        def mokken_coef_h(
            self,
            responses: np.ndarray,
            n_persons: int,
            n_items: int,
        ) -> dict[str, object]:
            assert type(responses) is np.ndarray
            assert responses.dtype == np.int64
            assert n_persons == 2
            assert n_items == 2
            captured.append(responses.copy())
            return {
                "hij": [float("nan"), 0.4, 0.4, float("nan")],
                "hi": [0.4, 0.4],
                "h": 0.4,
                "zij": [float("nan"), 1.0, 1.0, float("nan")],
                "zi": [1.0, 1.0],
                "z": 1.0,
            }

        def mokken_aisp(
            self,
            responses: np.ndarray,
            n_persons: int,
            n_items: int,
            lower_bound: float,
            alpha: float,
        ) -> list[int]:
            assert type(responses) is np.ndarray
            assert n_persons == 2
            assert n_items == 2
            assert type(lower_bound) is float
            assert type(alpha) is float
            return [1, 1]

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    builtin = [[np.int8(0), np.uint8(1)], (np.float32(1.0), 0)]
    numpy_rows = [
        np.array([0, 1], dtype=np.int16),
        np.array([1, 0], dtype=np.uint8),
    ]

    assert mokken.mokken_analysis(builtin).scale.tolist() == [1, 1]
    assert mokken.mokken_analysis(numpy_rows).scale.tolist() == [1, 1]
    assert [row.tolist() for row in captured] == [[0, 1, 1, 0], [0, 1, 1, 0]]
