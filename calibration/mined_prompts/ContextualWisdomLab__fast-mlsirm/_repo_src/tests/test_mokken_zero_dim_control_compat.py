"""Compatibility regression for trusted zero-dimensional Mokken controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import mokken


def test_mokken_preserves_zero_dimensional_numpy_real_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact 0-D numeric ndarrays retain their historical scalar semantics."""

    captured: dict[str, object] = {}

    class _Core:
        def mokken_coef_h(
            self,
            responses: np.ndarray,
            n_persons: int,
            n_items: int,
        ) -> dict[str, object]:
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
            captured.update(lower_bound=lower_bound, alpha=alpha)
            return [1, 1]

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    result = mokken.mokken_analysis(
        np.array([[0, 1], [1, 0]], dtype=np.int16),
        lower_bound=np.array(0.3, dtype=np.float32),
        alpha=np.array(0.05, dtype=np.float64),
    )

    assert type(captured["lower_bound"]) is float
    assert type(captured["alpha"]) is float
    assert result.scale.tolist() == [1, 1]
