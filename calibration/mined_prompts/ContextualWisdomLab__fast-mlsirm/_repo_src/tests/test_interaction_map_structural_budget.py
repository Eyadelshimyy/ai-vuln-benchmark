"""Residual interaction-map structural-work budget regressions."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from fast_mlsirm import residual_interaction_map

interaction_map_module = importlib.import_module("fast_mlsirm.interaction_map")


def _fake_map_payload(axis_count: int) -> dict[str, object]:
    """Return one minimal shape-consistent interaction-map core payload."""
    return {
        "person_indices": [0, 1],
        "item_indices": [0],
        "scored_person_count": 2,
        "scored_item_count": 1,
        "person_coordinates": [0.0] * (2 * axis_count),
        "item_coordinates": [0.0] * axis_count,
        "singular_values": [1.0],
        "axis_shares": [1.0] + [0.0] * (axis_count - 1),
        "residual": [1.0, 2.0],
        "distance": [0.0, 0.0],
        "reconstruction": [1.0, 2.0],
        "unexplained": [0.0, 0.0],
        "cross_share": [0.0, 0.0],
        "axis_count": axis_count,
    }


def test_empty_row_fanout_hits_structural_budget_before_dense_or_native_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-cell container fan-out is bounded before dense NumPy or Rust work."""
    monkeypatch.setattr(
        interaction_map_module,
        "_MAX_INTERACTION_MAP_CELLS",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        interaction_map_module,
        "_MAX_INTERACTION_MAP_STRUCTURAL_NODES",
        4,
        raising=False,
    )

    def fail_dense(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dense NumPy materialization must not run")

    def fail_core(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("compiled interaction-map core must not run")

    monkeypatch.setattr(interaction_map_module.np, "ascontiguousarray", fail_dense)
    monkeypatch.setattr(
        interaction_map_module._core,
        "residual_interaction_map",
        fail_core,
    )

    with pytest.raises(ValueError, match="structural-node"):
        residual_interaction_map(
            [[], [], [], [], []],
            [[], [], [], [], []],
            axis_count=1,
        )


def test_valid_matrix_at_reduced_structural_boundary_reaches_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The structural envelope preserves every valid matrix at its 2N boundary."""
    monkeypatch.setattr(
        interaction_map_module,
        "_MAX_INTERACTION_MAP_CELLS",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        interaction_map_module,
        "_MAX_INTERACTION_MAP_STRUCTURAL_NODES",
        4,
        raising=False,
    )
    captured: dict[str, object] = {}

    def fake_core(
        observed: np.ndarray,
        expected: np.ndarray,
        axis_count: int,
    ) -> dict[str, object]:
        captured["observed"] = observed
        captured["expected"] = expected
        captured["axis_count"] = axis_count
        return _fake_map_payload(axis_count)

    monkeypatch.setattr(
        interaction_map_module._core,
        "residual_interaction_map",
        fake_core,
    )

    result = residual_interaction_map(
        [[1.0], [2.0]],
        [[0.0], [0.0]],
        axis_count=1,
    )

    assert type(captured["axis_count"]) is int
    assert captured["axis_count"] == 1
    assert isinstance(captured["observed"], np.ndarray)
    assert isinstance(captured["expected"], np.ndarray)
    assert captured["observed"].dtype == np.float64  # type: ignore[union-attr]
    assert captured["expected"].dtype == np.float64  # type: ignore[union-attr]
    np.testing.assert_array_equal(captured["observed"], [[1.0], [2.0]])
    np.testing.assert_array_equal(captured["expected"], [[0.0], [0.0]])
    assert result.person_coordinates.shape == (2, 1)
    assert result.item_coordinates.shape == (1, 1)
