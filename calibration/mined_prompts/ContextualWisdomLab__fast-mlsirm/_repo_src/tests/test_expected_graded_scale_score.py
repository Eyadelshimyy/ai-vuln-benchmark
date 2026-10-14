"""Tests for expected total graded scale score composition."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.estimators.mmle import equal_probability_normal_nodes
from fast_mlsirm.graded_item import (
    compute_expected_graded_item_score,
    expected_graded_scale_score,
)

N_NODES = 2
THETA = np.array([0.30, 0.0, 0.0, 1.10])
FACTOR_VARIANCES = np.array([1.0, 2.25, 4.0, 1.0])
SLOPES = np.array(
    [
        [1.20, 0.85, 0.40, -0.25],
        [0.95, 0.70, 0.55, -0.15],
    ]
)
THRESHOLDS = np.array(
    [
        [1.05, 0.10, -0.70],
        [0.90, -0.05, -0.85],
    ]
)
INTEGRATE_COLUMNS = [(1,), (1, 2)]


def _scaled_nodes_weights(column: int) -> tuple[np.ndarray, np.ndarray]:
    base_nodes, base_weights = equal_probability_normal_nodes(N_NODES)
    scale = float(np.sqrt(FACTOR_VARIANCES[column]))
    return base_nodes * scale, base_weights


def _per_item_sum(theta: np.ndarray = THETA) -> float:
    total = 0.0
    for item_index, columns in enumerate(INTEGRATE_COLUMNS):
        nodes = tuple(_scaled_nodes_weights(column)[0] for column in columns)
        weights = tuple(_scaled_nodes_weights(column)[1] for column in columns)
        total += compute_expected_graded_item_score(
            SLOPES[item_index],
            THRESHOLDS[item_index],
            theta,
            columns,
            nodes,
            weights,
            max_grid_points=4,
        )
    return total


def test_total_equals_sum_of_per_item_calls() -> None:
    reference = _per_item_sum()
    result = expected_graded_scale_score(
        SLOPES,
        THRESHOLDS,
        THETA,
        INTEGRATE_COLUMNS,
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
        max_grid_points=6,
    )
    assert result == reference


def test_scaled_nodes_not_unit_variance() -> None:
    unscaled_reference = 0.0
    for item_index, columns in enumerate(INTEGRATE_COLUMNS):
        base_nodes, base_weights = equal_probability_normal_nodes(N_NODES)
        nodes = tuple(base_nodes for _ in columns)
        weights = tuple(base_weights for _ in columns)
        unscaled_reference += compute_expected_graded_item_score(
            SLOPES[item_index],
            THRESHOLDS[item_index],
            THETA,
            columns,
            nodes,
            weights,
            max_grid_points=4,
        )

    scaled_result = expected_graded_scale_score(
        SLOPES,
        THRESHOLDS,
        THETA,
        INTEGRATE_COLUMNS,
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
        max_grid_points=6,
    )
    assert scaled_result != unscaled_reference


def test_fixed_general_factor_preserved() -> None:
    baseline = expected_graded_scale_score(
        SLOPES,
        THRESHOLDS,
        THETA,
        INTEGRATE_COLUMNS,
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
        max_grid_points=6,
    )
    shifted_theta = THETA.copy()
    shifted_theta[0] = 1.75
    shifted = expected_graded_scale_score(
        SLOPES,
        THRESHOLDS,
        shifted_theta,
        INTEGRATE_COLUMNS,
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
        max_grid_points=6,
    )
    assert shifted != baseline
    assert shifted == _per_item_sum(shifted_theta)


def test_one_and_two_column_items_integrated() -> None:
    one_column = expected_graded_scale_score(
        SLOPES[:1],
        THRESHOLDS[:1],
        THETA,
        INTEGRATE_COLUMNS[:1],
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
        max_grid_points=6,
    )
    two_column_only = expected_graded_scale_score(
        SLOPES[1:],
        THRESHOLDS[1:],
        THETA,
        INTEGRATE_COLUMNS[1:],
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
        max_grid_points=6,
    )
    combined = expected_graded_scale_score(
        SLOPES,
        THRESHOLDS,
        THETA,
        INTEGRATE_COLUMNS,
        FACTOR_VARIANCES,
        n_nodes=N_NODES,
        max_grid_points=6,
    )
    assert combined == one_column + two_column_only


def _scale_call(**overrides) -> float:
    kwargs = {
        "slopes": SLOPES,
        "thresholds": THRESHOLDS,
        "theta": THETA,
        "integrate_columns": INTEGRATE_COLUMNS,
        "factor_variances": FACTOR_VARIANCES,
        "n_nodes": N_NODES,
        "max_grid_points": 6,
    }
    kwargs.update(overrides)
    return expected_graded_scale_score(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"slopes": np.zeros((0, 4))}, "slopes must be a non-empty 2-D array"),
        ({"slopes": np.where(SLOPES == SLOPES[0, 0], np.nan, SLOPES)}, "slopes must be finite"),
        ({"theta": THETA[:3]}, "theta must be a 1-D array with length n_dims"),
        ({"theta": np.array([0.3, np.nan, 0.0, 1.1])}, "theta must be finite"),
        ({"thresholds": THRESHOLDS[0]}, r"thresholds must be a 2-D array with shape"),
        ({"thresholds": np.zeros((2, 0))}, "at least one boundary per item"),
        ({"thresholds": np.where(THRESHOLDS == THRESHOLDS[0, 0], np.inf, THRESHOLDS)},
         "thresholds must be finite"),
        ({"factor_variances": FACTOR_VARIANCES[:3]}, "factor_variances must be a 1-D array"),
        ({"factor_variances": np.array([1.0, np.nan, 4.0, 1.0])}, "factor_variances must be finite"),
        ({"factor_variances": np.array([1.0, -2.25, 4.0, 1.0])}, "factor_variances must be non-negative"),
        ({"integrate_columns": INTEGRATE_COLUMNS[:1]}, "one entry per item"),
    ],
)
def test_scale_score_rejects_malformed_inputs(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _scale_call(**overrides)
