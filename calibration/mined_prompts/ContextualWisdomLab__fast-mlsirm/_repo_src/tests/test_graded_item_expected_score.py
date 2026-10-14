"""Hand-checked integration for one graded item's expected category score."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.estimators.marginal import compute_grm_category_logprobs
from fast_mlsirm.graded_item import compute_expected_graded_item_score

SLOPE = np.array([1.20, 0.85, 0.40, -0.25])
THETA = np.array([0.30, 0.0, 0.0, 1.10])
THRESHOLDS = np.array([1.05, 0.10, -0.70])
NODES = np.array([-0.80, 0.60])
WEIGHTS = np.array([0.35, 0.65])


def _hand_expected(theta_eff: np.ndarray) -> float:
    eta = float(SLOPE @ theta_eff)
    log_probs = compute_grm_category_logprobs(np.array([eta]), THRESHOLDS)[0]
    probs = np.exp(log_probs)
    return float((probs * np.arange(probs.size)).sum())


def _hand_one_column(column: int, theta: np.ndarray = THETA) -> float:
    total = 0.0
    for node, weight in zip(NODES, WEIGHTS, strict=True):
        theta_eff = theta.copy()
        theta_eff[column] = node
        total += weight * _hand_expected(theta_eff)
    return total


def _hand_two_columns(columns: tuple[int, int]) -> float:
    total = 0.0
    for j, node_j in enumerate(NODES):
        for k, node_k in enumerate(NODES):
            theta_eff = THETA.copy()
            theta_eff[columns[0]] = node_j
            theta_eff[columns[1]] = node_k
            total += WEIGHTS[j] * WEIGHTS[k] * _hand_expected(theta_eff)
    return total


def test_one_integrated_column_matches_hand_computation() -> None:
    reference = _hand_one_column(1)
    result = compute_expected_graded_item_score(
        SLOPE,
        THRESHOLDS,
        THETA,
        integrate_columns=(1,),
        integration_nodes=(NODES,),
        integration_weights=(WEIGHTS,),
        max_grid_points=2,
    )
    assert result == pytest.approx(reference, rel=1e-12)


def test_two_integrated_columns_use_product_weights() -> None:
    reference = _hand_two_columns((1, 2))
    result = compute_expected_graded_item_score(
        SLOPE,
        THRESHOLDS,
        THETA,
        integrate_columns=(1, 2),
        integration_nodes=(NODES, NODES),
        integration_weights=(WEIGHTS, WEIGHTS),
        max_grid_points=4,
    )
    assert result == pytest.approx(reference, rel=1e-12)


def test_non_integrated_column_stays_at_fixed_value() -> None:
    baseline_theta = THETA.copy()
    shifted_theta = THETA.copy()
    shifted_theta[3] = 2.25

    baseline = compute_expected_graded_item_score(
        SLOPE,
        THRESHOLDS,
        baseline_theta,
        integrate_columns=(1,),
        integration_nodes=(NODES,),
        integration_weights=(WEIGHTS,),
        max_grid_points=2,
    )
    shifted = compute_expected_graded_item_score(
        SLOPE,
        THRESHOLDS,
        shifted_theta,
        integrate_columns=(1,),
        integration_nodes=(NODES,),
        integration_weights=(WEIGHTS,),
        max_grid_points=2,
    )

    assert shifted != baseline
    assert shifted == pytest.approx(_hand_one_column(1, shifted_theta), rel=1e-12)


def _call_with_weights(weights: np.ndarray) -> float:
    return compute_expected_graded_item_score(
        SLOPE,
        THRESHOLDS,
        THETA,
        integrate_columns=(1,),
        integration_nodes=(NODES,),
        integration_weights=(weights,),
        max_grid_points=2,
    )


def test_valid_normalized_weights_return_score_in_category_range() -> None:
    result = _call_with_weights(WEIGHTS)
    max_category = THRESHOLDS.size
    assert 0.0 <= result <= max_category


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (np.array([-0.10, 1.10]), "must be non-negative"),
        (np.array([0.0, 0.0]), "must sum to one"),
        (np.array([0.10, 0.10]), "must sum to one"),
        (np.array([0.80, 0.80]), "must sum to one"),
        (np.array([0.35, np.nan]), "must be finite"),
        (np.array([0.35, np.inf]), "must be finite"),
        (np.array([1e308, 1e308]), "must sum to one"),
    ],
)
def test_integration_weights_fail_closed_on_invalid_probability_measure(
    weights: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _call_with_weights(weights)


def _call_with(**overrides) -> float:
    kwargs = {
        "slope": SLOPE,
        "thresholds": THRESHOLDS,
        "theta": THETA,
        "integrate_columns": (1,),
        "integration_nodes": (NODES,),
        "integration_weights": (WEIGHTS,),
        "max_grid_points": 4,
    }
    kwargs.update(overrides)
    return compute_expected_graded_item_score(**kwargs)


def test_no_integrated_columns_is_the_plug_in_expectation() -> None:
    result = _call_with(integrate_columns=(), integration_nodes=(), integration_weights=())
    assert result == pytest.approx(_hand_expected(THETA), rel=1e-12)


@pytest.mark.parametrize("column", [True, 1.0, 1.5, "1", -1, 4])
def test_integrate_columns_rejects_non_integer_or_out_of_range(column) -> None:
    with pytest.raises(ValueError, match="integrate_columns entries"):
        _call_with(integrate_columns=(column,))


def test_integrate_columns_accepts_numpy_integer() -> None:
    assert _call_with(integrate_columns=(np.int64(1),)) == pytest.approx(
        _hand_one_column(1), rel=1e-12
    )


@pytest.mark.parametrize(
    "thresholds",
    [np.array([-0.70, 0.10, 1.05]), np.array([1.05, 1.05, -0.70])],
)
def test_thresholds_must_be_strictly_decreasing(thresholds: np.ndarray) -> None:
    with pytest.raises(ValueError, match="strictly decreasing"):
        _call_with(thresholds=thresholds)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"slope": np.array([])}, "slope must be a non-empty 1-D array"),
        ({"slope": np.array([1.0, np.nan, 0.4, -0.25])}, "slope must be finite"),
        ({"theta": THETA[:3]}, "theta must be a 1-D array with the same length"),
        ({"theta": np.array([0.3, 0.0, np.inf, 1.1])}, "theta must be finite"),
        ({"thresholds": np.array([])}, "thresholds must be a 1-D array"),
        ({"thresholds": np.array([1.05, np.nan])}, "thresholds must be finite"),
        ({"integrate_columns": (1, 1), "integration_nodes": (NODES, NODES),
          "integration_weights": (WEIGHTS, WEIGHTS)}, "must not contain duplicates"),
        ({"integration_nodes": ()}, "must align with integrate_columns"),
        ({"integration_nodes": (np.array([]),), "integration_weights": (np.array([]),)},
         r"integration_nodes\[0\] must be a non-empty 1-D array"),
        ({"integration_weights": (WEIGHTS[:1],)}, r"integration_weights\[0\] must be a 1-D array matching"),
        ({"integration_nodes": (np.array([-0.8, np.nan]),)}, "integration nodes must be finite"),
    ],
)
def test_item_score_rejects_malformed_inputs(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _call_with(**overrides)
