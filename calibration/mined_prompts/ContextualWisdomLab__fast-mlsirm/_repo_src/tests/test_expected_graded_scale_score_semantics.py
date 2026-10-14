"""Existing mean-zero replacement semantics; not fitting/scientific acceptance."""

import numpy as np

from fast_mlsirm.graded_item import expected_graded_scale_score


def _score(theta, variance, columns):
    return expected_graded_scale_score(
        slopes=np.array([[1.4, 0.9]]),
        thresholds=np.array([[1.5, 0.2, -1.1]]),
        theta=theta,
        integrate_columns=[columns],
        factor_variances=np.array([1.0, variance]),
        n_nodes=121,
        max_grid_points=121 if columns else 1,
    )


def test_selected_coordinate_is_replaced_without_mutating_inputs():
    theta = np.array([0.35, 5.0])
    before = theta.copy()
    value = _score(theta, 1.0, [1])
    np.testing.assert_array_equal(theta, before)
    assert value == _score(np.array([0.35, -7.0]), 1.0, [1])
    assert value != _score(np.array([0.85, 5.0]), 1.0, [1])


def test_zero_variance_is_point_mass_at_zero_not_supplied_coordinate():
    integrated = _score(np.array([0.35, 5.0]), 0.0, [1])
    at_zero = _score(np.array([0.35, 0.0]), 0.0, [])
    np.testing.assert_allclose(integrated, at_zero, rtol=0.0, atol=2e-14)
    assert abs(integrated - _score(np.array([0.35, 5.0]), 0.0, [])) > 0.1
