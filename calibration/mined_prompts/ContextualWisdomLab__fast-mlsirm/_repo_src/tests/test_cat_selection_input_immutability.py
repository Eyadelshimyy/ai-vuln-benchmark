"""Regression contracts for CAT public-input immutability."""

from __future__ import annotations

import numpy as np

from fast_mlsirm.test_design import item_information, select_cat_item
from fast_mlsirm.types import MLSIRMParams


def _bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Return a small one-dimensional calibrated bank and factor map."""
    discrimination = np.array([0.8, 1.1, 1.4, 1.7], dtype=np.float64)
    bank = MLSIRMParams(
        theta=np.array([[0.0]], dtype=np.float64),
        alpha=np.log(discrimination),
        b=np.array([-1.0, -0.2, 0.4, 1.2], dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((4, 1), dtype=np.float64),
        tau=-30.0,
    )
    return bank, np.zeros(4, dtype=np.int64)


def test_item_information_does_not_mutate_theta_or_factor_map() -> None:
    """Rust marshalling for item information leaves caller arrays unchanged."""
    bank, factor_id = _bank()
    theta = np.array([-0.5], dtype=np.float64)
    theta_before = theta.copy()
    factor_before = factor_id.copy()

    item_information(bank, factor_id, theta=theta, model="MIRT")

    assert np.array_equal(theta, theta_before)
    assert np.array_equal(factor_id, factor_before)


def test_cat_selection_does_not_mutate_public_arrays() -> None:
    """Rust selection marshalling preserves theta, factor, and administered inputs."""
    bank, factor_id = _bank()
    theta = np.array([0.25], dtype=np.float64)
    administered = np.array([2, 0], dtype=np.int64)
    theta_before = theta.copy()
    factor_before = factor_id.copy()
    administered_before = administered.copy()

    selected = select_cat_item(
        bank,
        factor_id,
        theta=theta,
        administered=administered,
        model="MIRT",
    )

    assert selected in {1, 3}
    assert np.array_equal(theta, theta_before)
    assert np.array_equal(factor_id, factor_before)
    assert np.array_equal(administered, administered_before)
