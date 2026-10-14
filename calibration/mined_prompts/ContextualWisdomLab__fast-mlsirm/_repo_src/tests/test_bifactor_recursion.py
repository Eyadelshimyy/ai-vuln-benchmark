# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Two-stage Lord-Wingersky recursion vs direct enumeration.

Implementation basis: Lord, F. M., & Wingersky, M. S. (1984). Comparison of
IRT true-score and equipercentile observed-score "equatings." *Applied
Psychological Measurement, 8*(4), 453–461.
https://doi.org/10.1177/014662168400800409
"""

import numpy as np
import pytest

from fast_mlsirm.bifactor_recursion import (
    bifactor_lord_wingersky,
    direct_enumeration_bifactor,
    enumerate_bifactor_direct,
    enumerate_bifactor_lord_wingersky,
)


def test_bifactor_lord_wingersky_matches_direct_enumeration() -> None:
    """Lord-Wingersky 2-stage recursion must match direct enumeration within 1e-12.

    Evaluated on a 257-point general grid spanning [-8.0, 8.0] with step 0.0625,
    using a 4-item, 2-domain, 3-category configuration.
    """
    n_items = 4
    n_cat = 3
    n_domains = 2

    a_general = np.array([1.2, 0.9, 1.5, 1.1], dtype=np.float64)
    a_specific = np.array([0.8, 1.0, 0.7, 1.2], dtype=np.float64)
    # Thresholds: (n_cat - 1) = 2 per item, strictly ordered
    thresholds = np.array([
        [-0.6, 0.4],
        [-0.8, 0.2],
        [-0.4, 0.7],
        [-1.0, 0.1],
    ], dtype=np.float64)
    item_domains = np.array([0, 0, 1, 1], dtype=np.int64)

    # 257-point grid from -8.0 to 8.0 with step 0.0625
    theta_general = np.linspace(-8.0, 8.0, 257)

    # Specific factor quadrature: 21 standard normal nodes and weights
    theta_specific = np.linspace(-4.0, 4.0, 21)
    weights = np.exp(-0.5 * theta_specific**2)
    weights_specific = weights / np.sum(weights)

    res_lw = enumerate_bifactor_lord_wingersky(
        a_general=a_general,
        a_specific=a_specific,
        thresholds=thresholds,
        item_domains=item_domains,
        n_cat=n_cat,
        n_domains=n_domains,
        theta_general=theta_general,
        theta_specific=theta_specific,
        weights_specific=weights_specific,
    )

    res_direct = enumerate_bifactor_direct(
        a_general=a_general,
        a_specific=a_specific,
        thresholds=thresholds,
        item_domains=item_domains,
        n_cat=n_cat,
        n_domains=n_domains,
        theta_general=theta_general,
        theta_specific=theta_specific,
        weights_specific=weights_specific,
    )

    total_max_score = n_items * (n_cat - 1)
    assert res_lw.shape == (257, total_max_score + 1)
    assert res_direct.shape == (257, total_max_score + 1)

    # All probabilities must sum to 1.0 across total score distribution
    sums_lw = np.sum(res_lw, axis=1)
    np.testing.assert_allclose(sums_lw, 1.0, atol=1e-12)

    # Maximum difference across entire (257, 9) grid must be <= 1e-12
    max_diff = np.max(np.abs(res_lw - res_direct))
    assert max_diff <= 1e-12, f"Lord-Wingersky max diff {max_diff} exceeded 1e-12"


def test_deprecated_recursion_aliases_still_work_and_warn() -> None:
    """ADR-0028 (#1963): pre-rename names are kept as deprecated aliases."""
    a_general = np.array([1.2, 0.9], dtype=np.float64)
    a_specific = np.array([0.8, 1.0], dtype=np.float64)
    thresholds = np.array([[-0.6, 0.4], [-0.8, 0.2]], dtype=np.float64)
    item_domains = np.array([0, 0], dtype=np.int64)
    theta_general = np.linspace(-4.0, 4.0, 9)
    theta_specific = np.linspace(-4.0, 4.0, 5)
    weights = np.exp(-0.5 * theta_specific**2)
    weights_specific = weights / np.sum(weights)

    kwargs = dict(
        a_general=a_general,
        a_specific=a_specific,
        thresholds=thresholds,
        item_domains=item_domains,
        n_cat=3,
        n_domains=1,
        theta_general=theta_general,
        theta_specific=theta_specific,
        weights_specific=weights_specific,
    )

    with pytest.warns(DeprecationWarning):
        old_lw = bifactor_lord_wingersky(**kwargs)
    new_lw = enumerate_bifactor_lord_wingersky(**kwargs)
    np.testing.assert_array_equal(old_lw, new_lw)

    with pytest.warns(DeprecationWarning):
        old_direct = direct_enumeration_bifactor(**kwargs)
    new_direct = enumerate_bifactor_direct(**kwargs)
    np.testing.assert_array_equal(old_direct, new_direct)
