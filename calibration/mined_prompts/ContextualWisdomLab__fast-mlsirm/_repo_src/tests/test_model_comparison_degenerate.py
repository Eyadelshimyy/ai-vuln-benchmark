"""Regression tests for compiled zero-variance Vuong rejections."""

from __future__ import annotations

import math

from fast_mlsirm.model_comparison import (
    ComparisonStatus,
    ModelRelation,
    compare_nonnested_models,
)


def test_constant_casewise_difference_returns_redacted_kernel_error_result():
    """Legacy exact-zero rejection is stable without parsing exception text."""
    result = compare_nonnested_models(
        [1.0, 1.5, 2.0, 2.5],
        [0.75, 1.25, 1.75, 2.25],
        2,
        2,
        model_a="distance",
        model_b="bifactor",
        relation=ModelRelation.STRICTLY_NON_NESTED,
    )

    assert result.status is ComparisonStatus.KERNEL_ERROR
    assert result.preferred_model is None
    assert math.isnan(result.raw_mean_loglik_difference)
    assert math.isnan(result.omega)
    assert not result.variance_positive
    assert math.isnan(result.raw_z)
    assert math.isnan(result.raw_p_two_sided)
    assert math.isnan(result.z)
    assert math.isnan(result.p_two_sided)
