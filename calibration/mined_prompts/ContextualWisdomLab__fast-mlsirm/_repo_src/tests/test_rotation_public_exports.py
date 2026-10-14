"""Package-root discovery contract for adaptive factor rotation."""

from __future__ import annotations

import fast_mlsirm
from fast_mlsirm.rotation import (
    RotationCriterionInfo,
    RotationSolution,
    available_rotation_criteria,
    rotate_factor_loadings,
    rotation_criterion_value_gradient,
)


def test_package_root_exports_rotation_types_and_functions() -> None:
    """Buyers discover the complete public rotation slice from one namespace."""
    assert fast_mlsirm.RotationCriterionInfo is RotationCriterionInfo
    assert fast_mlsirm.RotationSolution is RotationSolution
    assert fast_mlsirm.available_rotation_criteria is available_rotation_criteria
    assert fast_mlsirm.rotate_factor_loadings is rotate_factor_loadings
    assert (
        fast_mlsirm.rotation_criterion_value_gradient
        is rotation_criterion_value_gradient
    )
