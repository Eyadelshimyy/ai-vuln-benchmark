"""Regression coverage for complex-valued Rasch CML input admission."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.rasch_cml import andersen_lr_test, fit_rasch_cml


def test_fit_rasch_cml_rejects_complex_responses_before_lossy_cast() -> None:
    """Non-real response data must not lose imaginary components during validation."""
    responses = np.array(
        [[0.0 + 1.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]],
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="responses must be complete 0/1"):
        fit_rasch_cml(responses, max_iter=100, tol=1e-8)


def test_andersen_lr_rejects_complex_groups_before_lossy_cast() -> None:
    """Complex group labels must not be silently projected onto the real axis."""
    responses = np.array(
        [[0, 1], [1, 0], [0, 1], [1, 0]],
        dtype=np.int64,
    )
    group = np.array([0.0 + 1.0j, 0.0, 1.0, 1.0], dtype=np.complex128)

    with pytest.raises(ValueError, match="group labels must be finite non-negative integers"):
        andersen_lr_test(responses, group, max_iter=100, tol=1e-8)
