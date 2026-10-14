"""Regression contracts for mixed-format response admission."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import fit_mixed_items


def test_mixed_fit_rejects_complex_responses_before_real_narrowing() -> None:
    """Complex observed responses fail before imaginary evidence can be discarded."""
    responses = np.array(
        [
            [0.0 + 0.0j, 1.0 + 0.0j],
            [1.0 + 1.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="responses must be real-valued"):
        fit_mixed_items(
            responses,
            ["2pl", "2pl"],
            [2, 2],
            max_iter=1, q_theta=21, q_xi=7, tol=1e-5
        )
