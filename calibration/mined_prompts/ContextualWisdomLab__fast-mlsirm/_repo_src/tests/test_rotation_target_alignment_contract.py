"""Regression contracts for criterion-neutral theory-target alignment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_mlsirm.rotation_selection import select_rotation_criterion


def _identity_stop_kwargs() -> dict[str, object]:
    """Return settings that preserve the identity start for alignment oracles."""
    return {
        "mode": "orthogonal",
        "policy": "theory_guided",
        "n_starts": 1,
        "max_iter": 1,
        "tolerance": 1e9,
        "max_threads": 1,
    }


def test_bifactor_target_alignment_keeps_general_column_labelled():
    """Theory-target scoring cannot swap the declared general and group factors."""
    loadings = np.asarray(
        [
            [0.10, 0.80, 0.00],
            [0.10, 0.70, 0.00],
            [0.10, 0.00, 0.80],
            [0.10, 0.00, 0.70],
        ],
        dtype=np.float64,
    )
    theory_target = np.asarray(
        [
            [0.80, 0.10, 0.00],
            [0.70, 0.10, 0.00],
            [0.00, 0.10, 0.80],
            [0.00, 0.10, 0.70],
        ],
        dtype=np.float64,
    )

    result = select_rotation_criterion(
        loadings,
        ("bifactor", "bigeomin"),
        theory_target=theory_target,
        delta=0.01,
        **_identity_stop_kwargs(),
    )

    # An unconstrained all-column Hungarian assignment produces zero by swapping
    # target columns 0 and 1. The bifactor contract pins general column 0, so the
    # remaining mismatch must stay visible.
    for candidate in result.candidates:
        assert candidate.target_rmse > 0.30


def test_partial_target_assignment_minimizes_final_cellwise_rmse():
    """Unequal target coverage uses total SSE, not equal-column mean SSE."""
    # Canonicalization keeps the higher-energy first column first.
    loadings = np.asarray(
        [
            [1.00, 0.00],
            [1.00, 0.00],
            [0.30, 0.30],
            [-0.30, -0.30],
        ],
        dtype=np.float64,
    )
    theory_target = np.asarray(
        [
            [1.00, 1.00],
            [1.00, np.nan],
            [0.30, np.nan],
            [-0.30, np.nan],
        ],
        dtype=np.float64,
    )

    result = select_rotation_criterion(
        loadings,
        ("varimax", "quartimax"),
        theory_target=theory_target,
        **_identity_stop_kwargs(),
    )

    # Total specified-cell SSE selects identity assignment: target column 0 is
    # exact and the single specified cell in column 1 contributes SSE=1. Mean
    # per-column assignment incorrectly swaps columns and yields SSE=2.
    expected = math.sqrt(1.0 / 5.0)
    for candidate in result.candidates:
        assert candidate.target_rmse == pytest.approx(expected, abs=1e-12)
