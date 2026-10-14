"""Compatibility regressions for concrete NumPy integer response evidence."""

from __future__ import annotations

import numpy as np

from fast_mlsirm.irt_contract import fit_irt_experiment, validate_irt_response_matrix


def _ready_binary_responses() -> np.ndarray:
    """Return a readiness-valid dichotomous response matrix."""
    return np.array(
        [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
        dtype=float,
    )


def test_longlong_response_cells_remain_supported() -> None:
    """Concrete NumPy longlong aliases must retain numeric response semantics."""
    responses = [
        [np.longlong(0), np.ulonglong(1)],
        [np.ulonglong(1), np.longlong(0)],
    ]

    matrix = validate_irt_response_matrix(responses, "dichotomous")

    assert matrix.dtype == np.float64
    assert matrix.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_longlong_mask_cells_remain_supported() -> None:
    """Concrete NumPy longlong aliases must retain numeric mask semantics."""
    mask = [
        [np.longlong(1), np.ulonglong(1)],
        [np.ulonglong(1), np.longlong(1)],
        [np.longlong(1), np.ulonglong(1)],
        [np.ulonglong(1), np.longlong(1)],
        [np.longlong(0), np.ulonglong(1)],
    ]

    result = fit_irt_experiment(
        lambda matrix, **_kwargs: matrix,
        _ready_binary_responses(),
        "dichotomous",
        factor_ids=(0, 0),
        mask=mask,
    )

    assert np.isnan(result[4, 0])
    assert result[4, 1] == 1.0
