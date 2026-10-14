"""Resource-boundary regressions for Rasch CML response admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.rasch_cml as rasch_cml


_MAX_CELLS = 20_000_000


def _forbid_numpy_materialization(*args: object, **kwargs: object) -> np.ndarray:
    """Fail if oversized evidence reaches NumPy materialization."""
    raise AssertionError("NumPy materialization executed before resource rejection")


def test_rasch_cml_rejects_oversized_exact_array_before_float64_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-allocation broadcast matrix is bounded from logical shape alone."""
    oversized = np.broadcast_to(
        np.array([[0]], dtype=np.int8),
        (_MAX_CELLS + 1, 1),
    )
    assert type(oversized) is np.ndarray
    assert oversized.size == _MAX_CELLS + 1

    monkeypatch.setattr(rasch_cml.np, "asarray", _forbid_numpy_materialization)

    with pytest.raises(ValueError, match="responses exceed 20,000,000 logical cells"):
        rasch_cml.fit_rasch_cml(oversized, max_iter=100, tol=1e-8)


def test_rasch_cml_rejects_oversized_numpy_row_before_sequence_stacking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exact NumPy row inside a built-in matrix is bounded before stacking."""
    oversized_row = np.broadcast_to(
        np.array([0], dtype=np.int8),
        (_MAX_CELLS + 1,),
    )
    responses = [oversized_row]

    monkeypatch.setattr(rasch_cml.np, "asarray", _forbid_numpy_materialization)

    with pytest.raises(ValueError, match="responses exceed 20,000,000 logical cells"):
        rasch_cml.fit_rasch_cml(responses, max_iter=100, tol=1e-8)
