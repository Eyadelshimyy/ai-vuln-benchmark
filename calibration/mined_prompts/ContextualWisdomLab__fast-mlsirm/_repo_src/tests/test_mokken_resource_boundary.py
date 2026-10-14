"""Resource-boundary regressions for Mokken response admission."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import mokken


_MAX_CELLS = 20_000_000


def _forbid_numpy_materialization(*args: object, **kwargs: object) -> np.ndarray:
    """Fail if resource admission reaches NumPy materialization."""
    raise AssertionError("NumPy materialization executed before resource rejection")


def test_mokken_rejects_oversized_exact_array_before_numpy_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-allocation broadcast view is bounded from logical shape alone."""
    oversized = np.broadcast_to(
        np.array([[0]], dtype=np.int8),
        (_MAX_CELLS + 1, 1),
    )
    assert type(oversized) is np.ndarray
    assert oversized.size == _MAX_CELLS + 1

    monkeypatch.setattr(mokken.np, "asarray", _forbid_numpy_materialization)

    with pytest.raises(ValueError, match="responses exceed 20,000,000 logical cells"):
        mokken.mokken_analysis(oversized)


def test_mokken_rejects_oversized_exact_numpy_row_before_sequence_stacking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested exact NumPy rows are bounded before NumPy stacks the matrix."""
    oversized_row = np.broadcast_to(
        np.array([0], dtype=np.int8),
        (_MAX_CELLS + 1,),
    )
    responses = [oversized_row]

    monkeypatch.setattr(mokken.np, "asarray", _forbid_numpy_materialization)

    with pytest.raises(ValueError, match="responses exceed 20,000,000 logical cells"):
        mokken.mokken_analysis(responses)
