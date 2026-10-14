"""Regression coverage for exact delta-plot additive-count marshalling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.deltaplot import delta_plot


class _DataSentinel:
    """Fail if a rejected count permits caller-data materialization."""

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Raise on any attempted materialization."""
        raise AssertionError("data must not be materialized for invalid nr_add")


def _unexpected_core_discovery():
    """Fail if a rejected count permits compiled-core discovery."""
    raise AssertionError("compiled core must not be discovered for invalid nr_add")


@pytest.mark.parametrize(
    "nr_add",
    (
        2**53 + 1,
        int(np.iinfo(np.uintp).max) + 1,
    ),
)
def test_add_rejects_counts_that_cannot_cross_native_boundary_exactly(
    monkeypatch, nr_add
):
    """Reject lossy f64 or out-of-usize counts before data and Rust discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="nr_add must fit native usize and round-trip exactly"):
        delta_plot(
            _DataSentinel(),
            _DataSentinel(),
            alpha=0.05,
            max_iter=10,
            extreme="add",
            nr_add=nr_add,
        )
