"""Fail-closed limits for the raw multilevel PyO3 boundary."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm._multilevel_core_loader import multilevel_core
from fast_mlsirm.multilevel.contracts import MAX_CONTEXT_MEMBERSHIPS


def test_raw_binding_rejects_too_many_row_offsets_before_core_work() -> None:
    """The raw extension must bound observation metadata before conversion."""
    core = multilevel_core()
    row_offsets = np.zeros(MAX_CONTEXT_MEMBERSHIPS + 2, dtype=np.uint64)

    with pytest.raises(ValueError, match="row_offsets exceeds"):
        core.weighted_contextual_effect(
            row_offsets,
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            1,
        )


def test_raw_binding_rejects_too_many_context_indices_before_conversion() -> None:
    """The raw extension must bound edge arrays before allocating usize vectors."""
    core = multilevel_core()
    edge_count = MAX_CONTEXT_MEMBERSHIPS + 1

    with pytest.raises(ValueError, match="context_indices exceeds"):
        core.weighted_contextual_effect(
            np.array([0, edge_count], dtype=np.uint64),
            np.zeros(edge_count, dtype=np.uint64),
            np.zeros(edge_count, dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            1,
        )
