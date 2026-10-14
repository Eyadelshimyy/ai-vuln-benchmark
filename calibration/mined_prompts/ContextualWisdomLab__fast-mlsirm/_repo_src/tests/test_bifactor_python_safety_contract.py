"""Adversarial public Python-boundary contracts for bifactor scoreability."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from fast_mlsirm import assess_bifactor_scoreability
from fast_mlsirm.bifactor_scoreability import (
    MAX_BIFACTOR_FACTORS,
    MAX_BIFACTOR_WORK_UNITS,
)


def _loadings() -> np.ndarray:
    """Return one standardized strict-bifactor loading matrix."""
    return np.asarray(
        [
            [0.70, 0.40, 0.00],
            [0.70, 0.30, 0.00],
            [0.70, 0.00, 0.50],
            [0.70, 0.00, 0.60],
        ],
        dtype=np.float64,
    )


def _uniquenesses() -> np.ndarray:
    """Return residual variances satisfying the standardized identity."""
    return np.asarray([0.35, 0.42, 0.26, 0.15], dtype=np.float64)


def test_result_vectors_cannot_reenable_write_access():
    """Immutable results resist assignment and NumPy write-flag reactivation."""
    result = assess_bifactor_scoreability(_loadings(), _uniquenesses(), zero_tolerance=0.0)
    for vector in (
        result.ecv_ss,
        result.ecv_sg,
        result.ecv_gs,
        result.item_ecv,
        result.omega_total,
        result.omega_hierarchical,
        result.construct_replicability,
    ):
        assert vector.flags.writeable is False
        with pytest.raises(ValueError):
            vector[0] = 0.0
        with pytest.raises(ValueError):
            vector.setflags(write=True)


def test_oversized_object_ndarray_is_rejected_before_float_conversion():
    """Shape/work preflight runs before an untrusted ndarray dtype conversion."""
    n_items = MAX_BIFACTOR_WORK_UNITS // (MAX_BIFACTOR_FACTORS**2) + 1
    loadings = np.empty((n_items, MAX_BIFACTOR_FACTORS), dtype=object)
    uniquenesses = np.ones(n_items, dtype=np.float64)
    with pytest.raises(ValueError, match="work budget"):
        assess_bifactor_scoreability(loadings, uniquenesses, zero_tolerance=0.0)


class _OversizedNestedSequence(Sequence):
    """Array-like sequence whose payload must not be materialized."""

    def __len__(self) -> int:
        """Advertise enough rows to exceed the bounded work contract."""
        return MAX_BIFACTOR_WORK_UNITS // (MAX_BIFACTOR_FACTORS**2) + 1

    def __getitem__(self, index):
        """Return one bounded-width row for shallow shape inspection."""
        if isinstance(index, slice):
            raise AssertionError("slice materialization is not allowed")
        return [0.0] * MAX_BIFACTOR_FACTORS


class _FailingNestedSequence(Sequence):
    """Sequence that raises during bounded shape inspection."""

    def __len__(self) -> int:
        """Raise an ordinary exception that the package must normalize."""
        raise RuntimeError("caller-controlled sequence failure")

    def __getitem__(self, index):
        """Provide the protocol member required by ``Sequence``."""
        del index
        raise AssertionError("sequence indexing must not be reached")


class _ArrayProtocolProvider:
    """Array-like provider without a shape attribute."""

    def __array__(self, dtype=None):
        """Return a small valid matrix through NumPy's array protocol."""
        return np.asarray([[0.60, 0.20], [0.70, 0.30]], dtype=dtype)


def test_oversized_nested_sequence_is_rejected_before_numpy_materialization():
    """Plain nested sequences cannot bypass the pre-allocation work budget."""
    loadings = _OversizedNestedSequence()
    uniquenesses = np.ones(len(loadings), dtype=np.float64)

    with pytest.raises(ValueError, match="work budget"):
        assess_bifactor_scoreability(loadings, uniquenesses, zero_tolerance=0.0)


def test_nested_sequence_inspection_normalizes_caller_exceptions():
    """Untrusted sequence errors become stable package-owned shape errors."""
    with pytest.raises(ValueError, match="2-D item-by-factor matrix"):
        assess_bifactor_scoreability(
        _FailingNestedSequence(), _uniquenesses(), zero_tolerance=0.0
    )


def test_shape_less_array_protocol_provider_uses_bounded_materialized_shape():
    """Non-sequence array protocols retain the post-materialization shape check."""
    result = assess_bifactor_scoreability(
        _ArrayProtocolProvider(), [0.60, 0.42], zero_tolerance=0.0
    )
    assert result.factor_item_counts == (2, 2)
