"""Regression coverage for polytomous DIF group-label narrowing."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.polytomous as polytomous


def test_group_id_rejects_uint64_at_signed_boundary_before_narrowing() -> None:
    """2**63 must not wrap to the signed-int64 reference-group domain."""
    labels = np.array([0, 2**63], dtype=np.uint64)

    with pytest.raises(ValueError, match="group_id must contain non-negative integers"):
        polytomous._nonnegative_integer_vector(labels, "group_id")


def test_group_id_rejects_uint64_max_before_narrowing() -> None:
    """UINT64_MAX must fail closed instead of becoming -1 after int64 casting."""
    labels = np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64)

    with pytest.raises(ValueError, match="group_id must contain non-negative integers"):
        polytomous._nonnegative_integer_vector(labels, "group_id")


def test_group_id_preserves_signed_int64_max() -> None:
    """The largest valid signed-int64 label remains a supported sparse label."""
    labels = np.array([0, np.iinfo(np.int64).max], dtype=np.int64)

    validated = polytomous._nonnegative_integer_vector(labels, "group_id")

    np.testing.assert_array_equal(validated, labels)


def test_group_id_preserves_sparse_noncontiguous_labels() -> None:
    """Ordinary sparse labels remain unchanged before DIF densification."""
    labels = np.array([2, 2, 17, 17], dtype=np.int64)

    validated = polytomous._nonnegative_integer_vector(labels, "group_id")

    np.testing.assert_array_equal(validated, labels)
