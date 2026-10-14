"""Fail-first contracts for subgroup evidence in automated-scoring validation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.validation import validate_judge


def test_requested_singleton_subgroup_fails_closed() -> None:
    """A requested subgroup with one paired case is insufficient evidence, not PASS."""
    human = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    judge = human.copy()
    subgroup = np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="subgroup evidence is insufficient"):
        validate_judge(judge, human, k=2, subgroup=subgroup)


def test_requested_zero_variance_subgroup_fails_closed() -> None:
    """Undefined subgroup SMD must remain insufficient rather than being skipped."""
    human = np.array([0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int64)
    judge = human.copy()
    subgroup = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="subgroup evidence is insufficient"):
        validate_judge(judge, human, k=2, subgroup=subgroup)
