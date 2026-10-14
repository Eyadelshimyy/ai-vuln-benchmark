"""Fail-first direct-core contract for response-time EM iteration bounds."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm._core as core
from fast_mlsirm.config import MAX_MAX_ITER


def test_direct_rt_core_rejects_iteration_count_above_package_ceiling() -> None:
    """Direct PyO3 callers must not bypass the package-wide RT EM ceiling."""
    times = np.array([2.0], dtype=np.float64)

    with pytest.raises(ValueError, match=r"max_iter.*100000"):
        core.fit_rt_lognormal(
            times,
            None,
            1,
            1,
            MAX_MAX_ITER + 1,
            1e-6,
            1e-4,
            1e-4,
            None,
        )
