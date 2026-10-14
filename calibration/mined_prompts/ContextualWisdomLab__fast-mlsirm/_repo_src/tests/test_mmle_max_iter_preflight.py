"""Fail-first contracts for NumPy MMLE iteration-count validation."""

from __future__ import annotations

import pytest

from fast_mlsirm.estimators.mmle import fit_mmle_2pl


class _ResponseCoercionBomb:
    """Array-like sentinel proving configuration validation precedes responses."""

    def __array__(self, dtype=None, copy=None):
        """Fail if MMLE tries to inspect or materialize response data."""
        del dtype, copy
        raise AssertionError("response coercion occurred before max_iter validation")


@pytest.mark.parametrize("bad_max_iter", [0, -1, 1.5, True])
def test_invalid_max_iter_is_rejected_before_response_coercion(bad_max_iter) -> None:
    """Iteration counts must be positive integers before caller arrays are touched."""
    with pytest.raises(ValueError, match="max_iter"):
        fit_mmle_2pl(
            _ResponseCoercionBomb(),
            _ResponseCoercionBomb(),
            max_iter=bad_max_iter, n_nodes=41, tol=1e-6, seed=1
        )
