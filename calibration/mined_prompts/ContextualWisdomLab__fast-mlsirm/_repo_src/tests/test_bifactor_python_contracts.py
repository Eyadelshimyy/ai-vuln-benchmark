"""Extension-independent mapping and pre-allocation boundary contracts."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

scoreability_module = importlib.import_module("fast_mlsirm.bifactor_scoreability")


def _compiled_mapping() -> dict[str, object]:
    """Return a compact compiled-core-shaped scoreability result."""
    return {
        "factor_item_counts": [2, 2],
        "is_strict_bifactor": True,
        "puc": 0.5,
        "ecv_ss": [0.8, 0.2],
        "ecv_sg": [0.7, 0.3],
        "ecv_gs": [0.9, 0.6],
        "item_ecv": [0.85, 0.75],
        "omega_total": [0.9, 0.8],
        "omega_hierarchical": [0.8, 0.4],
        "construct_replicability": [0.95, 0.7],
    }


def test_mapping_result_vectors_cannot_reenable_writes() -> None:
    """The extension-independent mapping adapter creates immutable buffers."""
    result = scoreability_module._result_from_mapping(_compiled_mapping())

    for vector in (
        result.ecv_ss,
        result.ecv_sg,
        result.ecv_gs,
        result.item_ecv,
        result.omega_total,
        result.omega_hierarchical,
        result.construct_replicability,
    ):
        with pytest.raises(ValueError):
            vector[0] = 0.0
        with pytest.raises(ValueError):
            vector.setflags(write=True)


def test_mapping_result_reports_compiled_field_contract_mismatch() -> None:
    """A stale compiled module fails with a diagnostic instead of ``KeyError``."""
    raw = _compiled_mapping()
    del raw["omega_hierarchical"]

    with pytest.raises(RuntimeError, match="omega_hierarchical"):
        scoreability_module._result_from_mapping(raw)


class _ExplodingFloat:
    """Sentinel proving dtype conversion did not occur before shape rejection."""

    def __float__(self) -> float:
        """Fail if an oversized object array is converted prematurely."""
        raise AssertionError("oversized matrix was converted before budget validation")


class _ExplodingArrayLike:
    """Array-like proving advertised uniqueness shape is checked before conversion."""

    shape = (3,)

    def __array__(self, dtype: object = None) -> np.ndarray:
        """Fail if NumPy conversion occurs before the shape mismatch is rejected."""
        raise AssertionError("uniquenesses were converted before shape validation")


def test_oversized_ndarray_shape_is_rejected_before_dtype_conversion() -> None:
    """The extension-independent matrix adapter bounds work before casting."""
    oversized = np.empty(
        (2, scoreability_module.MAX_BIFACTOR_FACTORS + 1),
        dtype=object,
    )
    oversized.fill(_ExplodingFloat())

    with pytest.raises(ValueError, match="factors"):
        scoreability_module._matrix(oversized, "loadings")


def test_uniqueness_shape_is_rejected_before_dtype_conversion() -> None:
    """Advertised vector length is checked before an allocating dtype cast."""
    with pytest.raises(ValueError, match="loading row count"):
        scoreability_module._uniqueness_vector(_ExplodingArrayLike(), 2)
