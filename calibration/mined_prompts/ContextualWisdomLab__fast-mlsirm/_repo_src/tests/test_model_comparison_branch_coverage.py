"""Focused branch coverage for model-comparison normalization and guards."""

from __future__ import annotations

import math

import numpy as np
import pytest

import fast_mlsirm.model_comparison as comparison_module
from fast_mlsirm.model_comparison import (
    ComparisonStatus,
    ModelRelation,
    VuongKernelError,
    compare_nonnested_models,
)


def _kernel_result(*, omega: float = 0.4) -> dict[str, float]:
    """Return a valid low-level result with a configurable variance scale."""
    return {
        "z": 1.5,
        "p_two_sided": 0.2,
        "omega": omega,
        "mean_diff": 0.1,
    }


def test_valid_string_relation_is_normalized(monkeypatch):
    """String callers reach the successful enum-conversion branch."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _kernel_result(),
    )
    result = compare_nonnested_models(
        [0.0, 0.1],
        [0.2, 0.3],
        1,
        1,
        relation="strictly_non_nested",
    )
    assert result.relation is ModelRelation.STRICTLY_NON_NESTED
    assert result.status is ComparisonStatus.REQUIRES_DISTINGUISHABILITY_TEST


def test_nonfinite_omega_fails_closed(monkeypatch):
    """A non-finite Rust variance scale cannot produce preference inference."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _kernel_result(omega=float("nan")),
    )
    result = compare_nonnested_models(
        [0.0, 0.1],
        [0.2, 0.3],
        1,
        1,
        relation=ModelRelation.STRICTLY_NON_NESTED,
    )
    assert result.status is ComparisonStatus.VARIANCE_DEGENERATE
    assert not result.variance_positive
    assert result.preferred_model is None
    assert math.isnan(result.z)
    assert math.isnan(result.p_two_sided)


def test_numpy_integer_counts_are_retained_and_numpy_boolean_is_rejected(monkeypatch):
    """NumPy integers remain valid counts while NumPy booleans cannot become one."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _kernel_result(),
    )
    result = compare_nonnested_models(
        [0.0, 0.1],
        [0.2, 0.3],
        np.int64(2),
        np.int64(1),
        relation=ModelRelation.STRICTLY_NON_NESTED,
    )
    assert result.k_a == 2
    assert result.k_b == 1

    with pytest.raises(ValueError, match="k_a"):
        compare_nonnested_models(
            [0.0, 0.1],
            [0.2, 0.3],
            np.bool_(True),
            1,
            relation=ModelRelation.STRICTLY_NON_NESTED,
        )


def test_compiled_validation_text_is_redacted(monkeypatch):
    """Kernel error wording is neither exposed nor retained as an exception cause."""
    def invalid_kernel(*_args, **_kwargs):
        raise ValueError("some other low-level validation error")

    monkeypatch.setattr(comparison_module, "vuong_nonnested", invalid_kernel)
    with pytest.raises(VuongKernelError, match="compiled Vuong kernel rejected") as error:
        comparison_module._run_vuong(
            (0.0, 0.1),
            (0.2, 0.3),
            1,
            1,
            bic_correction=True,
        )
    assert "some other" not in str(error.value)
    assert error.value.__cause__ is None


def test_missing_compiled_core_is_consumed_fail_closed(monkeypatch):
    """A missing compiled core becomes a redacted non-decisional result."""
    def missing_core(*_args, **_kwargs):
        raise RuntimeError("vuong_nonnested requires the compiled Rust core")

    monkeypatch.setattr(comparison_module, "vuong_nonnested", missing_core)
    result = compare_nonnested_models(
        [0.0, 0.1],
        [0.2, 0.3],
        1,
        1,
        relation=ModelRelation.STRICTLY_NON_NESTED,
    )
    assert result.status is ComparisonStatus.KERNEL_ERROR
    assert math.isnan(result.raw_mean_loglik_difference)
    assert math.isnan(result.omega)
    assert math.isnan(result.raw_z)
    assert math.isnan(result.raw_p_two_sided)
    assert result.preferred_model is None
