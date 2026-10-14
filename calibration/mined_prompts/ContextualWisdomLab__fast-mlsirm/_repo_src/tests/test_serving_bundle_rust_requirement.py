"""Fail-first contract for governed serving-bundle capability integrity."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.serving as serving
from fast_mlsirm.serving import export_serving_bundle
from fast_mlsirm.types import FitResult, MLSIRMParams


def _converged_result() -> FitResult:
    """Return one tiny converged calibration result for export-boundary tests."""
    params = MLSIRMParams(
        theta=np.zeros((1, 1), dtype=np.float64),
        alpha=np.zeros(2, dtype=np.float64),
        b=np.zeros(2, dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((2, 1), dtype=np.float64),
        tau=0.0,
    )
    return FitResult(
        params=params,
        model="MLS2PLM",
        optimizer="em",
        backend="rust",
        rust_device="cpu",
        objective=1.0,
        loglik_trace=[-1.0],
        objective_trace=[],
        convergence_status="converged",
        n_iter=1,
        population=None,
    )


def test_standard_export_requires_rust_for_complete_bundle(monkeypatch) -> None:
    """Missing Rust must fail at export instead of silently dropping EAPsum tables."""
    monkeypatch.setattr(serving, "_core_module", lambda: None)

    with pytest.raises(RuntimeError, match="serving bundle export requires the compiled Rust core"):
        export_serving_bundle(
            _converged_result(),
            ["item_one", "item_two"],
            np.array([0, 0], dtype=np.int64),
        )
