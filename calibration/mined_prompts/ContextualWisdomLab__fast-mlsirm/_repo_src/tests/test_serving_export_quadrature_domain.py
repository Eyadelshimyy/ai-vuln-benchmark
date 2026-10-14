"""Serving-export quadrature-domain regressions."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import serving
from fast_mlsirm.types import FitResult, MLSIRMParams


def _result() -> FitResult:
    """Return a minimal converged two-item result for quadrature tests."""
    params = MLSIRMParams(
        theta=np.zeros((1, 2), dtype=np.float64),
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
        objective=0.0,
        loglik_trace=[0.0],
        objective_trace=[],
        convergence_status="converged",
        n_iter=1,
    )


def _core_must_not_be_discovered() -> None:
    raise AssertionError("invalid quadrature reached compiled-core discovery")


@pytest.mark.parametrize("control_name", ["q_theta", "q_xi"])
def test_export_rejects_zero_quadrature_order_before_native(
    monkeypatch, control_name
) -> None:
    """Export must not produce a bundle that its own validator will reject.

    #1929: no node-count cap; 9 used to be rejected (not in the fixed
    7/11/15/21/31/41 table) and is now accepted, so only q < 1 fails.
    """
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match=control_name):
        serving.export_serving_bundle(
            _result(),
            ["q0", "q1"],
            (0, 1),
            **{control_name: 0},
        )
