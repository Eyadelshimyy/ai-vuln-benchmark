"""Serving-export quadrature resource-bound regressions."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import serving
from fast_mlsirm.types import FitResult, MLSIRMParams


class _HostileZeta:
    """Array provider that records any forbidden NumPy callback dispatch."""

    calls = 0

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        type(self).calls += 1
        raise AssertionError("caller zeta array callback executed")


def _result(
    *, n_items: int, latent_dim: int, list_backed_zeta: bool = False
) -> FitResult:
    """Return a converged result with the requested serving-grid dimensions."""
    zeta: object = np.zeros((n_items, latent_dim), dtype=np.float64)
    if list_backed_zeta:
        zeta = [[0.0 for _ in range(latent_dim)] for _ in range(n_items)]
    params = MLSIRMParams(
        theta=np.zeros((1, 1), dtype=np.float64),
        alpha=np.zeros(n_items, dtype=np.float64),
        b=np.zeros(n_items, dtype=np.float64),
        xi=np.zeros((1, latent_dim), dtype=np.float64),
        zeta=zeta,  # type: ignore[arg-type]
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
    raise AssertionError("oversized serving grid reached compiled-core discovery")


def test_export_rejects_oversized_latent_grid_before_native(monkeypatch) -> None:
    """The bundle's q_xi**latent_dim ceiling must apply during export too."""
    result = _result(n_items=2, latent_dim=8)
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match=r"q_xi \*\* latent_dim"):
        serving.export_serving_bundle(
            result,
            ["q0", "q1"],
            (0, 0),
            q_theta=21,
            q_xi=41,
        )


def test_export_preserves_list_backed_zeta_resource_admission(monkeypatch) -> None:
    """Trusted list-backed zeta must reach the same resource ceiling as ndarray zeta."""
    result = _result(n_items=2, latent_dim=8, list_backed_zeta=True)
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match=r"q_xi \*\* latent_dim"):
        serving.export_serving_bundle(
            result,
            ["q0", "q1"],
            (0, 0),
            q_theta=21,
            q_xi=41,
        )


def test_export_rejects_hostile_zeta_provider_before_callback(monkeypatch) -> None:
    """Resource preflight must not execute caller-controlled zeta protocols."""
    result = _result(n_items=2, latent_dim=2)
    hostile = _HostileZeta()
    type(hostile).calls = 0
    result.params.zeta = hostile  # type: ignore[assignment]
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="zeta must be a 2-D real numeric"):
        serving.export_serving_bundle(result, ["q0", "q1"], (0, 0))

    assert type(hostile).calls == 0


def test_export_rejects_oversized_scoring_table_before_native(monkeypatch) -> None:
    """The bundle's scoring-table ceiling must apply before Rust table generation."""
    result = _result(n_items=20, latent_dim=3)
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="scoring-table size"):
        serving.export_serving_bundle(
            result,
            [f"q{index}" for index in range(20)],
            tuple(0 for _ in range(20)),
            q_theta=41,
            q_xi=41,
        )
