import numpy as np
import pytest

from fast_mlsirm import FitConfig, MLS2PLMConfig, recovery_report, simulate
from fast_mlsirm.fit import fit


def test_fit_pipeline_smoke():
    data = simulate(MLS2PLMConfig(n_persons=30, n_dims=2, items_per_dim=3, latent_dim=2, seed=11))
    result = fit(
        data.Y,
        data.factor_id,
        config=FitConfig(model="MLS2PLM", optimizer="adam", max_iter=3, n_restarts=1, seed=11),
    )

    assert result.params.theta.shape == (30, 2)
    assert result.params.alpha.shape == (6,)
    assert result.params.xi.shape == (30, 2)
    # Default backend is "auto": production execution requires the compiled
    # Rust numerical core rather than silently changing numerical owner.
    assert result.backend == "rust"
    assert np.isfinite(result.objective)

    report = recovery_report(data.truth, result.params)
    assert "distance_rmse" in report.summary


def test_mirt_fit_sets_gamma_near_zero():
    data = simulate(MLS2PLMConfig(n_persons=20, n_dims=2, items_per_dim=2, gamma=0.0, seed=17))
    result = fit(data.Y, data.factor_id, config=FitConfig(model="MIRT", optimizer="adam", max_iter=2, n_restarts=1))
    assert result.params.gamma < 1e-10


def test_auto_backend_requires_rust_core(monkeypatch):
    data = simulate(MLS2PLMConfig(n_persons=12, n_dims=1, items_per_dim=2, latent_dim=1, seed=23))
    monkeypatch.setattr("fast_mlsirm.backend._load_core", lambda: None)

    with pytest.raises(RuntimeError, match="compiled Rust core is required for automatic backend resolution"):
        fit(
            data.Y,
            data.factor_id,
            config=FitConfig(model="MLS2PLM", optimizer="adam", max_iter=1, n_restarts=1, backend="auto"),
        )


def test_rust_backend_fit_smoke():
    pytest.importorskip("fast_mlsirm._core")
    data = simulate(MLS2PLMConfig(n_persons=12, n_dims=1, items_per_dim=2, latent_dim=1, seed=31))

    result = fit(
        data.Y,
        data.factor_id,
        config=FitConfig(model="MLS2PLM", optimizer="adam", max_iter=1, n_restarts=1, backend="rust"),
    )

    assert result.backend == "rust"
    assert np.isfinite(result.objective)


def test_rust_backend_requires_core_extension(monkeypatch):
    data = simulate(MLS2PLMConfig(n_persons=12, n_dims=1, items_per_dim=2, latent_dim=1, seed=29))
    monkeypatch.setattr("fast_mlsirm.backend._load_core", lambda: None)

    with pytest.raises(RuntimeError, match="Rust backend requested but fast_mlsirm._core is unavailable"):
        fit(
            data.Y,
            data.factor_id,
            config=FitConfig(model="MLS2PLM", optimizer="adam", max_iter=1, n_restarts=1, backend="rust"),
        )
