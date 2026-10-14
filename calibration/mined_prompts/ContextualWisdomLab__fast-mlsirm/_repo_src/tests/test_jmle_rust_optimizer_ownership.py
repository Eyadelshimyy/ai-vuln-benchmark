"""Fail-first production ownership contract for JMLE optimization.

The public JMLE path may validate, initialize, marshal, and report in Python,
but Adam/L-BFGS optimizer state updates and convergence control are production
psychometric numerical work and must not execute through the legacy Python
optimizer loops.
"""

from __future__ import annotations

import importlib

import pytest

from fast_mlsirm import FitConfig, MLS2PLMConfig, simulate


fit_module = importlib.import_module("fast_mlsirm.fit")


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs", "adam_lbfgs"])
def test_public_jmle_optimization_does_not_execute_python_optimizer_loops(
    monkeypatch: pytest.MonkeyPatch,
    optimizer: str,
) -> None:
    """Installed Rust-backed JMLE must not delegate optimizer arithmetic to Python."""
    pytest.importorskip("fast_mlsirm._core")
    data = simulate(
        MLS2PLMConfig(
            n_persons=12,
            n_dims=1,
            items_per_dim=3,
            latent_dim=1,
            seed=626,
        )
    )
    python_optimizer_calls: list[str] = []

    def forbidden_python_adam(*_args: object, **_kwargs: object) -> object:
        python_optimizer_calls.append("adam")
        raise AssertionError("public JMLE executed Python Adam arithmetic")

    def forbidden_python_lbfgs(*_args: object, **_kwargs: object) -> object:
        python_optimizer_calls.append("lbfgs")
        raise AssertionError("public JMLE executed Python L-BFGS arithmetic")

    monkeypatch.setattr(fit_module, "_adam", forbidden_python_adam)
    monkeypatch.setattr(fit_module, "_lbfgs", forbidden_python_lbfgs)

    result = fit_module.fit(
        data.Y,
        data.factor_id,
        config=FitConfig(
            model="MLS2PLM",
            estimator="jmle",
            optimizer=optimizer,
            max_iter=2,
            n_restarts=1,
            seed=626,
            backend="rust",
        ),
    )

    assert python_optimizer_calls == []
    assert result.backend == "rust"
    assert result.optimizer == optimizer
