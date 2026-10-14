"""Coverage-B: inference.py guards, mmle.py control flow, objective.py rust guard,
and the package version fallback."""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import types

import numpy as np
import pytest

import fast_mlsirm
import fast_mlsirm.inference as inference_mod
from fast_mlsirm import FitConfig, MLSIRMParams, PenaltyConfig
from fast_mlsirm.estimators.mmle import fit_mmle_2pl
from fast_mlsirm.inference import (
    oakes_standard_errors,
    observed_information,
    second_order_test,
    standard_errors_from_vcov,
    vcov_from_hessian,
)
from fast_mlsirm.objective import neg_loglik_and_grad


def _small_params():
    return MLSIRMParams(
        theta=np.array([[-0.6], [0.2], [0.8]]),
        alpha=np.array([0.1, -0.2]),
        b=np.array([0.0, 0.4]),
        xi=np.zeros((3, 1)),
        zeta=np.zeros((2, 1)),
        tau=-30.0,
    )


# -- inference.py ------------------------------------------------------------


def test_observed_information_rejects_non_finite_objective(monkeypatch):
    monkeypatch.setattr(
        inference_mod, "neg_loglik_and_grad", lambda *a, **k: (float("inf"), None, None)
    )
    params = _small_params()
    responses = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]])
    config = FitConfig(model="MIRT", max_iter=1)
    with pytest.raises(ValueError, match="objective must be finite"):
        observed_information(responses, np.zeros(2, dtype=int), params, config=config)


def test_second_order_test_rejects_non_square():
    with pytest.raises(ValueError, match="hessian must be a square matrix"):
        second_order_test(np.zeros((2, 3)))


def test_vcov_from_hessian_rejects_non_square():
    with pytest.raises(ValueError, match="hessian must be a square matrix"):
        vcov_from_hessian(np.zeros((2, 3)))


def test_vcov_from_hessian_falls_back_to_pseudoinverse():
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    vcov = vcov_from_hessian(singular)
    assert vcov.shape == (2, 2)
    assert np.all(np.isfinite(vcov))


def test_standard_errors_from_vcov_rejects_non_square():
    with pytest.raises(ValueError, match="vcov must be a square matrix"):
        standard_errors_from_vcov(np.zeros((2, 3)))


def _oakes_result(*, population=None, status="converged", optimizer="mmle_marginal_em/rust"):
    return types.SimpleNamespace(
        model="MLS2PLM",
        population={} if population is None else population,
        params=MLSIRMParams(
            theta=np.zeros((4, 1)),
            alpha=np.zeros(2),
            b=np.zeros(2),
            xi=np.zeros((4, 1)),
            zeta=np.zeros((2, 1)),
            tau=-2.0,
        ),
        optimizer=optimizer,
        convergence_status=status,
    )


def test_oakes_rejects_non_dict_population():
    result = _oakes_result(population=["attacker"])
    with pytest.raises(ValueError, match="population must be a dictionary"):
        oakes_standard_errors(result, np.zeros((4, 2)), np.array([0, 0]))


def _patch_core_oakes(monkeypatch):
    from fast_mlsirm import _core

    monkeypatch.setattr(
        _core,
        "oakes_standard_errors",
        lambda *a, **k: {"labels": ["b:0"], "se": [0.2], "information": [25.0]},
    )


def test_oakes_handles_multigroup_population(monkeypatch):
    _patch_core_oakes(monkeypatch)
    out = oakes_standard_errors(
        _oakes_result(), np.zeros((4, 2)), np.array([0, 0]), group_id=np.array([0, 0, 1, 1])
    )
    assert out == {"labels": ["b:0"], "se": [0.2], "information": [25.0]}


def test_oakes_handles_multilevel_population(monkeypatch):
    _patch_core_oakes(monkeypatch)
    out = oakes_standard_errors(
        _oakes_result(), np.zeros((4, 2)), np.array([0, 0]), cluster_id=np.array([0, 0, 1, 1])
    )
    assert out == {"labels": ["b:0"], "se": [0.2], "information": [25.0]}


# -- estimators/mmle.py ------------------------------------------------------


def test_fit_mmle_2pl_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="identically shaped"):
        fit_mmle_2pl(np.zeros((2, 3)), np.ones((2, 2), dtype=bool), n_nodes=41, max_iter=500, tol=1e-6, seed=1)


def test_fit_mmle_2pl_rejects_no_observations():
    with pytest.raises(ValueError, match="no observed responses"):
        fit_mmle_2pl(np.zeros((2, 3)), np.zeros((2, 3), dtype=bool), n_nodes=41, max_iter=500, tol=1e-6, seed=1)


def test_fit_mmle_2pl_reaches_max_iter_without_converging():
    rng = np.random.default_rng(3)
    y = (rng.random((40, 4)) < 0.5).astype(float)
    observed = np.ones_like(y, dtype=bool)
    out = fit_mmle_2pl(y, observed, n_nodes=11, max_iter=1, tol=1e-6, seed=1)
    assert out["status"] == "max_iter_reached"
    assert out["n_iter"] == 1


def test_fit_mmle_2pl_handles_singular_item_hessian():
    rng = np.random.default_rng(5)
    y = (rng.random((30, 3)) < 0.5).astype(float)
    observed = np.ones_like(y, dtype=bool)
    observed[:, 2] = False  # item 2 has no observations -> zero expected counts
    out = fit_mmle_2pl(y, observed, n_nodes=11, max_iter=3, ridge_a=0.0, ridge_b=0.0, tol=1e-6, seed=1)
    assert np.all(np.isfinite(out["a"][:2]))


def test_fit_mmle_2pl_newton_exhausts_inner_iterations():
    # A perfect Guttman response set drives each item's slope to the clip
    # boundary (10.0); once pinned there the inner Newton loop keeps proposing a
    # step above the boundary, so it never meets the early-exit tolerance and
    # runs all 25 iterations before falling through to the assignment.
    theta_rank = np.linspace(-4.0, 4.0, 80)
    y = (theta_rank[:, None] > np.array([-1.0, 0.0, 1.0])[None, :]).astype(float)
    observed = np.ones_like(y, dtype=bool)
    out = fit_mmle_2pl(y, observed, n_nodes=11, max_iter=30, tol=1e-6, seed=1)
    assert np.isclose(out["a"].max(), 10.0)


# -- objective.py ------------------------------------------------------------


def test_rust_objective_rejects_multitrait_uls2plm():
    params = MLSIRMParams(
        theta=np.zeros((2, 2)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((2, 2)),
        zeta=np.zeros((2, 2)),
        tau=1.0,
    )
    with pytest.raises(ValueError, match="ULS2PLM requires one trait dimension"):
        neg_loglik_and_grad(
            np.zeros((2, 2)),
            np.zeros(2, dtype=int),
            params,
            config=FitConfig(model="ULS2PLM"),
            backend="rust",
        )


# -- package version fallback ------------------------------------------------


def test_version_falls_back_when_metadata_absent(monkeypatch):
    def _raise(name):
        raise importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib_metadata, "version", _raise)
    try:
        reloaded = importlib.reload(fast_mlsirm)
        assert reloaded.__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(fast_mlsirm)
