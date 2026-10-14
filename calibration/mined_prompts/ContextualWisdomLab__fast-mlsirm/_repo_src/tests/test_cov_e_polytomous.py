"""Coverage: validation guards and core-absent branches of ``polytomous``.

These exercise the input-validation raises and the ``_core_module() is None``
fail-closed paths that the recovery-oriented suites skip. The core-absent
paths are reached with ``monkeypatch`` so the branch runs even though the
compiled Rust core is present in this environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import polytomous
from fast_mlsirm.polytomous import (
    PolyLsirmFit,
    PolytomousFit,
    cat_simulate_polytomous,
    dif_polytomous,
    fit_lsirm_polytomous,
    fit_nominal_polytomous,
    fit_polytomous,
    information_polytomous,
    item_fit_polytomous,
    local_dependence_polytomous,
    m2_polytomous,
    person_fit_polytomous,
    polytomous_information_criteria,
    score_polytomous,
    u3_cutoff_polytomous,
    u3_person_fit_polytomous,
)


def _grm_fit(n_items: int = 2, n_cat: int = 3, converged: bool = True) -> PolytomousFit:
    """Build a deterministic well-formed GRM fit for guard/core-absent tests."""
    slope = np.linspace(0.8, 1.3, n_items)
    cat_params = np.tile(np.linspace(1.0, -1.0, n_cat - 1), (n_items, 1))
    return PolytomousFit(
        model="grm",
        slope=slope,
        cat_params=cat_params,
        loglik=-10.0,
        n_iter=4,
        converged=converged,
        termination_reason="tolerance" if converged else "max_iter",
    )


def _responses(n_persons: int = 6, n_items: int = 2, n_cat: int = 3) -> np.ndarray:
    """Deterministic integer polytomous responses in ``0..n_cat-1``."""
    rng = np.random.default_rng(0)
    return rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)


# --- _poly_int_and_mask guards (reached through u3_person_fit_polytomous) ---


def test_poly_int_and_mask_rejects_bad_n_cat():
    with pytest.raises(ValueError, match="n_cat must be an integer between"):
        u3_person_fit_polytomous(np.array([[0.0, 1.0]]), n_cat=1)


def test_poly_int_and_mask_rejects_non_2d_responses():
    with pytest.raises(ValueError, match="2-D persons x items"):
        u3_person_fit_polytomous(np.array([0.0, 1.0, 2.0]), n_cat=3)


# --- _nonnegative_integer_vector guards (reached through dif_polytomous) ---


def test_dif_rejects_2d_group_id():
    with pytest.raises(ValueError, match="group_id must be a non-empty 1-D array"):
        dif_polytomous(_responses(4, 3, 2), np.zeros((4, 1)), n_cat=2, model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05)


def test_dif_rejects_non_numeric_group_id():
    with pytest.raises(ValueError, match="group_id must contain non-negative integers"):
        dif_polytomous(_responses(4, 3, 2), np.array(["a", "b", "c", "d"]), n_cat=2, model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05)


# --- fit_polytomous ---


def test_fit_polytomous_rejects_bad_q_theta():
    # #1929: no node-count cap; q_theta=12 is now accepted, only < 1 is not.
    with pytest.raises(ValueError, match="q_theta must be in 1"):
        fit_polytomous(_responses(4, 2, 3), n_cat=3, model="grm", q_theta=0, max_iter=80, tol=1e-6)


def test_fit_polytomous_accepts_81_node_rule():
    fit = fit_polytomous(_responses(6, 2, 3), n_cat=3, model="grm", q_theta=81, max_iter=1, tol=1e-6)
    assert fit.slope.shape == (2,)
    assert np.isfinite(fit.loglik)


def test_fit_lsirm_accepts_81_node_xi_rule():
    # #1929: no node-count cap; 81 used to be rejected (outside the fixed
    # xi table) and is now a perfectly valid node count.
    fit = fit_lsirm_polytomous(
        _responses(4, 2, 3), n_cat=3, model="grm", q_theta=81, q_xi=81, max_iter=1, tol=1e-5
    )
    assert np.isfinite(fit.loglik)


def test_fit_polytomous_requires_core(monkeypatch):
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        fit_polytomous(_responses(5, 2, 2), n_cat=2, model="grm", q_theta=21, max_iter=80, tol=1e-6)


# --- score_polytomous ---


def test_score_polytomous_rejects_bad_slope():
    fit = _grm_fit()
    fit.slope = np.zeros((2, 2))
    with pytest.raises(ValueError, match=r"fit\.slope must be a non-empty 1-D array"):
        score_polytomous(_responses(), fit, q_theta=21)


def test_score_polytomous_rejects_bad_cat_params():
    fit = _grm_fit()
    fit.cat_params = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match=r"fit\.cat_params must be"):
        score_polytomous(_responses(), fit, q_theta=21)


def test_score_polytomous_rejects_bad_model():
    fit = _grm_fit()
    fit.model = "nope"
    with pytest.raises(ValueError, match=r"fit\.model must be one of"):
        score_polytomous(_responses(), fit, q_theta=21)


def test_score_polytomous_rejects_column_mismatch():
    fit = _grm_fit(n_items=2, n_cat=3)
    with pytest.raises(ValueError, match="column count must match"):
        score_polytomous(_responses(6, 3, 3), fit, q_theta=21)


def test_score_polytomous_requires_core(monkeypatch):
    fit = _grm_fit(n_items=2, n_cat=3)
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        score_polytomous(_responses(6, 2, 3), fit, q_theta=21)


# --- information_polytomous ---


def test_information_polytomous_rejects_bad_model():
    fit = _grm_fit()
    fit.model = "nope"
    with pytest.raises(ValueError, match=r"fit\.model must be one of"):
        information_polytomous(fit, np.linspace(-2, 2, 5))


def test_information_polytomous_requires_core(monkeypatch):
    fit = _grm_fit()
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        information_polytomous(fit, np.linspace(-2, 2, 5))


# --- fit_lsirm_polytomous ---


def test_fit_lsirm_rejects_bad_model():
    with pytest.raises(ValueError, match="model must be one of"):
        fit_lsirm_polytomous(_responses(4, 2, 3), n_cat=3, model="nope", q_theta=11, q_xi=11, max_iter=60, tol=1e-5)


def test_fit_lsirm_rejects_bad_latent_dim():
    with pytest.raises(ValueError, match=r"latent_dim must be an integer in 1\.\.3"):
        fit_lsirm_polytomous(_responses(4, 2, 3), n_cat=3, latent_dim=4, model="grm", q_theta=11, q_xi=11, max_iter=60, tol=1e-5)


def test_fit_lsirm_rejects_bad_quadrature():
    # #1929: no node-count cap; q_xi=12 is now accepted, only < 1 is not.
    with pytest.raises(ValueError, match="q_theta and q_xi must be >= 1"):
        fit_lsirm_polytomous(_responses(4, 2, 3), n_cat=3, model="grm", q_theta=11, q_xi=0, max_iter=60, tol=1e-5)


def test_fit_lsirm_rejects_bad_tol():
    with pytest.raises(ValueError, match="tol must be finite"):
        fit_lsirm_polytomous(_responses(4, 2, 3), n_cat=3, model="grm", q_theta=11, q_xi=11, max_iter=60, tol=0.0)


def test_fit_lsirm_requires_core(monkeypatch):
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        fit_lsirm_polytomous(_responses(5, 2, 3), n_cat=3, model="grm", q_theta=11, q_xi=11, max_iter=60, tol=1e-5)


# --- polytomous_information_criteria: latent-space parameter count ---


def test_information_criteria_counts_latent_positions():
    fit = PolyLsirmFit(
        model="grm",
        slope=np.array([1.0]),
        cat_params=np.array([[1.0, -1.0]]),
        zeta=np.zeros((1, 2)),
        theta_eap=np.zeros(3),
        theta_sd=np.ones(3),
        xi_eap=np.zeros((3, 2)),
        loglik=-5.0,
        n_iter=3,
    )
    ic = polytomous_information_criteria(fit, n_persons=100)
    # slope(1) + cat_params(2) + zeta(2) = 5 free parameters
    assert ic["n_parameters"] == 5
    assert np.isfinite(ic["aicc"])


# --- item_fit_polytomous / m2 / local_dependence core-absent ---


def test_item_fit_requires_core(monkeypatch):
    fit = _grm_fit(n_items=2, n_cat=3)
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        item_fit_polytomous(_responses(6, 2, 3), fit)


def test_m2_rejects_column_mismatch():
    fit = _grm_fit(n_items=2, n_cat=3, converged=True)
    with pytest.raises(ValueError, match="column count must match"):
        m2_polytomous(_responses(6, 3, 3), fit, q_theta=21)


def test_m2_requires_core(monkeypatch):
    fit = _grm_fit(n_items=2, n_cat=3, converged=True)
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        m2_polytomous(_responses(6, 2, 3), fit, q_theta=21)


def test_local_dependence_requires_core(monkeypatch):
    fit = _grm_fit(n_items=2, n_cat=3)
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        local_dependence_polytomous(_responses(6, 2, 3), fit)


# --- fit_nominal_polytomous ---


def test_fit_nominal_rejects_bad_q_theta():
    # #1929: no node-count cap; q_theta=12 is now accepted, only < 1 is not.
    with pytest.raises(ValueError, match="q_theta must be in 1"):
        fit_nominal_polytomous(_responses(4, 2, 3), n_cat=3, q_theta=0, max_iter=200, tol=1e-6)


def test_fit_nominal_requires_core(monkeypatch):
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        fit_nominal_polytomous(_responses(5, 2, 3), n_cat=3, q_theta=21, max_iter=200, tol=1e-6)


# --- person_fit_polytomous ---


def test_person_fit_rejects_bad_prior_mean():
    fit = _grm_fit(n_items=2, n_cat=3)
    with pytest.raises(ValueError, match="prior_mean must be finite"):
        person_fit_polytomous(_responses(6, 2, 3), fit, prior_mean=np.nan)


def test_person_fit_rejects_bad_prior_sd():
    fit = _grm_fit(n_items=2, n_cat=3)
    with pytest.raises(ValueError, match="prior_sd must be finite and > 0"):
        person_fit_polytomous(_responses(6, 2, 3), fit, prior_sd=0.0)


def test_person_fit_rejects_bad_flag_threshold():
    fit = _grm_fit(n_items=2, n_cat=3)
    with pytest.raises(ValueError, match="flag_threshold must be finite"):
        person_fit_polytomous(_responses(6, 2, 3), fit, flag_threshold=np.inf)


def test_person_fit_requires_core(monkeypatch):
    fit = _grm_fit(n_items=2, n_cat=3)
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        person_fit_polytomous(_responses(6, 2, 3), fit)


# --- cat_simulate_polytomous ---


def test_cat_simulate_rejects_empty_theta():
    fit = _grm_fit(n_items=2, n_cat=3)
    with pytest.raises(ValueError, match="true_theta must be a non-empty finite"):
        cat_simulate_polytomous(np.array([]), fit)


def test_cat_simulate_rejects_min_items_over_bank():
    fit = _grm_fit(n_items=2, n_cat=3)
    with pytest.raises(ValueError, match="min_items must not exceed"):
        cat_simulate_polytomous(np.array([0.0]), fit, min_items=5, max_items=10)


def test_cat_simulate_rejects_bad_se_threshold():
    fit = _grm_fit(n_items=2, n_cat=3)
    with pytest.raises(ValueError, match="se_threshold must be finite and >= 0"):
        cat_simulate_polytomous(
            np.array([0.0]), fit, se_threshold=-1.0, min_items=1, max_items=1
        )


def test_cat_simulate_rejects_oversized_work():
    fit = _grm_fit(n_items=1000, n_cat=3)
    with pytest.raises(ValueError, match="aggregate work limit"):
        cat_simulate_polytomous(
            np.zeros(200_001), fit, min_items=1, max_items=1000
        )


def test_cat_simulate_requires_core(monkeypatch):
    fit = _grm_fit(n_items=2, n_cat=3)
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        cat_simulate_polytomous(np.array([0.0, 0.5]), fit, min_items=1, max_items=2)


# --- dif_polytomous ---


def test_dif_rejects_empty_items():
    with pytest.raises(ValueError, match="at least one person and one item"):
        dif_polytomous(np.zeros((4, 0)), np.zeros(4, dtype=int), n_cat=2, model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05)


def test_dif_rejects_single_group():
    with pytest.raises(ValueError, match="DIF requires at least two groups"):
        dif_polytomous(_responses(6, 3, 2), np.zeros(6, dtype=int), n_cat=2, model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05)


def test_dif_requires_core(monkeypatch):
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    gid = np.arange(6) % 2
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        dif_polytomous(_responses(6, 3, 2), gid, n_cat=2, model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05)


# --- u3 helpers ---


def test_u3_person_fit_requires_core(monkeypatch):
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        u3_person_fit_polytomous(_responses(6, 3, 3), n_cat=3)


def test_u3_cutoff_rejects_oversized_work():
    fit = _grm_fit(n_items=1000, n_cat=3)
    with pytest.raises(ValueError, match="aggregate work limit"):
        u3_cutoff_polytomous(fit, n_persons=1000, n_rep=1000)


def test_u3_cutoff_requires_core(monkeypatch):
    fit = _grm_fit(n_items=2, n_cat=3)
    monkeypatch.setattr(polytomous, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        u3_cutoff_polytomous(fit, n_persons=20, n_rep=10)
