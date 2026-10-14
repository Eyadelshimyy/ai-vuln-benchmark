"""Targeted coverage for ``fast_mlsirm.fit`` guard and edge branches.

Covers input-validation guards (raising the real exception types), the
core-absent NumPy MMLE fallback, the custom-penalty marginal branch, the
lbfgs-only optimizer path, the no-gradient-clip closure branch, the Rasch
pack branch, and the two hard-to-reach L-BFGS line-search branches driven by
synthetic objectives. Everything uses tiny fixed data and bounded iterations.
"""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm._core as _core
from fast_mlsirm import MLS2PLMConfig, fit, simulate
from fast_mlsirm.config import FitConfig, PenaltyConfig
from fast_mlsirm.fit import _lbfgs


def _small_data():
    """Return a tiny 12x4, two-trait MLS2PLM dataset for driver-level tests."""
    return simulate(MLS2PLMConfig(n_persons=12, n_dims=2, items_per_dim=2, latent_dim=1, seed=1))


# --- fit() top-level validation guards --------------------------------------


def test_fit_rejects_factor_id_length_mismatch():
    """factor_id must have one entry per item (line 111)."""
    data = _small_data()
    with pytest.raises(ValueError, match="factor_id length must match number of items"):
        fit(data.Y, np.zeros(3, dtype=np.int64))


def test_fit_rejects_anchors_with_cluster_structure():
    """FIPC anchors combined with a multilevel structure are unsupported (line 131)."""
    data = _small_data()
    with pytest.raises(ValueError, match="anchors with a multilevel structure are not supported yet"):
        fit(
            data.Y,
            data.factor_id,
            config=FitConfig(estimator="mmle"),
            anchors={"fixed": np.array([True, True, True, True])},
            cluster_id=np.zeros(12, dtype=np.int64),
        )


def test_fit_rejects_covariate_without_mmle():
    """Item covariates require the marginal estimator (line 133)."""
    data = _small_data()
    with pytest.raises(ValueError, match="item covariates require estimator='mmle'"):
        fit(
            data.Y,
            data.factor_id,
            config=FitConfig(estimator="jmle"),
            covariate={"w": np.zeros(4)},
        )


# --- _fit_mmle NumPy fallback (core absent) ---------------------------------


def test_fit_mmle_fails_closed_when_core_missing(monkeypatch):
    """Without the compiled 2PL MMLE kernel, MMLE fails closed (no silent NumPy path)."""
    monkeypatch.setattr(_core, "fit_mmle_2pl", None)
    rng = np.random.default_rng(0)
    y = (rng.random((30, 6)) < 0.5).astype(float)
    fid = np.zeros(6, dtype=np.int64)

    with pytest.raises(RuntimeError, match="compiled Rust core is required for MMLE"):
        fit(y, fid, FitConfig(estimator="mmle", model="ULS2PLM", max_iter=30))


# --- _fit_mmle_marginal covariate validation --------------------------------


def test_fit_marginal_rejects_covariate_wrong_size():
    """A covariate weight vector of the wrong length is rejected (line 293)."""
    data = _small_data()
    with pytest.raises(ValueError, match="covariate w must have"):
        fit(
            data.Y,
            data.factor_id,
            config=FitConfig(estimator="mmle", model="MLS2PLM"),
            covariate={"w": np.zeros(5)},
        )


def test_fit_marginal_rejects_covariate_non_finite():
    """A non-finite covariate weight vector is rejected (line 297)."""
    data = _small_data()
    w = np.zeros(4)
    w[0] = np.nan
    with pytest.raises(ValueError, match="covariate w must be finite"):
        fit(
            data.Y,
            data.factor_id,
            config=FitConfig(estimator="mmle", model="MLS2PLM"),
            covariate={"w": w},
        )


# --- _fit_mmle_marginal anchor validation -----------------------------------


def _anchor_fit(anchors):
    """Run a marginal fit driving only the anchor-validation block."""
    data = _small_data()
    return fit(
        data.Y,
        data.factor_id,
        config=FitConfig(estimator="mmle", model="MLS2PLM"),
        anchors=anchors,
    )


def test_fit_marginal_rejects_anchor_fixed_wrong_shape():
    """The anchor fixed mask must have one entry per item (line 311)."""
    with pytest.raises(ValueError, match=r"anchor_fixed must have shape"):
        _anchor_fit(
            {
                "fixed": np.array([True, True]),
                "alpha": np.zeros(4),
                "b": np.zeros(4),
                "zeta": np.zeros(8),
            }
        )


def test_fit_marginal_rejects_anchor_alpha_wrong_shape():
    """Anchor alpha/b must have one entry per item (line 313)."""
    with pytest.raises(ValueError, match="anchor alpha/b must have shape"):
        _anchor_fit(
            {
                "fixed": np.array([True, True, True, True]),
                "alpha": np.zeros(5),
                "b": np.zeros(4),
                "zeta": np.zeros(8),
            }
        )


def test_fit_marginal_rejects_anchor_zeta_wrong_size():
    """Anchor zeta must have n_items x latent_dim entries (line 315)."""
    with pytest.raises(ValueError, match="anchor zeta must have n_items x latent_dim entries"):
        _anchor_fit(
            {
                "fixed": np.array([True, True, True, True]),
                "alpha": np.zeros(4),
                "b": np.zeros(4),
                "zeta": np.zeros(7),
            }
        )


def test_fit_marginal_rejects_anchor_non_finite():
    """Non-finite anchor parameters are rejected (line 317)."""
    alpha = np.zeros(4)
    alpha[0] = np.nan
    with pytest.raises(ValueError, match="anchor alpha/b/zeta must be finite"):
        _anchor_fit(
            {
                "fixed": np.array([True, True, True, True]),
                "alpha": alpha,
                "b": np.zeros(4),
                "zeta": np.zeros(8),
            }
        )


# --- _fit_mmle_marginal custom penalty branch -------------------------------


def test_fit_marginal_uses_custom_penalty_config():
    """A non-default penalty config replaces the LSIRM priors (line 345)."""
    data = simulate(MLS2PLMConfig(n_persons=20, n_dims=1, items_per_dim=4, latent_dim=1, seed=2))
    from fast_mlsirm.reference import fit_reference

    res = fit_reference(
        data.Y,
        data.factor_id,
        config=FitConfig(
            estimator="mmle",
            model="MLS2PLM",
            latent_dim=1,
            backend="numpy",
            max_iter=2,
            penalty=PenaltyConfig(lambda_b=0.5),
        ),
    )
    assert res.optimizer == "mmle_marginal_em/numpy"


# --- JMLE optimizer / packing edge branches ---------------------------------


def test_jmle_lbfgs_only_rasch_without_gradient_clip():
    """optimizer='lbfgs' skips the Adam block (541->552), a None gradient_clip
    takes the no-clip closure branch (650->654), and a Rasch model omits the
    discrimination block from packing (667->669)."""
    data = _small_data()
    res = fit(
        data.Y,
        data.factor_id,
        config=FitConfig(
            estimator="jmle",
            model="MLSRM",
            optimizer="lbfgs",
            gradient_clip=None,
            latent_dim=1,
            max_iter=3,
            n_restarts=1,
        ),
    )
    assert res.optimizer == "lbfgs"
    assert res.model == "MLSRM"


# --- _lbfgs line-search internal branches (synthetic objectives) ------------


def test_lbfgs_reports_line_search_failure():
    """A descent direction whose objective never decreases exhausts the line
    search and reports failure (803->810)."""
    config = FitConfig()

    def objective(x):
        """Constant objective + gradient: Armijo can never be satisfied."""
        return 5.0, np.ones_like(x), -5.0

    x0 = np.array([1.0, 2.0])
    x, trace, loglik_trace, status = _lbfgs(x0, objective, config, max_iter=3)

    assert status == "line_search_failed"
    np.testing.assert_array_equal(x, x0)
    assert trace == [5.0]


def test_lbfgs_skips_curvature_update_on_zero_ys():
    """A constant gradient gives zero curvature, so the history update is
    skipped while the iterate still advances (817->826)."""
    config = FitConfig()

    def objective(x):
        """Linear objective with a constant gradient (y_delta == 0 -> ys == 0)."""
        total = float(np.sum(x))
        return total, np.array([1.0, 1.0]), -total

    x0 = np.array([5.0, 5.0])
    x, trace, loglik_trace, status = _lbfgs(x0, objective, config, max_iter=2)

    assert status == "max_iter_reached"
    assert trace[-1] < trace[0]
    np.testing.assert_allclose(x, np.array([3.0, 3.0]))
