"""Coverage tests (batch C) for ``fast_mlsirm.fitstats``.

These target fit-statistics validation and the fail-closed Rust ownership
boundary, input-validation guards, and edge branches in the fit-statistics
core. They assert real behaviour (the exact exception types the guards raise,
and native structural agreement) rather than merely executing lines.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fm
from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import fit


def _pure_chi2_sf(x: float, df: float) -> float:
    import math
    if df <= 0:
        return float("nan")
    # use public pure helper still on module if present
    from fast_mlsirm import fitstats as fm
    return fm._gammainc_upper_reg(df / 2.0, max(x, 0.0) / 2.0)


def _pure_bh(p_values, q: float = 0.05):
    import numpy as np
    p = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p)
    m = int(valid.sum())
    reject = np.zeros(p.shape, dtype=bool)
    if m == 0:
        return reject.tolist() if hasattr(p_values, '__iter__') else reject
    order = np.argsort(np.where(valid, p, np.inf))
    ranked = p[order][:m]
    thresh = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    if below.any():
        k = int(np.max(np.nonzero(below)[0]))
        reject[order[: k + 1]] = True
    return reject.tolist()


class _CoreWithFitStatsOnly:
    """Stub core without the native S-X²/person-fit entrypoints."""

    def chi2_sf(self, x, df):
        return _pure_chi2_sf(float(x), float(df))

    def benjamini_hochberg(self, p_values, q):
        return _pure_bh(p_values, float(q))



# ---------------------------------------------------------------------------
# small synthetic parameter/response builders (no model fitting required)
# ---------------------------------------------------------------------------


def _mirt_params(n_items, n_persons, seed=0):
    """Build a plausible MIRT parameter carrier and matching responses."""
    rng = np.random.default_rng(seed)
    alpha = np.zeros(n_items)
    b = np.linspace(-1.0, 1.0, n_items)
    zeta = np.zeros((n_items, 1))
    theta = rng.standard_normal((n_persons, 1))
    xi = np.zeros((n_persons, 1))
    eta = np.exp(alpha)[None, :] * theta[:, np.zeros(n_items, dtype=int)] + b[None, :]
    y = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    params = SimpleNamespace(alpha=alpha, b=b, zeta=zeta, tau=-30.0, theta=theta, xi=xi)
    fid = np.zeros(n_items, dtype=np.int64)
    return y, fid, params


def _spatial_params(factor_id, n_persons, latent_dim=1, seed=1, model="MLS2PLM"):
    """Build a spatial (latent-space) parameter carrier and matching responses."""
    rng = np.random.default_rng(seed)
    factor_id = np.asarray(factor_id, dtype=np.int64)
    n_items = factor_id.size
    n_dims = int(factor_id.max()) + 1
    alpha = np.log(np.full(n_items, 1.1))
    b = np.linspace(-1.0, 1.0, n_items)
    zeta = rng.normal(0.0, 0.3, (n_items, latent_dim))
    tau = math.log(0.4)
    theta = rng.standard_normal((n_persons, n_dims))
    xi = rng.normal(0.0, 0.3, (n_persons, latent_dim))
    eta = np.exp(alpha)[None, :] * theta[:, factor_id] + b[None, :]
    dist = np.sqrt(1e-8 + ((xi[:, None, :] - zeta[None, :, :]) ** 2).sum(2))
    eta = eta - math.exp(tau) * dist
    y = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    params = SimpleNamespace(alpha=alpha, b=b, zeta=zeta, tau=tau, theta=theta, xi=xi)
    return y, factor_id, params


# ---------------------------------------------------------------------------
# S-X2 control and input guards (run before the native core)
# ---------------------------------------------------------------------------


def test_sx2_control_and_input_guards():
    y, fid, params = _mirt_params(4, 6, seed=2)

    with pytest.raises(ValueError, match="min_expected must be a finite number"):
        fm.s_x2(y, fid, params, "MIRT", min_expected="nope")
    with pytest.raises(ValueError, match=r"fdr_q must be in \(0, 1\]"):
        fm.s_x2(y, fid, params, "MIRT", fdr_q=1.5)

    with pytest.raises(ValueError, match="2-D numeric array"):
        fm.s_x2([["a", "b"], ["c", "d"]], fid, params, "MIRT")
    with pytest.raises(ValueError, match="2-D persons x items"):
        fm.s_x2(np.zeros(4), fid, params, "MIRT")
    with pytest.raises(ValueError, match="mask shape must match"):
        fm.s_x2(y, fid, params, "MIRT", mask=np.ones((2, 2), dtype=bool))
    nonbinary = y.copy()
    nonbinary[0, 0] = 2.0
    with pytest.raises(ValueError, match="dichotomous"):
        fm.s_x2(nonbinary, fid, params, "MIRT")
    with pytest.raises(ValueError, match="person_weight must have length"):
        fm.s_x2(y, fid, params, "MIRT", person_weight=np.ones(y.shape[0] - 1))


def test_validate_factor_id_and_prepare_guards():
    with pytest.raises(ValueError, match="1-D array"):
        fm._validate_factor_id(np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="finite non-negative integers"):
        fm._validate_factor_id(np.array([0.0, 1.0]))

    params = SimpleNamespace(
        alpha=np.zeros(4), b=np.zeros(4), zeta=np.zeros((4, 1)), tau=-30.0,
        theta=np.zeros((3, 1)), xi=np.zeros((3, 1)),
    )
    with pytest.raises(ValueError, match="responses must be a 2-D array"):
        fm.person_fit(np.zeros(4), np.zeros(4, dtype=np.int64), params, "MIRT")
    with pytest.raises(ValueError, match="at least one item"):
        fm.person_fit(np.zeros((3, 0)), np.zeros(0, dtype=np.int64), params, "MIRT")
    with pytest.raises(ValueError, match="mask shape must match"):
        fm.person_fit(
            np.zeros((3, 4)), np.zeros(4, dtype=np.int64), params, "MIRT",
            mask=np.ones((2, 4), dtype=bool),
        )


# ---------------------------------------------------------------------------
# incomplete-gamma / chi-square helpers
# ---------------------------------------------------------------------------


def test_gammainc_and_chi2_edges():
    with pytest.raises(ValueError, match="invalid arguments"):
        fm._gammainc_upper_reg(1.0, -1.0)
    with pytest.raises(ValueError, match="invalid arguments"):
        fm._gammainc_upper_reg(0.0, 1.0)
    assert math.isnan(fm.chi2_sf(1.0, 0.0))
    assert math.isnan(fm.chi2_sf(1.0, -3.0))
    # continued-fraction regime (x >= a + 1)
    assert fm.chi2_sf(20.0, 2.0) == pytest.approx(4.5399929e-05, abs=1e-9)


# ---------------------------------------------------------------------------
# ICC grid + factorized-moment guards
# ---------------------------------------------------------------------------


def test_icc_grid_and_factorized_guards():
    _y, fid, params = _spatial_params([0, 0, 1, 1], 20, seed=3)
    probs, t_w, x_w, _ = fm._icc_grid(params, fid, "MLS2PLM", q_theta=7, q_xi=7)
    assert probs.shape[0] == 4 and np.all(np.isfinite(probs))

    with pytest.raises(ValueError, match="finite vectors"):
        fm._icc_grid(params, fid, "MLS2PLM", prior_mean=np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="positive values"):
        fm._icc_grid(params, fid, "MLS2PLM", prior_sd=np.array([0.0, 1.0]))

    with pytest.raises(ValueError, match="does not match quadrature weights"):
        fm._factorized_trait_moments(
            probs, t_w[:-1], x_w, fid, [[0, 1]]
        )


# ---------------------------------------------------------------------------
# S-X2 native ownership (incomplete core fails closed)
# ---------------------------------------------------------------------------


def test_sx2_numpy_fallback_realistic(monkeypatch):
    """Incomplete core must fail closed; public S-X² no longer uses NumPy fallback."""
    rng = np.random.default_rng(7)
    n_items, n_persons = 6, 40
    params = SimpleNamespace(
        alpha=np.zeros(n_items),
        b=np.linspace(-1.0, 1.0, n_items),
        zeta=np.zeros((n_items, 1)),
        tau=-30.0,
    )
    fid = np.zeros(n_items, dtype=np.int64)
    y = (rng.random((n_persons, n_items)) < 0.5).astype(float)
    monkeypatch.setattr(fm, "_core_module", lambda: _CoreWithFitStatsOnly())
    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        fm.s_x2(y, fid, params, "MIRT", min_expected=1.0)


def test_sx2_numpy_fallback_spatial_and_dim_floors(monkeypatch):
    """Spatial S-X² with incomplete core fails closed before Python numerics."""
    y, fid, params = _spatial_params([0, 0, 0, 1, 2, 2], 30, seed=4)
    monkeypatch.setattr(fm, "_core_module", lambda: _CoreWithFitStatsOnly())
    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        fm.s_x2(y, fid, params, "MLS2PLM", min_expected=1.0)


# ---------------------------------------------------------------------------
# person-fit Rust ownership (incomplete core fails closed)
# ---------------------------------------------------------------------------


def test_person_fit_numpy_fallback(monkeypatch):
    """Incomplete core must fail closed; public person-fit no longer uses NumPy fallback."""
    y, fid, params = _mirt_params(6, 25, seed=8)
    monkeypatch.setattr(fm, "_core_module", lambda: _CoreWithFitStatsOnly())
    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        fm.person_fit(
            y, fid, params, "MIRT", prior_mean=np.full((y.shape[0], 1), 0.2)
        )


def test_infit_outfit_numpy_fallback(monkeypatch):
    """Incomplete core must fail closed for public infit/outfit."""
    y, fid, params = _mirt_params(6, 25, seed=10)
    monkeypatch.setattr(fm, "_core_module", lambda: _CoreWithFitStatsOnly())
    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        fm.infit_outfit(y, fid, params, "MIRT")




# ---------------------------------------------------------------------------
# select_items guards, per-dimension floor, and final refit
# ---------------------------------------------------------------------------


def test_select_items_argument_guards():
    y, fid, _ = _mirt_params(6, 40, seed=12)
    with pytest.raises(ValueError, match="requires estimator='mmle'"):
        fm.select_items(y, fid, config=FitConfig(model="MIRT", estimator="jmle"))
    with pytest.raises(ValueError, match="max_rounds must be >= 1"):
        fm.select_items(
            y, fid, config=FitConfig(model="MIRT", estimator="mmle"), max_rounds=0
        )


def test_select_items_floor_and_final_refit():
    rng = np.random.default_rng(13)
    n_persons, n_items = 220, 6
    a = 0.9 + 0.6 * rng.random(n_items)
    b = np.linspace(-1.0, 1.0, n_items)
    theta = rng.standard_normal(n_persons)
    eta = a[None, :] * theta[:, None] + b[None, :]
    y = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    # two near-constant (sparse) items
    y[:, 4] = 0.0
    y[:3, 4] = 1.0
    y[:, 5] = 0.0
    y[:4, 5] = 1.0
    codes = [f"col_{i:02d}" for i in range(n_items)]
    out = fm.select_items(
        y,
        np.zeros(n_items, dtype=np.int64),
        item_codes=codes,
        config=FitConfig(
            model="MIRT", estimator="mmle", max_iter=120, tolerance=1e-3,
            q_theta=11, latent_dim=1,
        ),
        max_rounds=1,
        min_items_per_dim=5,
    )
    # the per-dimension floor keeps at least five items despite two sparse flags.
    assert len(out.kept_items) >= 5
    assert out.final_result.convergence_status == "converged"
    assert len(out.final_result.params.b) == len(out.kept_items)


def _screening_responses(seed, n_persons, n_items, offsets=None):
    """Build a well-behaved 2PL response matrix for screening tests."""
    rng = np.random.default_rng(seed)
    a = 0.9 + 0.5 * rng.random(n_items)
    b = np.linspace(-1.0, 1.0, n_items)
    theta = offsets if offsets is not None else rng.standard_normal(n_persons)
    eta = a[None, :] * theta[:, None] + b[None, :]
    return (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)


def test_select_items_multigroup_prior_centering():
    y = _screening_responses(1, 300, 6)
    gid = np.tile([0, 1], 150)
    out = fm.select_items(
        y,
        np.zeros(6, dtype=np.int64),
        config=FitConfig(
            model="MIRT", estimator="mmle", max_iter=120, tolerance=1e-3,
            q_theta=11, latent_dim=1,
        ),
        group_id=gid,
        max_rounds=1,
        min_items_per_dim=4,
    )
    assert out.final_result.population["kind"] == "multigroup"
    assert len(out.kept_items) >= 4


def test_select_items_multilevel_prior_centering():
    rng = np.random.default_rng(5)
    n_persons, n_items = 300, 6
    cluster_id = np.repeat(np.arange(30), 10)
    u = rng.standard_normal(30) * 0.6
    y = _screening_responses(5, n_persons, n_items, offsets=u[cluster_id] + rng.standard_normal(n_persons))
    out = fm.select_items(
        y,
        np.zeros(n_items, dtype=np.int64),
        config=FitConfig(
            model="MIRT", estimator="mmle", max_iter=400, tolerance=1e-3,
            q_theta=11, q_u=7, latent_dim=1,
        ),
        cluster_id=cluster_id,
        max_rounds=1,
        min_items_per_dim=4,
    )
    assert out.final_result.population["kind"] == "multilevel"


def test_select_items_population_none_path():
    rng = np.random.default_rng(6)
    y = (rng.random((300, 6)) < 0.55).astype(float)
    out = fm.select_items(
        y,
        np.zeros(6, dtype=np.int64),
        config=FitConfig(
            model="ULS2PLM", estimator="mmle", max_iter=120, tolerance=1e-3,
            q_theta=11, latent_dim=1,
        ),
        max_rounds=1,
        min_items_per_dim=4,
    )
    # the legacy unidimensional 2PL path reports no population structure.
    assert out.final_result.population is None


# ---------------------------------------------------------------------------
# model-comparison wrappers: core-absent + input guards
# ---------------------------------------------------------------------------


def test_vuong_nonnested_guards(monkeypatch):
    la = np.zeros(5)
    lb = np.ones(5)
    monkeypatch.setattr(fm, "_core_module", lambda: _CoreWithFitStatsOnly())
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        fm.vuong_nonnested(la, lb, 1, 1)
    monkeypatch.undo()

    with pytest.raises(ValueError, match="one-dimensional"):
        fm.vuong_nonnested(np.zeros((2, 2)), np.zeros((2, 2)), 1, 1)
    with pytest.raises(ValueError, match="must be numeric"):
        fm.vuong_nonnested(np.array(["a", "b"]), np.zeros(2), 1, 1)
    with pytest.raises(ValueError, match="equal-length with n >= 2"):
        fm.vuong_nonnested(np.array([0.0]), np.array([0.0]), 1, 1)


def test_dimensionality_residuals_guards(monkeypatch):
    from fast_mlsirm.types import MLSIRMParams

    y, fid, _p = _mirt_params(4, 20, seed=14)
    params = MLSIRMParams(
        theta=np.zeros((20, 1)), alpha=np.zeros(4), b=np.zeros(4),
        xi=np.zeros((20, 1)), zeta=np.zeros((4, 1)), tau=-30.0,
    )
    monkeypatch.setattr(fm, "_core_module", lambda: _CoreWithFitStatsOnly())
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        fm.dimensionality_residuals(y, fid, params, "MIRT")
    monkeypatch.undo()

    with pytest.raises(ValueError, match="eps_distance must be > 0"):
        fm.dimensionality_residuals(y, fid, params, "MIRT", eps_distance=True)
    with pytest.raises(ValueError, match="eps_distance must be > 0"):
        fm.dimensionality_residuals(y, fid, params, "MIRT", eps_distance=-1.0)

    bad_theta = MLSIRMParams(
        theta=np.zeros(20), alpha=np.zeros(4), b=np.zeros(4),
        xi=np.zeros((20, 1)), zeta=np.zeros((4, 1)), tau=-30.0,
    )
    with pytest.raises(ValueError, match="theta must be a 2-D array"):
        fm.dimensionality_residuals(y, fid, bad_theta, "MIRT")

    inf_params = MLSIRMParams(
        theta=np.zeros((20, 1)), alpha=np.zeros(4), b=np.full(4, np.inf),
        xi=np.zeros((20, 1)), zeta=np.zeros((4, 1)), tau=-30.0,
    )
    with pytest.raises(ValueError, match="must be finite"):
        fm.dimensionality_residuals(y, fid, inf_params, "MIRT")


# ---------------------------------------------------------------------------
# residual-item-fit, adjusted-chi2, person-fit-resampling: guards + happy path
# ---------------------------------------------------------------------------


def test_residual_item_fit_guards_and_happy_path(monkeypatch):
    y, fid, params = _mirt_params(4, 60, seed=15)
    monkeypatch.setattr(fm, "_core_module", lambda: _CoreWithFitStatsOnly())
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        fm.residual_item_fit(y, fid, params, "MIRT", n_bins=2)
    monkeypatch.undo()

    with pytest.raises(ValueError, match="at least one person"):
        fm.residual_item_fit(np.empty((0, 4)), fid, params, "MIRT", n_bins=2)

    bad_theta = SimpleNamespace(
        alpha=params.alpha, b=params.b, zeta=params.zeta, tau=params.tau,
        theta=np.zeros((60, 2)), xi=params.xi,
    )
    with pytest.raises(ValueError, match=r"theta shape must be"):
        fm.residual_item_fit(y, fid, bad_theta, "MIRT", n_bins=2)

    bad_xi = SimpleNamespace(
        alpha=params.alpha, b=params.b, zeta=params.zeta, tau=params.tau,
        theta=params.theta, xi=np.zeros((60, 2)),
    )
    with pytest.raises(ValueError, match=r"xi shape must be"):
        fm.residual_item_fit(y, fid, bad_xi, "MIRT", n_bins=2)

    out = fm.residual_item_fit(y, fid, params, "MIRT", n_bins=2)
    assert out["max_abs_z"].shape == (4,)
    assert out["p_value"].shape == (4,)


def test_adjusted_chi2_pairs_guards(monkeypatch):
    y, fid, params = _mirt_params(3, 40, seed=16)
    monkeypatch.setattr(fm, "_core_module", lambda: _CoreWithFitStatsOnly())
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        fm.adjusted_chi2_pairs(y, fid, params, "MIRT")
    monkeypatch.undo()

    with pytest.raises(ValueError, match="at least one person"):
        fm.adjusted_chi2_pairs(np.empty((0, 3)), fid, params, "MIRT")
    one_item = SimpleNamespace(
        alpha=np.zeros(1), b=np.zeros(1), zeta=np.zeros((1, 1)), tau=-30.0,
        theta=np.zeros((40, 1)), xi=np.zeros((40, 1)),
    )
    with pytest.raises(ValueError, match="at least two items"):
        fm.adjusted_chi2_pairs(
            np.zeros((40, 1)), np.zeros(1, dtype=np.int64), one_item, "MIRT"
        )
    with pytest.raises(ValueError, match="q_theta must be an integer"):
        fm.adjusted_chi2_pairs(y, fid, params, "MIRT", q_theta=21.5)


def test_person_fit_resampling_guards_and_happy_path(monkeypatch):
    y, fid, params = _mirt_params(4, 30, seed=17)
    monkeypatch.setattr(fm, "_core_module", lambda: _CoreWithFitStatsOnly())
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        fm.person_fit_resampling(y, fid, params, "MIRT")
    monkeypatch.undo()

    big_y = np.zeros((5000, 5))
    big_params = SimpleNamespace(
        alpha=np.zeros(5), b=np.zeros(5), zeta=np.zeros((5, 1)), tau=-30.0,
        theta=np.zeros((5000, 1)), xi=np.zeros((5000, 1)),
    )
    with pytest.raises(ValueError, match="aggregate work limit"):
        fm.person_fit_resampling(
            big_y, np.zeros(5, dtype=np.int64), big_params, "MIRT", n_replicates=10000
        )

    bad_theta = SimpleNamespace(
        alpha=params.alpha, b=params.b, zeta=params.zeta, tau=params.tau,
        theta=np.zeros((30, 2)), xi=params.xi,
    )
    with pytest.raises(ValueError, match=r"theta shape must be"):
        fm.person_fit_resampling(y, fid, bad_theta, "MIRT")

    bad_xi = SimpleNamespace(
        alpha=params.alpha, b=params.b, zeta=params.zeta, tau=params.tau,
        theta=params.theta, xi=np.full((30, 1), np.nan),
    )
    with pytest.raises(ValueError, match="xi must be finite"):
        fm.person_fit_resampling(y, fid, bad_xi, "MIRT")

    with pytest.raises(ValueError, match="prior_mean must be finite"):
        fm.person_fit_resampling(
            y, fid, params, "MIRT", prior_mean=np.full((30, 1), np.nan)
        )
    with pytest.raises(ValueError, match="must broadcast"):
        fm.person_fit_resampling(y, fid, params, "MIRT", prior_mean=np.zeros(5))

    # happy path with an explicit prior mean and with the default (None) prior.
    pv = fm.person_fit_resampling(
        y, fid, params, "MIRT", prior_mean=np.zeros((30, 1)), n_replicates=8, seed=3
    )
    assert pv.shape[0] == 30
    pv_default = fm.person_fit_resampling(y, fid, params, "MIRT", n_replicates=8, seed=4)
    assert pv_default.shape[0] == 30


def test_tcc_drift_requires_core(monkeypatch):
    params = SimpleNamespace(
        alpha=np.zeros(4), b=np.zeros(4), zeta=np.zeros((4, 1)), tau=-30.0
    )
    monkeypatch.setattr(fm, "_core_module", lambda: _CoreWithFitStatsOnly())
    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        fm.tcc_drift(params, params, np.zeros(4, dtype=np.int64), "MIRT")


# ---------------------------------------------------------------------------
# noncentral chi-square helpers
# ---------------------------------------------------------------------------


def test_noncentral_chi2_edges():
    # lam <= 0 reduces to the central survival complement.
    assert fm._ncchi2_cdf(3.0, 5.0, 0.0) == pytest.approx(1.0 - fm.chi2_sf(3.0, 5.0))
    assert fm._ncchi2_cdf(3.0, 5.0, -1.0) == pytest.approx(1.0 - fm.chi2_sf(3.0, 5.0))
    # non-finite argument with positive noncentrality yields NaN.
    assert math.isnan(fm._ncchi2_cdf(float("inf"), 5.0, 1.0))
    # an extreme noncentrality exhausts the Poisson-mixture series -> NaN.
    assert math.isnan(fm._ncchi2_cdf(5e9, 50.0, 1e10))
    # a target unattainable below lambda = 1e8 returns NaN from the root search.
    assert math.isnan(fm._nc_lambda_for(1e9, 50.0, 0.5))


# ---------------------------------------------------------------------------
# _m2_numpy reference: spatial params, priors, degenerate items, and guards
# ---------------------------------------------------------------------------


def test_m2_numpy_reference_guards_and_spatial():
    # fewer than three items is rejected.
    y2, fid2, p2 = _spatial_params([0, 0], 40, seed=18)
    with pytest.raises(ValueError, match="at least 3 items"):
        fm._m2_numpy(y2, ~np.isnan(y2), fid2, p2, "MLS2PLM", 7, 7, 1e-8)

    # spatial model with explicit prior mean/sd (exercises the z/tau delta and
    # the prior-supplied branches).
    y, fid, params = _spatial_params([0] * 7, 240, seed=19)
    res = fm._m2_numpy(
        y, ~np.isnan(y), fid, params, "MLS2PLM", 7, 7, 1e-8,
        prior_mean=np.zeros(1), prior_sd=np.ones(1),
    )
    assert res.n_moments == 7 + 7 * 6 // 2
    assert np.isfinite(res.m2)

    # non-2PL spatial model (no free discrimination column).
    yr, fidr, pr = _spatial_params([0] * 7, 200, seed=20, model="MLSRM")
    res_rm = fm._m2_numpy(yr, ~np.isnan(yr), fidr, pr, "MLSRM", 7, 7, 1e-8)
    assert res_rm.n_parameters < res.n_parameters  # no per-item alpha column

    # spatial model with too few items for the parameter count.
    y3, fid3, p3 = _spatial_params([0, 0, 0], 60, seed=21)
    with pytest.raises(ValueError, match="df non-positive"):
        fm._m2_numpy(y3, ~np.isnan(y3), fid3, p3, "MLS2PLM", 7, 7, 1e-8)

    # too few complete cases for the free parameter count (MIRT).
    ym, fidm, pm = _mirt_params(4, 5, seed=22)
    with pytest.raises(ValueError, match="too few complete cases"):
        fm._m2_numpy(ym, ~np.isnan(ym), fidm, pm, "MIRT", 7, 7, 1e-8)


# ---------------------------------------------------------------------------
# public m2() validation guards + core-absent NumPy dispatch
# ---------------------------------------------------------------------------


def test_m2_public_guards_and_numpy_dispatch(monkeypatch):
    y, fid, params = _mirt_params(6, 400, seed=23)

    with pytest.raises(ValueError, match="estimator must be one of"):
        fm.m2(y, fid, params, "MIRT", estimator="bogus")
    with pytest.raises(ValueError, match="persons-by-items matrix"):
        fm.m2(np.zeros(6), fid, params, "MIRT")
    with pytest.raises(ValueError, match="mask must match"):
        fm.m2(y, fid, params, "MIRT", mask=np.ones((2, 6), dtype=bool))
    nonbinary = y.copy()
    nonbinary[0, 0] = 2.0
    with pytest.raises(ValueError, match="finite binary values"):
        fm.m2(nonbinary, fid, params, "MIRT")
    with pytest.raises(ValueError, match="factor_id length must match"):
        fm.m2(y, np.zeros(5, dtype=np.int64), params, "MIRT")
    with pytest.raises(ValueError, match=r"both have shape"):
        fm.m2(y, fid, params, "MIRT", prior_mean=np.zeros(2))
    with pytest.raises(ValueError, match="must be finite"):
        fm.m2(y, fid, params, "MIRT", prior_mean=np.array([np.nan]))
    with pytest.raises(ValueError, match="prior_sd must be positive"):
        fm.m2(y, fid, params, "MIRT", prior_sd=np.array([0.0]))

    monkeypatch.setattr(fm, "_core_module", lambda: _CoreWithFitStatsOnly())
    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        fm.m2(y, fid, params, "MIRT", q_theta=7, q_xi=7)


# ---------------------------------------------------------------------------
# CMLE Rasch M2 validation guards
# ---------------------------------------------------------------------------


def test_m2_cmle_rasch_guards():
    b5 = np.zeros(5)
    with pytest.raises(ValueError, match="persons-by-items matrix"):
        fm.m2_cmle_rasch(np.zeros(5), b5)
    with pytest.raises(ValueError, match="mask must match"):
        fm.m2_cmle_rasch(np.zeros((4, 5)), b5, mask=np.ones((2, 5), dtype=bool))
    nonbinary = np.zeros((6, 5))
    nonbinary[0, 0] = 2.0
    with pytest.raises(ValueError, match="complete binary response rows"):
        fm.m2_cmle_rasch(nonbinary, b5)
    with pytest.raises(ValueError, match="finite vector of length 5"):
        fm.m2_cmle_rasch(np.zeros((6, 5)), np.zeros(4))
    with pytest.raises(ValueError, match="at least 5 items"):
        fm.m2_cmle_rasch(np.zeros((6, 4)), np.zeros(4))
    # complete binary data missing a raw-score category (score 3 absent).
    patterns = np.array(
        [
            [0, 0, 0, 0, 0],  # 0
            [1, 0, 0, 0, 0],  # 1
            [1, 1, 0, 0, 0],  # 2
            [1, 1, 1, 1, 0],  # 4
            [1, 1, 1, 1, 1],  # 5
        ],
        dtype=float,
    )
    responses = np.repeat(patterns, 3, axis=0)
    with pytest.raises(ValueError, match="every raw-score category represented"):
        fm.m2_cmle_rasch(responses, np.linspace(-1.0, 1.0, 5))


# ---------------------------------------------------------------------------
# single-population / group-component spatial paths via public m2()
# ---------------------------------------------------------------------------


def test_m2_single_population_spatial_and_fixed_items():
    # fixed_items must be boolean-valued.
    y, fid, params = _spatial_params([0] * 7, 200, seed=24)
    bad_fixed = np.zeros(7, dtype=int)
    bad_fixed[0] = 2
    with pytest.raises(ValueError, match="only boolean values"):
        fm.m2(y, fid, params, "MLS2PLM", fixed_items=bad_fixed)

    # estimate_population adds the mean/SD nuisance columns (spatial 2PL).
    res_pop = fm.m2(
        y, fid, params, "MLS2PLM", q_theta=7, q_xi=7,
        estimate_population=True, prior_mean=np.zeros(1), prior_sd=np.ones(1),
    )
    assert np.isfinite(res_pop.m2)
    assert "estimated mean/SD" in res_pop.inference_note

    # fixed_items path without estimate_population excludes anchored columns.
    fixed = np.zeros(7, dtype=bool)
    fixed[:2] = True
    res_fixed = fm.m2(
        y, fid, params, "MLS2PLM", q_theta=7, q_xi=7, fixed_items=fixed
    )
    assert np.isfinite(res_fixed.m2)
    assert "fixed calibration columns excluded" in res_fixed.inference_note

    # non-2PL spatial model exercises the alpha-free component branch.
    yr, fidr, pr = _spatial_params([0] * 7, 200, seed=25, model="MLSRM")
    res_rm = fm.m2(
        yr, fidr, pr, "MLSRM", q_theta=7, q_xi=7,
        estimate_population=True, prior_mean=np.zeros(1), prior_sd=np.ones(1),
    )
    assert np.isfinite(res_rm.m2)


def test_dif_analysis_input_guards():
    y, fid, _params = _mirt_params(6, 40, seed=30)
    gid = np.tile([0, 1], 20)
    with pytest.raises(ValueError, match="non-empty 2D array"):
        fm.dif_analysis(np.zeros(6), fid, gid)
    with pytest.raises(ValueError, match="same shape as responses"):
        fm.dif_analysis(y, fid, gid, mask=np.ones((2, 6), dtype=bool))
    # correct-length item_codes but an invalid studied-item specification.
    codes = [f"i{i}" for i in range(6)]
    with pytest.raises(ValueError, match="one-dimensional integer sequence"):
        fm.dif_analysis(y, fid, gid, item_codes=codes, studied_items=[[0, 1]])
    with pytest.raises(ValueError, match="requires estimator='mmle'"):
        fm.dif_analysis(y, fid, gid, config=FitConfig(model="MIRT", estimator="jmle"))


def test_dif_analysis_accepts_boolean_mask():
    y = _screening_responses(32, 300, 6)
    gid = np.tile([0, 1], 150)
    mask = np.ones_like(y, dtype=bool)
    mask[0, 0] = False  # a single unobserved cell exercises the mask branch
    # The LR statistic stabilizes by 501 nodes on this fixture (501 vs. 961:
    # ~3e-8 difference); 11 nodes also prevented convergence at 120 iterations.
    result = fm.dif_analysis(
        y,
        np.zeros(6, dtype=np.int64),
        gid,
        config=FitConfig(model="MIRT", estimator="mmle", max_iter=120, q_theta=501),
        mask=mask,
        studied_items=[2],
    )
    assert np.isfinite(result.lr_statistic[2])


def test_m2_single_population_requires_two_complete_cases():
    # a single-population spatial M2 with fewer than two complete cases.
    y, fid, params = _spatial_params([0] * 7, 12, seed=31)
    y[1:, 0] = np.nan  # only the first row remains fully observed
    with pytest.raises(ValueError, match="at least two complete cases"):
        fm.m2(
            y, fid, params, "MLS2PLM", q_theta=7, q_xi=7,
            estimate_population=True, prior_mean=np.zeros(1), prior_sd=np.ones(1),
        )


def test_m2_single_population_too_few_columns_and_cases():
    # spatial model with too few items => moments <= parameters.
    y6, fid6, p6 = _spatial_params([0] * 6, 200, seed=26)
    with pytest.raises(ValueError, match="df non-positive"):
        fm.m2(
            y6, fid6, p6, "MLS2PLM", q_theta=7, q_xi=7,
            estimate_population=True, prior_mean=np.zeros(1), prior_sd=np.ones(1),
        )
    # enough moments but too few complete cases.
    y7, fid7, p7 = _spatial_params([0] * 7, 25, seed=27)
    with pytest.raises(ValueError, match="too few complete cases"):
        fm.m2(
            y7, fid7, p7, "MLS2PLM", q_theta=7, q_xi=7,
            estimate_population=True, prior_mean=np.zeros(1), prior_sd=np.ones(1),
        )


# ---------------------------------------------------------------------------
# multigroup / multilevel M2 validation guards + small helpers
# ---------------------------------------------------------------------------


def test_m2_multigroup_guards():
    y, fid, params = _mirt_params(6, 40, seed=28)
    gid = np.tile([0, 1], 20)
    mean = np.zeros((2, 1))
    sd = np.ones((2, 1))
    with pytest.raises(ValueError, match="persons-by-items matrix"):
        fm.m2_multigroup(np.zeros(6), fid, params, "MIRT", gid, mean, sd)
    nonbinary = y.copy()
    nonbinary[0, 0] = 2.0
    with pytest.raises(ValueError, match="finite binary values"):
        fm.m2_multigroup(nonbinary, fid, params, "MIRT", gid, mean, sd)
    with pytest.raises(ValueError, match="factor_id length must match"):
        fm.m2_multigroup(y, np.zeros(5, dtype=np.int64), params, "MIRT", gid, mean, sd)
    with pytest.raises(ValueError, match=r"must have shape"):
        fm.m2_multigroup(y, fid, params, "MIRT", gid, np.zeros((2, 2)), sd)
    with pytest.raises(ValueError, match="SDs finite and positive"):
        fm.m2_multigroup(y, fid, params, "MIRT", gid, mean, np.zeros((2, 1)))


def test_m2_multilevel_guards():
    y, fid, params = _mirt_params(6, 40, seed=29)
    cid = np.repeat(np.arange(20), 2)
    with pytest.raises(ValueError, match="persons-by-items matrix"):
        fm.m2_multilevel(np.zeros(6), fid, params, "MIRT", cid, 0.5)
    with pytest.raises(ValueError, match="mask must match"):
        fm.m2_multilevel(y, fid, params, "MIRT", cid, 0.5, mask=np.ones((2, 6), dtype=bool))
    nonbinary = y.copy()
    nonbinary[0, 0] = 2.0
    with pytest.raises(ValueError, match="finite binary values"):
        fm.m2_multilevel(nonbinary, fid, params, "MIRT", cid, 0.5)
    with pytest.raises(ValueError, match="sigma_u must be finite and non-negative"):
        fm.m2_multilevel(y, fid, params, "MIRT", cid, -1.0)
    with pytest.raises(ValueError, match="factor_id length must match"):
        fm.m2_multilevel(y, np.zeros(5, dtype=np.int64), params, "MIRT", cid, 0.5)


def test_cluster_moment_covariance_requires_enough_clusters():
    z_rows = np.zeros((6, 5))
    model_moments = np.zeros(5)
    cluster_id = np.array([0, 0, 1, 1, 2, 2])  # 3 clusters <= 5 moments
    with pytest.raises(ValueError, match="more clusters than moments"):
        fm._cluster_moment_covariance(z_rows, model_moments, cluster_id)


def test_n_dims_of_counts_trait_dimensions():
    assert fm.n_dims_of(np.array([0, 0, 1, 2, 2], dtype=np.int64)) == 3
    assert fm.n_dims_of(np.zeros(4, dtype=np.int64)) == 1
