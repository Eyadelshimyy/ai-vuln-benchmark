"""Tests for the S-X², l_z/l_z*, and item-screening additions."""

from __future__ import annotations

from types import SimpleNamespace

import fast_mlsirm.fitstats as fitstats_module
import numpy as np
import pytest

from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import fit
from fast_mlsirm.fitstats import (
    adjusted_chi2_pairs,
    benjamini_hochberg,
    chi2_sf,
    _lord_wingersky,
    empirical_reliability,
    infit_outfit,
    person_fit,
    person_fit_resampling,
    residual_item_fit,
    s_x2,
    select_items,
    tcc_drift,
)


def test_chi2_sf_reference_values():
    # classic critical values
    assert chi2_sf(3.841, 1) == pytest.approx(0.05, abs=1e-3)
    assert chi2_sf(18.307, 10) == pytest.approx(0.05, abs=1e-3)
    assert chi2_sf(0.0, 5) == 1.0
    assert chi2_sf(1e6, 2) < 1e-12


def test_benjamini_hochberg_known_case():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    reject = benjamini_hochberg(p, q=0.05)
    # step-up: largest k with p_(k) <= (k/m) q is k=2 (0.008 <= 0.010)
    assert reject.sum() == 2
    assert reject[:2].all()
    p_with_nan = np.array([0.001, np.nan, 0.9])
    r2 = benjamini_hochberg(p_with_nan, q=0.05)
    assert r2[0] and not r2[2]


def test_empirical_reliability_python_wrapper():
    theta = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    theta_sd = np.array([[0.5, 1.0]] * 4)
    result = SimpleNamespace(
        params=SimpleNamespace(theta=theta),
        population={"theta_sd": theta_sd},
    )

    reliability = empirical_reliability(result, device="cpu")

    np.testing.assert_allclose(reliability, [5.0 / 6.0, 0.0])


def test_empirical_reliability_validates_device_before_native(monkeypatch):
    theta = np.zeros((2, 1))
    result = SimpleNamespace(
        params=SimpleNamespace(theta=theta),
        population={"theta_sd": np.ones_like(theta)},
    )

    class BombCore:
        def empirical_reliability(self, *_args, **_kwargs):
            raise AssertionError("invalid device reached native reliability")

    monkeypatch.setattr(fitstats_module, "_core_module", lambda: BombCore())
    with pytest.raises(ValueError, match="rust_device"):
        empirical_reliability(result, device="tpu")


def test_empirical_reliability_requires_core_and_marginal_sd(monkeypatch):
    theta = np.zeros((2, 1))
    result = SimpleNamespace(params=SimpleNamespace(theta=theta), population=None)

    with pytest.raises(ValueError, match="marginal fit with theta_sd"):
        empirical_reliability(result)

    result.population = {}
    with pytest.raises(ValueError, match="marginal fit with theta_sd"):
        empirical_reliability(result)

    result.population = {"theta_sd": np.ones_like(theta)}
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="compiled Rust core"):
        empirical_reliability(result)


def test_lord_wingersky_matches_enumeration():
    rng = np.random.default_rng(0)
    probs = rng.random((3, 4))  # 3 items, 4 nodes
    f = _lord_wingersky(probs)
    np.testing.assert_allclose(f.sum(axis=0), 1.0, atol=1e-12)
    # brute force over the 2^3 patterns
    expected = np.zeros((4, 4))
    for pattern in range(8):
        bits = [(pattern >> k) & 1 for k in range(3)]
        prob = np.ones(4)
        for k, bit in enumerate(bits):
            prob *= probs[k] if bit else (1.0 - probs[k])
        expected[sum(bits)] += prob
    np.testing.assert_allclose(f, expected, atol=1e-12)


def _simulate_2pl(seed=0, n_persons=800, n_items=12, bad_item=None):
    rng = np.random.default_rng(seed)
    a = 0.8 + 0.8 * rng.random(n_items)
    b = -1.0 + 2.0 * rng.random(n_items)
    theta = rng.standard_normal(n_persons)
    eta = a[None, :] * theta[:, None] + b[None, :]
    y = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    if bad_item is not None:
        # a grossly misfitting item: response independent of theta, extreme
        # split by an unrelated coin — S-X² should flag it
        y[:, bad_item] = (rng.random(n_persons) < 0.5).astype(float)
        y[theta > 0, bad_item] = 0.0  # negatively related to ability
    fid = np.zeros(n_items, dtype=np.int64)
    return y, fid, theta


def _fit_mirt(y, fid):
    cfg = FitConfig(
        model="MIRT", estimator="mmle", max_iter=60, q_theta=15, latent_dim=1
    )
    return fit(y, fid, cfg)


def test_sx2_flags_misfitting_item_and_spares_good_ones():
    y, fid, _ = _simulate_2pl(seed=3, bad_item=5)
    res = _fit_mirt(y, fid)
    out = s_x2(y, fid, res.params, "MIRT", q_theta=15)
    assert np.isfinite(out.statistic).sum() >= 10
    assert out.flagged_bh[5], "scrambled item must be BH-flagged"
    # most well-specified items should survive
    others = np.delete(np.arange(y.shape[1]), 5)
    assert out.flagged_bh[others].mean() < 0.5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"q_theta": 21.5}, "q_theta"),
        ({"q_xi": True}, "q_xi"),
        ({"min_expected": 0.0}, "min_expected"),
        ({"fdr_q": np.nan}, "fdr_q"),
        ({"min_effect": -0.1}, "min_effect"),
        ({"person_weight": np.array([1.0, 0.5, 1.0])}, "person_weight"),
    ],
)
def test_sx2_rejects_unsafe_controls_before_native(monkeypatch, kwargs, match):
    class BombCore:
        def s_x2_stat(self, *_args, **_kwargs):
            raise AssertionError("unsafe S-X2 inputs reached the native core")

    y = np.zeros((3, 4))
    factor_id = np.zeros(4, dtype=np.int64)
    params = SimpleNamespace(
        alpha=np.zeros(4),
        b=np.zeros(4),
        zeta=np.zeros((4, 1)),
        tau=-30.0,
    )
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: BombCore())
    with pytest.raises(ValueError, match=match):
        s_x2(y, factor_id, params, "MIRT", **kwargs)


def test_sx2_rejects_factor_length_mismatch_before_native(monkeypatch):
    class BombCore:
        def s_x2_stat(self, *_args, **_kwargs):
            raise AssertionError("unsafe S-X2 inputs reached the native core")

    params = SimpleNamespace(
        alpha=np.zeros(4),
        b=np.zeros(4),
        zeta=np.zeros((4, 1)),
        tau=-30.0,
    )
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: BombCore())
    with pytest.raises(ValueError, match="factor_id length"):
        s_x2(np.zeros((3, 4)), np.zeros(3, dtype=np.int64), params, "MIRT")


def test_sx2_extreme_probabilities_preserve_native_numpy_parity(monkeypatch):
    if fitstats_module._core_module() is None:
        pytest.skip("compiled core is unavailable")
    rng = np.random.default_rng(29)
    y = (rng.random((100, 6)) < 0.5).astype(float)
    factor_id = np.zeros(6, dtype=np.int64)
    params = SimpleNamespace(
        alpha=np.zeros(6),
        b=np.full(6, -1000.0),
        zeta=np.zeros((6, 1)),
        tau=-30.0,
    )
    native = s_x2(y, factor_id, params, "MIRT")

    # Incomplete core: tails available via pure helpers, s_x2_stat absent.
    class _TailsOnly:
        def chi2_sf(self, x, df):
            x = float(x)
            df = float(df)
            if df <= 0:
                return float("nan")
            return fitstats_module._gammainc_upper_reg(df / 2.0, max(x, 0.0) / 2.0)

        def benjamini_hochberg(self, p_values, q=0.05):
            import numpy as np

            p = np.asarray(p_values, dtype=float).ravel()
            n = p.size
            if n == 0:
                return np.zeros(0, dtype=bool)
            order = np.argsort(p)
            ranked = p[order]
            thresh = q * (np.arange(1, n + 1) / n)
            below = ranked <= thresh
            if not below.any():
                return np.zeros(n, dtype=bool)
            k = int(np.max(np.where(below)[0]))
            cut = ranked[k]
            return p <= cut

    monkeypatch.setattr(fitstats_module, "_core_module", lambda: _TailsOnly())
    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        s_x2(y, factor_id, params, "MIRT")
    assert np.all(np.isfinite(native.statistic))
    assert np.all(np.isfinite(native.g2_statistic))


def test_sx2_g2_p_values_follow_chi2_mapping():
    y, fid, _ = _simulate_2pl(seed=13, n_persons=1200, n_items=10)
    res = _fit_mirt(y, fid)
    out = s_x2(y, fid, res.params, "MIRT", q_theta=15)
    finite = np.isfinite(out.df) & (out.df >= 1) & np.isfinite(out.g2_statistic)
    assert finite.any()
    expected = np.array(
        [chi2_sf(float(g2), float(df)) for g2, df in zip(out.g2_statistic[finite], out.df[finite])]
    )
    np.testing.assert_allclose(
        out.g2_p_value[finite], expected
    )


def test_person_fit_flags_random_responders():
    y, fid, theta = _simulate_2pl(seed=4)
    rng = np.random.default_rng(42)
    aberrant = np.arange(25)
    y[aberrant] = (rng.random((25, y.shape[1])) < 0.5).astype(float)
    res = _fit_mirt(y, fid)
    pf = person_fit(y, fid, res.params, "MIRT")
    normal = np.arange(100, y.shape[0])
    # aberrant persons score systematically lower on l_z*
    assert np.nanmean(pf.lz_star[aberrant, 0]) < np.nanmean(pf.lz_star[normal, 0]) - 0.5
    # calibration: for model-consistent persons l_z* is near standard normal
    m = np.nanmean(pf.lz_star[normal, 0])
    s = np.nanstd(pf.lz_star[normal, 0])
    assert abs(m) < 0.35 and 0.6 < s < 1.6


@pytest.mark.parametrize("diagnostic", [person_fit, infit_outfit])
def test_person_diagnostics_reject_invalid_inputs_before_native(monkeypatch, diagnostic):
    class BombCore:
        def __getattr__(self, _name):
            raise AssertionError("invalid diagnostic inputs reached the native core")

    params = SimpleNamespace(
        alpha=np.zeros(4),
        b=np.zeros(4),
        zeta=np.zeros((4, 1)),
        tau=-30.0,
        theta=np.zeros((3, 1)),
        xi=np.zeros((3, 1)),
    )
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: BombCore())

    with pytest.raises(ValueError, match="factor_id length"):
        diagnostic(
            np.zeros((3, 4)), np.zeros(3, dtype=np.int64), params, "MIRT"
        )

    nonbinary = np.zeros((3, 4))
    nonbinary[0, 0] = 2.0
    with pytest.raises(ValueError, match="0 or 1"):
        diagnostic(nonbinary, np.zeros(4, dtype=np.int64), params, "MIRT")


def test_residual_item_fit_rejects_invalid_inputs_before_native(monkeypatch):
    class BombCore:
        def residual_item_fit(self, *_args, **_kwargs):
            raise AssertionError("invalid residual-fit inputs reached the native core")

    params = SimpleNamespace(
        alpha=np.zeros(4),
        b=np.zeros(4),
        zeta=np.zeros((4, 1)),
        tau=-30.0,
        theta=np.zeros((10, 1)),
        xi=np.zeros((10, 1)),
    )
    factor_id = np.zeros(4, dtype=np.int64)
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: BombCore())

    nonbinary = np.zeros((10, 4))
    nonbinary[0, 0] = 2.0
    with pytest.raises(ValueError, match="0 or 1"):
        residual_item_fit(nonbinary, factor_id, params, "MIRT", n_bins=2)

    params.theta[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        residual_item_fit(np.zeros((10, 4)), factor_id, params, "MIRT", n_bins=2)
    params.theta[0, 0] = 0.0

    with pytest.raises(ValueError, match="integer >= 2"):
        residual_item_fit(np.zeros((10, 4)), factor_id, params, "MIRT", n_bins=True)

    with pytest.raises(ValueError, match="five persons per bin"):
        residual_item_fit(np.zeros((10, 4)), factor_id, params, "MIRT", n_bins=3)

    with pytest.raises(ValueError, match="boolean"):
        residual_item_fit(
            np.zeros((10, 4)),
            factor_id,
            params,
            "MIRT",
            mask=np.ones((10, 4)),
            n_bins=2,
        )


@pytest.mark.skipif(
    fitstats_module._core_module() is None, reason="compiled core unavailable"
)
def test_adjusted_chi2_pairs_rejects_invalid_inputs_and_preserves_undefined_max():
    def params(n_persons):
        return SimpleNamespace(
            alpha=np.zeros(2),
            b=np.zeros(2),
            zeta=np.zeros((2, 1)),
            tau=-30.0,
            theta=np.zeros((n_persons, 1)),
            xi=np.zeros((n_persons, 1)),
        )

    factor_id = np.zeros(2, dtype=np.int64)
    nonbinary = np.zeros((20, 2))
    nonbinary[0, 0] = 2.0
    with pytest.raises(ValueError, match="0 or 1"):
        adjusted_chi2_pairs(
            nonbinary, factor_id, params(20), "MIRT", q_theta=7, q_xi=3
        )

    with pytest.raises(ValueError, match="boolean"):
        adjusted_chi2_pairs(
            np.zeros((20, 2)),
            factor_id,
            params(20),
            "MIRT",
            mask=np.ones((20, 2)),
            q_theta=7,
            q_xi=3,
        )

    sparse = adjusted_chi2_pairs(
        np.zeros((19, 2)), factor_id, params(19), "MIRT", q_theta=7, q_xi=3
    )
    assert np.isnan(sparse["ratio"]).all()
    assert np.isnan(sparse["mean_ratio"])
    assert np.isnan(sparse["max_ratio"])


def test_person_fit_resampling_rejects_invalid_inputs_before_native(monkeypatch):
    class BombCore:
        def person_fit_resampling(self, *_args, **_kwargs):
            raise AssertionError("invalid resampling inputs reached the native core")

    def params(n_persons=3):
        return SimpleNamespace(
            alpha=np.zeros(4),
            b=np.zeros(4),
            zeta=np.zeros((4, 1)),
            tau=-30.0,
            theta=np.zeros((n_persons, 1)),
            xi=np.zeros((n_persons, 1)),
        )

    y = np.zeros((3, 4))
    factor_id = np.zeros(4, dtype=np.int64)
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: BombCore())

    nonbinary = y.copy()
    nonbinary[0, 0] = 2.0
    with pytest.raises(ValueError, match="0 or 1"):
        person_fit_resampling(nonbinary, factor_id, params(), "MIRT")

    with pytest.raises(ValueError, match="boolean"):
        person_fit_resampling(y, factor_id, params(), "MIRT", mask=np.ones_like(y))

    bad_theta = params()
    bad_theta.theta[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        person_fit_resampling(y, factor_id, bad_theta, "MIRT")

    bad_xi_shape = params()
    bad_xi_shape.xi = np.zeros((3, 2))
    with pytest.raises(ValueError, match="params.xi shape"):
        person_fit_resampling(y, factor_id, bad_xi_shape, "MIRT")

    with pytest.raises(ValueError, match="prior_mean must be finite"):
        person_fit_resampling(
            y, factor_id, params(), "MIRT", prior_mean=np.array([np.nan])
        )

    for bad_seed in (True, 1.5, -1, 2**64):
        with pytest.raises(ValueError, match="seed"):
            person_fit_resampling(y, factor_id, params(), "MIRT", seed=bad_seed)

    with pytest.raises(ValueError, match="at least one person"):
        person_fit_resampling(np.empty((0, 4)), factor_id, params(n_persons=0), "MIRT")


def test_tcc_drift_rejects_invalid_controls_before_native(monkeypatch):
    class BombCore:
        def tcc_drift(self, *_args, **_kwargs):
            raise AssertionError("invalid TCC drift controls reached the native core")

    params = SimpleNamespace(
        alpha=np.zeros(4),
        b=np.zeros(4),
        zeta=np.zeros((4, 1)),
        tau=-30.0,
    )
    factor_id = np.zeros(4, dtype=np.int64)
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: BombCore())

    for bad_threshold in (np.nan, np.inf, -1.0, True, "0.05"):
        with pytest.raises(ValueError, match="finite non-negative"):
            tcc_drift(params, params, factor_id, "MIRT", threshold=bad_threshold)

    for name, bad_value in (("q_theta", 7.9), ("q_xi", True), ("q_theta", 0)):
        with pytest.raises(ValueError, match=name):
            tcc_drift(
                params,
                params,
                factor_id,
                "MIRT",
                **{name: bad_value},
            )


@pytest.mark.skipif(
    fitstats_module._core_module() is None, reason="compiled core unavailable"
)
def test_tcc_drift_reports_threshold_termination():
    old = SimpleNamespace(
        alpha=np.zeros(6),
        b=np.linspace(-1.2, 1.2, 6),
        zeta=np.zeros((6, 1)),
        tau=-30.0,
    )
    new = SimpleNamespace(
        alpha=old.alpha.copy(),
        b=old.b + np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        zeta=old.zeta.copy(),
        tau=old.tau,
    )
    result = tcc_drift(
        old,
        new,
        np.zeros(6, dtype=np.int64),
        "MIRT",
        threshold=0.05,
        q_theta=7,
        q_xi=3,
    )
    assert result["drifted"] == [2]
    assert result["termination_reason"] == "threshold_met"
    assert result["iterations"] == 1
    assert result["max_iterations"] == 4
    assert result["area_trace"][-1] <= 0.05


def test_select_items_removes_sparse_and_scrambled():
    y, fid, _ = _simulate_2pl(seed=5, n_persons=600, n_items=12, bad_item=7)
    y[:, 3] = 0.0
    y[:5, 3] = 1.0  # near-zero variance: sparse flag
    codes = [f"IT{i:02d}" for i in range(12)]
    out = select_items(
        y,
        fid,
        item_codes=codes,
        config=FitConfig(
            model="MIRT",
            estimator="mmle",
            max_iter=100,
            tolerance=1e-3,
            q_theta=15,
            latent_dim=1,
        ),
        max_rounds=2,
        min_items_per_dim=4,
    )
    assert "IT03" in out.removed_items, "sparse item must be removed"
    assert "IT03" in out.removed_items and "sparse" in out.removed_items["IT03"]
    assert len(out.kept_items) >= 4
    assert out.final_result is not None
    assert out.final_result.convergence_status == "converged"
    assert len(out.final_result.params.b) == len(out.kept_items)
    assert len(out.rounds) >= 1


def test_select_items_rejects_nonconverged_fit():
    y, fid, _ = _simulate_2pl(seed=19, n_persons=150, n_items=6)
    with pytest.raises(
        RuntimeError,
        match=r"status=max_iter_reached, n_iter=1, max_iter=1, .*tolerance=1e-12",
    ):
        select_items(
            y,
            fid,
            config=FitConfig(
                model="MIRT",
                estimator="mmle",
                max_iter=1,
                tolerance=1e-12,
                q_theta=11,
                latent_dim=1,
            ),
            max_rounds=1,
            min_items_per_dim=4,
        )
