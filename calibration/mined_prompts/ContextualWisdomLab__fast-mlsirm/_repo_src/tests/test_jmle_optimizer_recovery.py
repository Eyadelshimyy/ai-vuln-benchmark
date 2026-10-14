"""True-parameter recovery evidence for Rust-owned public JMLE optimizers.

The optimizer-ownership migration is not accepted scientifically from delegation
sentinels alone. This study generates one deterministic, correctly specified
unidimensional MIRT data set, estimates it through each advertised public JMLE
optimizer, and checks convergence plus identification-aligned bias/MAE/RMSE
recovery. The thresholds are intentionally broad finite-sample acceptance
bounds rather than claims of asymptotic unbiasedness; JMLE retains the usual
incidental-parameter limitations.

References
----------
Harwell, M., Stone, C. A., Hsu, T.-C., & Kirisci, L. (1996). Monte Carlo
    studies in item response theory. *Applied Psychological Measurement,
    20*(2), 101-125. https://doi.org/10.1177/014662169602000201
Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
    https://doi.org/10.1007/978-0-387-89976-3
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import FitConfig, MLS2PLMConfig, fit, simulate


def _bias(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Return signed mean error over identified parameter values."""
    return float(np.mean(np.asarray(estimate) - np.asarray(truth)))


def _mae(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Return mean absolute error over identified parameter values."""
    return float(np.mean(np.abs(np.asarray(estimate) - np.asarray(truth))))


def _rmse(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Return root mean squared error over identified parameter values."""
    error = np.asarray(estimate) - np.asarray(truth)
    return float(np.sqrt(np.mean(error * error)))


def _identify_1d_mirt(truth, estimate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine-align a 1-D JMLE fit to the generating ability location/scale.

    The 2PL predictor ``a * theta + b`` is invariant to an affine change of the
    latent coordinate. For ``theta_aligned = q * (theta_est - mean_est) +
    mean_true`` with ``q = sd_true / sd_est``, the algebraically equivalent
    item parameters are ``a_aligned = a_est / q`` and
    ``b_aligned = b_est + a_est*mean_est - a_aligned*mean_true``. This is an
    identification transform only: it cannot improve the fitted logits or hide
    optimizer error.
    """
    truth_theta = np.asarray(truth.theta[:, 0], dtype=np.float64)
    estimate_theta = np.asarray(estimate.theta[:, 0], dtype=np.float64)
    truth_mean = float(truth_theta.mean())
    estimate_mean = float(estimate_theta.mean())
    truth_sd = float(truth_theta.std())
    estimate_sd = float(estimate_theta.std())
    assert truth_sd > 0.0
    assert estimate_sd > 0.0

    q = truth_sd / estimate_sd
    theta_aligned = q * (estimate_theta - estimate_mean) + truth_mean
    a_aligned = np.asarray(estimate.a, dtype=np.float64) / q
    b_aligned = (
        np.asarray(estimate.b, dtype=np.float64)
        + np.asarray(estimate.a, dtype=np.float64) * estimate_mean
        - a_aligned * truth_mean
    )

    original_eta = (
        np.asarray(estimate.a, dtype=np.float64)[None, :] * estimate_theta[:, None]
        + np.asarray(estimate.b, dtype=np.float64)[None, :]
    )
    aligned_eta = a_aligned[None, :] * theta_aligned[:, None] + b_aligned[None, :]
    np.testing.assert_allclose(aligned_eta, original_eta, rtol=1e-11, atol=1e-11)
    return theta_aligned, a_aligned, b_aligned


@pytest.fixture(scope="module")
def _jmle_recovery_data():
    """Return one deterministic, non-separated 2PL sample shared by optimizer modes.

    ``simulate`` intentionally spans easiness from 0 to 5 for broad simulation
    coverage. The original 12-item recovery fixture corrected only item margins;
    its deterministic response sample still contained 15 all-zero and 3 all-one
    person patterns, so JMLE correctly encountered person-parameter separation
    rather than an optimizer-recovery problem. Keep the generated abilities and
    discriminations, use 40 balanced-easiness items, and resample a deterministic
    response matrix whose item and person margins are both non-extreme. No fit,
    convergence, or recovery threshold is relaxed by this fixture correction.
    """
    data = simulate(
        MLS2PLMConfig(
            n_persons=160,
            n_dims=1,
            items_per_dim=40,
            latent_dim=1,
            gamma=0.0,
            seed=62612,
        )
    )
    rng = np.random.default_rng(62621)
    data.truth.b = rng.permutation(
        np.linspace(-1.5, 1.5, data.truth.b.size, dtype=np.float64)
    )
    eta = (
        data.truth.a[None, :] * data.truth.theta[:, data.factor_id]
        + data.truth.b[None, :]
    )
    data.probabilities = 1.0 / (1.0 + np.exp(-eta))
    data.Y = rng.binomial(1, data.probabilities).astype(np.uint8)

    item_rates = data.Y.mean(axis=0)
    person_rates = data.Y.mean(axis=1)
    assert np.all((item_rates > 0.05) & (item_rates < 0.95))
    assert np.all((person_rates >= 0.05) & (person_rates <= 0.95))
    return data


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs", "adam_lbfgs"])
def test_rust_jmle_optimizer_modes_recover_known_parameters(
    _jmle_recovery_data,
    optimizer: str,
) -> None:
    """Each Rust JMLE optimizer must recover after the required affine alignment."""
    data = _jmle_recovery_data
    result = fit(
        data.Y.astype(np.float64),
        data.factor_id,
        FitConfig(
            model="MIRT",
            estimator="jmle",
            optimizer=optimizer,
            latent_dim=1,
            max_iter=2000,
            n_restarts=1,
            learning_rate=0.03,
            tolerance=1e-5,
            seed=62612,
            backend="rust",
            rust_device="cpu",
        ),
    )

    theta_aligned, a_aligned, b_aligned = _identify_1d_mirt(
        data.truth,
        result.params,
    )
    truth_theta = np.asarray(data.truth.theta[:, 0], dtype=np.float64)
    metrics = {
        "a_bias": _bias(data.truth.a, a_aligned),
        "a_mae": _mae(data.truth.a, a_aligned),
        "a_rmse": _rmse(data.truth.a, a_aligned),
        "b_bias": _bias(data.truth.b, b_aligned),
        "b_mae": _mae(data.truth.b, b_aligned),
        "b_rmse": _rmse(data.truth.b, b_aligned),
        "theta_bias": _bias(truth_theta, theta_aligned),
        "theta_mae": _mae(truth_theta, theta_aligned),
        "theta_rmse": _rmse(truth_theta, theta_aligned),
    }

    evidence = {
        "optimizer": optimizer,
        "status": result.convergence_status,
        "n_iter": result.n_iter,
        **metrics,
    }

    assert result.backend == "rust", evidence
    assert result.rust_device == "cpu", evidence
    assert result.convergence_status == "converged", evidence
    assert result.n_iter > 0, evidence
    assert result.objective_trace and np.all(np.isfinite(result.objective_trace)), evidence
    assert result.objective_trace[-1] <= result.objective_trace[0], evidence

    # Finite-sample recovery gates with deliberately wide headroom. Bias, MAE,
    # and RMSE are evaluated only after the exact 2PL affine-identification
    # transform above; no threshold is relaxed to accommodate an arbitrary JMLE
    # latent scale or a non-converged optimizer.
    assert abs(metrics["a_bias"]) < 1.0, evidence
    assert metrics["a_mae"] < 1.2, evidence
    assert metrics["a_rmse"] < 1.5, evidence
    assert abs(metrics["b_bias"]) < 1.0, evidence
    assert metrics["b_mae"] < 1.2, evidence
    assert metrics["b_rmse"] < 1.5, evidence
    assert abs(metrics["theta_bias"]) < 0.15, evidence
    assert metrics["theta_mae"] < 1.1, evidence
    assert metrics["theta_rmse"] < 1.35, evidence
