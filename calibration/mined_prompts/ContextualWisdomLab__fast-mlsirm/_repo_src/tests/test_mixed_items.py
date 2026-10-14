from __future__ import annotations

import os

import numpy as np
import pytest

from fast_mlsirm import MixedItemParameters, fit_mixed_items, fit_polytomous


def _draw_rows(rng: np.random.Generator, probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(
        [rng.choice(probabilities.shape[1], p=row) for row in probabilities],
        dtype=float,
    )


def _dominance_bank(seed: int = 42, n_persons: int = 400):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n_persons)
    y = np.empty((n_persons, 4), dtype=float)

    y[:, 0] = rng.random(n_persons) < 1.0 / (1.0 + np.exp(-(theta - 0.2)))

    boundaries = np.asarray([0.8, -0.6])
    cumulative = 1.0 / (1.0 + np.exp(-(theta[:, None] + boundaries)))
    y[:, 1] = _draw_rows(
        rng,
        np.column_stack(
            [
                1.0 - cumulative[:, 0],
                cumulative[:, 0] - cumulative[:, 1],
                cumulative[:, 1],
            ]
        ),
    )

    logits = theta[:, None] * np.arange(3) + np.asarray([0.0, -0.2, -0.8])
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    y[:, 2] = _draw_rows(rng, probabilities)

    logits = theta[:, None] * np.asarray([0.0, -0.8, 1.2]) + np.asarray(
        [0.0, 0.2, -0.3]
    )
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    y[:, 3] = _draw_rows(rng, probabilities)
    return y, theta


def _expanded_family_bank(seed: int = 142, n_persons: int = 800):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n_persons)
    y = np.empty((n_persons, 8), dtype=float)

    y[:, 0] = rng.random(n_persons) < 1.0 / (1.0 + np.exp(-(theta - 0.2)))

    logits = theta[:, None] * np.arange(3) + np.asarray([0.0, 0.1, -0.7])
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    y[:, 1] = _draw_rows(rng, probabilities)

    core = 1.0 / (1.0 + np.exp(-(1.2 * theta - 0.3)))
    y[:, 2] = rng.random(n_persons) < 0.12 + 0.88 * core
    y[:, 3] = rng.random(n_persons) < 0.88 * core
    y[:, 4] = rng.random(n_persons) < 0.10 + 0.78 * core

    for column, slope in [(5, 1.15), (6, 1.0)]:
        q1 = 1.0 / (1.0 + np.exp(-(slope * theta + 0.5)))
        q2 = 1.0 / (1.0 + np.exp(-(slope * theta - 0.6)))
        y[:, column] = _draw_rows(
            rng,
            np.column_stack([1.0 - q1, q1 * (1.0 - q2), q1 * q2]),
        )

    y[:, 7] = rng.random(n_persons) < -np.expm1(-np.exp(theta - 0.25))
    return y, theta


def _assert_actual_convergence(fit, tol: float, max_iter: int) -> None:
    trace = np.asarray(fit.loglik_trace)
    assert fit.converged
    assert fit.termination_reason == "converged"
    assert 0 < fit.n_iter < max_iter
    assert trace.shape == (fit.n_iter + 1,)
    assert np.all(np.isfinite(trace))
    slack = 1e-8 * (1.0 + np.abs(trace[:-1]))
    assert np.all(np.diff(trace) >= -slack)
    assert abs(trace[-1] - trace[-2]) <= tol * (1.0 + abs(trace[-1]))
    assert fit.loglik == pytest.approx(trace[-1], abs=1e-12)


def test_mixed_dominance_bank_converges_and_cpu_threads_are_equivalent():
    y, theta = _dominance_bank()
    options = dict(
        item_models=["2pl", "grm", "gpcm", "nominal"],
        n_categories=[2, 3, 3, 3],
        q_theta=11,
        q_xi=7,
        max_iter=80,
        tol=1e-5,
        require_convergence=True,
    )
    serial = fit_mixed_items(y, n_threads=1, **options)
    requested_threads = min(2, os.cpu_count() or 1)
    parallel = fit_mixed_items(y, n_threads=requested_threads, **options)

    _assert_actual_convergence(serial, tol=1e-5, max_iter=80)
    _assert_actual_convergence(parallel, tol=1e-5, max_iter=80)
    assert serial.n_threads == 1
    assert parallel.n_threads == requested_threads
    assert parallel.loglik == pytest.approx(serial.loglik, abs=2e-7)
    np.testing.assert_allclose(
        parallel.theta_eap, serial.theta_eap, atol=2e-7, rtol=0.0
    )
    assert np.corrcoef(theta, parallel.theta_eap)[0, 1] > 0.65

    for serial_item, parallel_item in zip(serial.items, parallel.items, strict=True):
        assert serial_item.model == parallel_item.model
        if serial_item.slope is not None:
            assert parallel_item.slope == pytest.approx(serial_item.slope, abs=2e-7)
        np.testing.assert_allclose(
            parallel_item.intercepts, serial_item.intercepts, atol=2e-7
        )
        np.testing.assert_allclose(
            parallel_item.thresholds, serial_item.thresholds, atol=2e-7
        )
        np.testing.assert_allclose(parallel_item.scores, serial_item.scores, atol=2e-7)


def test_expanded_response_families_jointly_converge_with_valid_constraints():
    y, theta = _expanded_family_bank()
    models = ["rasch", "pcm", "3pl", "3plu", "4pl", "sequential", "tutz", "cll"]
    fit = fit_mixed_items(
        y,
        models,
        [2, 3, 2, 2, 2, 3, 3, 2],
        q_theta=11,
        max_iter=100,
        tol=1e-6,
        n_threads=2,
        require_convergence=True, q_xi=7
    )

    _assert_actual_convergence(fit, tol=1e-6, max_iter=100)
    assert [item.model for item in fit.items] == models
    assert fit.items[0].slope == 1.0
    assert fit.items[1].slope == 1.0
    assert fit.items[6].slope == 1.0
    assert fit.items[7].slope == 1.0
    assert fit.items[5].intercepts.shape == (2,)
    assert fit.items[6].intercepts.shape == (2,)
    for index in [2, 3, 4]:
        lower = fit.items[index].lower_asymptote
        upper = fit.items[index].upper_asymptote
        assert lower is not None and upper is not None
        assert 0.0 <= lower < upper <= 1.0
    assert np.corrcoef(theta, fit.theta_eap)[0, 1] > 0.65


def test_homogeneous_two_pl_matches_existing_gpcm_binary_cell():
    rng = np.random.default_rng(19)
    n_persons = 500
    theta = rng.normal(size=n_persons)
    slope = np.asarray([0.8, 1.1, 1.4, 1.0])
    intercept = np.asarray([-1.0, -0.3, 0.4, 1.0])
    probability = 1.0 / (1.0 + np.exp(-(theta[:, None] * slope + intercept)))
    y = (rng.random(probability.shape) < probability).astype(float)

    mixed = fit_mixed_items(
        y,
        "2pl",
        [2] * y.shape[1],
        q_theta=11,
        max_iter=80,
        tol=1e-5,
        n_threads=2,
        require_convergence=True, q_xi=7
    )
    homogeneous = fit_polytomous(y, 2, model="gpcm", q_theta=11, max_iter=80, tol=1e-5)

    _assert_actual_convergence(mixed, tol=1e-5, max_iter=80)
    assert mixed.loglik == pytest.approx(homogeneous.loglik, abs=1e-6)
    np.testing.assert_allclose(
        [item.slope for item in mixed.items], homogeneous.slope, atol=2e-2, rtol=0.0
    )
    np.testing.assert_allclose(
        [item.intercepts[0] for item in mixed.items],
        homogeneous.cat_params[:, 0],
        atol=5e-3,
        rtol=0.0,
    )


def test_mixed_lsirm_and_nonspatial_items_share_one_fitted_population():
    rng = np.random.default_rng(71)
    n_persons = 300
    theta = rng.normal(size=n_persons)
    xi = rng.normal(size=n_persons)
    y = np.empty((n_persons, 4), dtype=float)
    y[:, 0] = rng.random(n_persons) < 1.0 / (1.0 + np.exp(-(theta - 0.2)))

    boundaries = np.asarray([0.7, -0.7])
    cumulative = 1.0 / (1.0 + np.exp(-(theta[:, None] + boundaries)))
    y[:, 1] = _draw_rows(
        rng,
        np.column_stack(
            [
                1.0 - cumulative[:, 0],
                cumulative[:, 0] - cumulative[:, 1],
                cumulative[:, 1],
            ]
        ),
    )

    base = 1.1 * theta - 0.7 - np.sqrt((xi - 0.5) ** 2 + 1e-8)
    y[:, 2] = rng.random(n_persons) < 1.0 / (1.0 + np.exp(-base))
    base = 0.9 * theta - np.sqrt((xi + 0.4) ** 2 + 1e-8)
    logits = base[:, None] * np.arange(3) + np.asarray([0.0, -0.1, -0.6])
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    y[:, 3] = _draw_rows(rng, probability)

    fit = fit_mixed_items(
        y,
        ["2pl", "grm", "lsirm", "lsirm_gpcm"],
        [2, 3, 2, 3],
        latent_dim=1,
        q_theta=7,
        q_xi=7,
        max_iter=60,
        tol=1e-4,
        n_threads=2,
        require_convergence=True,
    )

    _assert_actual_convergence(fit, tol=1e-4, max_iter=60)
    assert fit.xi_eap.shape == (n_persons, 1)
    assert np.all(np.isfinite(fit.xi_eap))
    assert [item.zeta.size for item in fit.items] == [0, 0, 1, 1]
    assert np.corrcoef(theta, fit.theta_eap)[0, 1] > 0.55


def test_every_response_family_dispatches_without_hiding_iteration_exhaustion():
    rng = np.random.default_rng(3)
    categories = [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 2, 4, 2, 3, 3]
    models = [
        "rasch",
        "2pl",
        "3pl",
        "3plu",
        "4pl",
        "cll",
        "grm",
        "pcm",
        "gpcm",
        "sequential",
        "tutz",
        "nominal",
        "ideal",
        "ggum",
        "lsirm",
        "lsirm_grm",
        "lsirm_gpcm",
    ]
    y = np.column_stack([rng.integers(0, count, 180) for count in categories]).astype(
        float
    )

    with pytest.warns(RuntimeWarning, match="max_iter_reached"):
        fit = fit_mixed_items(
            y,
            models,
            categories,
            latent_dim=1,
            q_theta=7,
            q_xi=7,
            max_iter=1,
            tol=1e-14,
            n_threads=2,
        )

    assert not fit.converged
    assert fit.termination_reason == "max_iter_reached"
    assert fit.n_iter == 1
    assert len(fit.loglik_trace) == 2
    assert [item.model for item in fit.items] == models
    assert np.all(np.isfinite(fit.theta_eap))


def test_item_parameter_constructor_preserves_pre_asymptote_positional_contract():
    empty = np.empty(0, dtype=float)
    item = MixedItemParameters("2pl", 2, 1.0, empty, empty, empty, None, empty)

    assert item.zeta is empty
    assert item.lower_asymptote is None
    assert item.upper_asymptote is None


def test_mixed_input_contract_and_required_convergence():
    y = np.tile([[0.0, 0.0], [1.0, 1.0]], (10, 1))
    with pytest.raises(ValueError, match="item_models length"):
        fit_mixed_items(y, ["2pl"], [2, 2], q_theta=21, q_xi=7, max_iter=100, tol=1e-5)
    with pytest.raises(ValueError, match="requires exactly 2 categories"):
        fit_mixed_items(y, ["2pl", "2pl"], [3, 2], q_theta=21, q_xi=7, max_iter=100, tol=1e-5)
    with pytest.raises(RuntimeError, match="max_iter_reached"):
        fit_mixed_items(
            y,
            ["2pl", "2pl"],
            [2, 2],
            q_theta=7,
            max_iter=1,
            tol=1e-14,
            require_convergence=True, q_xi=7
        )

    y[0, 0] = np.nan
    fit = fit_mixed_items(
        y,
        ["2pl", "2pl"],
        [2, 2],
        q_theta=7,
        max_iter=20,
        tol=1e-3,
        n_threads=1,
        require_convergence=True, q_xi=7
    )
    _assert_actual_convergence(fit, tol=1e-3, max_iter=20)
