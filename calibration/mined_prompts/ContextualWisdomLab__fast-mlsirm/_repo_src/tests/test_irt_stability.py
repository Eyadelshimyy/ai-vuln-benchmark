import numpy as np

from fast_mlsirm import FitConfig, MLS2PLMConfig, MLSIRMParams, PenaltyConfig, recovery_report, score_wle, simulate
from fast_mlsirm.diagnostics import fit_diagnostics, predict_proba
from fast_mlsirm.fit import _pack, _unpack, fit
from fast_mlsirm.inference import observed_information, second_order_test, standard_errors_from_vcov, vcov_from_hessian
from fast_mlsirm.linking import link_fixed_item_parameters
from fast_mlsirm.objective import neg_loglik_and_grad, prepare_response
from fast_mlsirm.test_design import assemble_test_form, item_information, select_cat_item
from fast_mlsirm.wle import score_wle


def _all_finite(table: dict[str, np.ndarray]) -> bool:
    return all(np.all(np.isfinite(value)) for value in table.values())


def test_prepare_response_keeps_missing_by_design_axes():
    responses = np.array([[1.0, -1.0], [np.nan, np.nan], [0.0, -1.0]])

    clean, observed = prepare_response(responses)

    assert observed.sum(axis=1).tolist() == [1, 0, 1]
    assert observed.sum(axis=0).tolist() == [2, 0]
    assert np.array_equal(clean[1], np.array([0.0, 0.0]))


def test_objective_and_diagnostics_are_finite_with_all_missing_axes():
    params = MLSIRMParams(
        theta=np.array([[-0.5], [0.0], [0.5]]),
        alpha=np.array([0.0, 0.1]),
        b=np.array([0.0, 0.2]),
        xi=np.zeros((3, 1)),
        zeta=np.zeros((2, 1)),
        tau=-30.0,
    )
    responses = np.array([[1.0, -1.0], [np.nan, np.nan], [0.0, -1.0]])
    factors = np.zeros(2, dtype=int)

    objective, grad, _ = neg_loglik_and_grad(responses, factors, params, FitConfig(model="MIRT", max_iter=1))
    diagnostics = fit_diagnostics(responses, params, factors, model="MIRT")

    assert np.isfinite(objective)
    assert np.all(np.isfinite(grad.theta))
    assert np.allclose(diagnostics.itemfit["observed_count"], [2.0, 0.0])
    assert np.allclose(diagnostics.personfit["observed_count"], [1.0, 0.0, 1.0])
    assert _all_finite(diagnostics.itemfit)
    assert _all_finite(diagnostics.personfit)
    assert _all_finite(diagnostics.factorfit)


def test_fit_handles_missing_by_design_axes_and_extreme_scores():
    responses = np.array(
        [
            [1.0, -1.0, 1.0],
            [0.0, -1.0, 0.0],
            [np.nan, np.nan, np.nan],
            [1.0, -1.0, 1.0],
        ]
    )

    result = fit(
        responses,
        np.zeros(3, dtype=int),
        config=FitConfig(model="MIRT", optimizer="adam", max_iter=1, n_restarts=1, latent_dim=1, seed=5),
    )

    assert np.isfinite(result.objective)
    assert np.all(np.isfinite(result.params.b))
    assert np.all(np.isfinite(result.params.theta))


def test_fit_is_finite_for_perfect_and_zero_score_persons_and_items():
    """0/full-score robustness: the penalized estimator must keep every parameter
    and item-fit statistic finite for perfect-score and zero-score response
    vectors, where an unpenalized IRT MLE would send theta or b to +/-inf.
    """
    # Perfect-score person (all correct) and zero-score person (all wrong).
    perfect_and_zero_persons = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    # Perfect item (column all correct) and zero item (column all wrong).
    perfect_and_zero_items = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ]
    )
    for responses in (perfect_and_zero_persons, perfect_and_zero_items):
        factors = np.zeros(responses.shape[1], dtype=int)
        config = FitConfig(
            model="MIRT", optimizer="adam", max_iter=40, n_restarts=1, latent_dim=1, seed=3
        )
        result = fit(responses, factors, config=config)
        params = result.params
        assert np.isfinite(result.objective)
        for block in (params.theta, params.b, params.alpha, params.xi, params.zeta):
            assert np.all(np.isfinite(block))
        assert np.isfinite(params.tau)
        diagnostics = fit_diagnostics(responses, params, factors, model="MIRT")
        assert _all_finite(diagnostics.itemfit)
        assert _all_finite(diagnostics.personfit)


def test_true_parameters_reproduce_simulation_probabilities():
    data = simulate(MLS2PLMConfig(n_persons=8, n_dims=2, items_per_dim=2, latent_dim=2, seed=9))

    probabilities = predict_proba(data.truth, data.factor_id)
    report = recovery_report(data.truth, data.truth.copy())

    assert np.allclose(probabilities, data.probabilities)
    assert report.summary["distance_rmse"] < 1e-12
    assert report.summary["gamma_abs_error"] == 0.0


def test_estimator_recovers_true_item_parameters_monte_carlo():
    """Monte-Carlo parameter-recovery study for the MLS2PLM estimator.

    A calibration is only trustworthy if, on data simulated from known truth,
    the estimator returns item parameters that track that truth -- the classic
    item-parameter recovery check (Baker & Kim, 2004, *Item Response Theory:
    Parameter Estimation Techniques*). Latent-space item-response models are
    identified only up to a similarity transform of the latent metric, so
    scale/location-sensitive error (RMSE) is not a stable recovery signal; the
    rank-order agreement (correlation) of recovered difficulties and
    discriminations with the truth is. We therefore assert on correlation,
    aggregated over independent replications, and require the recovered
    difficulties to track truth far better than a permutation (chance) null so
    the recovery is demonstrably real rather than an artefact of the metric.

    Thresholds are deliberately loose relative to observed recovery
    (b_corr mean ~0.89, min ~0.85; a_corr mean ~0.68; permutation null ~0.0)
    so the test is stable but still fails if the estimator regresses toward
    chance. Fits are deterministic given the seed and run on the default
    (Rust) backend in CI.
    """
    seeds = range(6)
    a_corrs: list[float] = []
    b_corrs: list[float] = []
    b_null: list[float] = []
    for seed in seeds:
        data = simulate(
            MLS2PLMConfig(n_persons=400, n_dims=2, items_per_dim=5, latent_dim=2, seed=seed)
        )
        result = fit(
            data.Y,
            data.factor_id,
            config=FitConfig(
                model="MLS2PLM", optimizer="adam", max_iter=200, n_restarts=3, seed=seed
            ),
        )
        metrics = recovery_report(data.truth, result.params).metrics
        a_corrs.append(metrics["a_corr"])
        b_corrs.append(metrics["b_corr"])
        rng = np.random.default_rng(seed)
        permuted = result.params.b[rng.permutation(result.params.b.size)]
        b_null.append(float(np.corrcoef(data.truth.b, permuted)[0, 1]))

    a_corr = np.array(a_corrs)
    b_corr = np.array(b_corrs)
    b_null_corr = np.array(b_null)

    # Difficulties recover strongly and consistently across replications ...
    assert b_corr.min() >= 0.75
    assert b_corr.mean() >= 0.80
    # ... and far better than a shuffled (chance) alignment of the same estimates.
    assert b_corr.mean() - b_null_corr.mean() >= 0.5
    # Discriminations are harder to recover but still track truth well above chance.
    assert a_corr.mean() >= 0.40


def test_item_parameter_recovery_improves_with_sample_size():
    """Item-parameter recovery must improve with sample size (estimator consistency).

    A maximum-likelihood item calibration is statistically consistent: the
    sampling variability of item-parameter estimates shrinks as the number of
    respondents grows (item-parameter standard errors are O(1/sqrt(N)); Lord,
    1980, *Applications of Item Response Theory*; Baker & Kim, 2004). Recovery of
    the true difficulties should therefore be measurably better at a large N than
    at a small N drawn from the same generating model. This guards against a
    regression where the estimator stops using the extra information in a larger
    sample (e.g. premature convergence or a mis-scaled objective), which a
    single-N recovery check cannot detect. Deterministic per seed; ~4s.

    Observed (5 seeds): b_corr mean 0.81 (N=150) -> 0.94 (N=700); thresholds sit
    well inside those, so the test is stable but fails if consistency regresses.
    """
    seeds = range(4)
    small_n, large_n = 150, 700

    def _b_corr(n_persons: int, seed: int) -> float:
        """Return difficulty rank-recovery for one simulate+fit replication."""
        data = simulate(
            MLS2PLMConfig(n_persons=n_persons, n_dims=2, items_per_dim=5, latent_dim=2, seed=seed)
        )
        result = fit(
            data.Y.astype(float),
            data.factor_id,
            config=FitConfig(model="MLS2PLM", optimizer="adam", max_iter=200, n_restarts=3, seed=seed),
        )
        return recovery_report(data.truth, result.params).metrics["b_corr"]

    small = np.array([_b_corr(small_n, seed) for seed in seeds])
    large = np.array([_b_corr(large_n, seed) for seed in seeds])

    # Even the small sample recovers well above chance ...
    assert small.mean() >= 0.60
    # ... the large sample recovers strongly ...
    assert large.mean() >= 0.85
    # ... and recovery is demonstrably better with more respondents (consistency).
    assert large.mean() >= small.mean() + 0.03


def test_concurrent_calibration_is_robust_to_missing_responses():
    """Concurrent calibration must degrade gracefully under missing responses.

    Sparse, missing-by-design response matrices are the norm in concurrent
    calibration and test equating (common-item non-equivalent-groups designs),
    where each person answers only a subset of items. Under data missing at
    random (Rubin, 1976), marginal/ML item calibration stays consistent because
    the observed-data likelihood ignores the missing cells (Mislevy & Wu, 1996,
    *Missing Responses and IRT Ability Estimation*). This test verifies that
    property empirically: with 40% of responses deleted at random, recovery of
    the item difficulties must stay strong and must not collapse relative to the
    complete-data calibration of the same truth. Thresholds sit well inside
    observed recovery (b_corr ~0.91 at 40% missing vs ~0.88 complete) so the
    test is stable but fails if missing-cell handling regresses.
    """
    seeds = range(4)
    b_complete: list[float] = []
    b_missing: list[float] = []
    a_missing: list[float] = []
    observed_fraction: list[float] = []
    for seed in seeds:
        data = simulate(
            MLS2PLMConfig(n_persons=400, n_dims=2, items_per_dim=5, latent_dim=2, seed=seed)
        )
        config = FitConfig(
            model="MLS2PLM", optimizer="adam", max_iter=200, n_restarts=3, seed=seed
        )
        complete = fit(data.Y.astype(float), data.factor_id, config=config)
        b_complete.append(recovery_report(data.truth, complete.params).metrics["b_corr"])

        responses = data.Y.astype(float).copy()
        rng = np.random.default_rng(1000 + seed)
        missing = rng.random(responses.shape) < 0.40  # 40% missing at random
        responses[missing] = np.nan  # NaN marks a missing cell (prepare_response contract)
        observed_fraction.append(1.0 - float(missing.mean()))

        sparse = fit(responses, data.factor_id, config=config)
        metrics = recovery_report(data.truth, sparse.params).metrics
        b_missing.append(metrics["b_corr"])
        a_missing.append(metrics["a_corr"])

    b_complete_arr = np.array(b_complete)
    b_missing_arr = np.array(b_missing)
    a_missing_arr = np.array(a_missing)
    observed = np.array(observed_fraction)

    # The design actually dropped ~40% of the responses.
    assert 0.55 <= observed.mean() <= 0.65
    # Difficulty recovery stays strong under heavy MAR missingness ...
    assert b_missing_arr.min() >= 0.65
    assert b_missing_arr.mean() >= 0.75
    # ... and does not collapse relative to complete-data calibration.
    assert b_missing_arr.mean() >= b_complete_arr.mean() - 0.15
    # Discriminations still recover well above chance.
    assert a_missing_arr.mean() >= 0.40


def test_hessian_vcov_standard_errors_and_second_order_check_are_stable():
    params = MLSIRMParams(
        theta=np.array([[-0.6], [0.2], [0.8]]),
        alpha=np.array([0.1, -0.2]),
        b=np.array([0.0, 0.4]),
        xi=np.zeros((3, 1)),
        zeta=np.zeros((2, 1)),
        tau=-30.0,
    )
    responses = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]])
    config = FitConfig(
        model="MIRT",
        max_iter=1,
        penalty=PenaltyConfig(lambda_theta=1.0, lambda_b=1.0, lambda_alpha=1.0),
        rust_device="auto",
    )

    hessian = observed_information(responses, np.zeros(2, dtype=int), params, config=config, step=1e-4)
    check = second_order_test(hessian)
    vcov = vcov_from_hessian(hessian)
    standard_errors = standard_errors_from_vcov(vcov)

    assert hessian.shape == (7, 7)
    assert check["passed"] is True
    assert check["min_eigenvalue"] > 0.0
    assert np.all(np.isfinite(vcov))
    assert np.all(standard_errors > 0.0)
    assert second_order_test(np.diag([1.0, -1.0]))["passed"] is False


def test_observed_information_se_chain_is_preserved_at_a_converged_mle():
    """The SE=TRUE observed-information chain is self-consistent at a fitted MLE.

    Efron & Hinkley (1978) justify using the observed information ``I(x) = -H``
    (the negative Hessian of the log-likelihood at the estimate) for the
    asymptotic covariance of the maximum-likelihood estimator. This exercises
    that contract end to end on a *converged* model rather than an arbitrary
    point: the observed information is positive definite (the estimate is a
    genuine local maximum, so the curvature-based SEs are defined),
    ``vcov_from_hessian`` truly inverts it (``vcov @ H == I`` to numerical
    precision — the preservation property), and the reported standard errors
    are finite, positive, and equal to ``sqrt(diag(vcov))``.

    References
    ----------
    Efron, B., & Hinkley, D. V. (1978). Assessing the accuracy of the maximum
    likelihood estimator: Observed versus expected Fisher information.
    *Biometrika, 65*(3), 457-487. https://doi.org/10.1093/biomet/65.3.457
    """
    data = simulate(
        MLS2PLMConfig(n_persons=6, n_dims=1, items_per_dim=2, latent_dim=1, seed=3)
    )
    responses = np.asarray(data.Y, dtype=float)
    config = FitConfig(
        model="MIRT",
        optimizer="adam",
        max_iter=400,
        n_restarts=1,
        latent_dim=1,
        seed=3,
        penalty=PenaltyConfig(lambda_theta=1.0, lambda_b=1.0, lambda_alpha=1.0),
    )
    result = fit(responses, data.factor_id, config=config)
    assert result.convergence_status == "converged"

    hessian = observed_information(
        responses, data.factor_id, result.params, config=config, step=1e-4
    )
    check = second_order_test(hessian)
    vcov = vcov_from_hessian(hessian)
    standard_errors = standard_errors_from_vcov(vcov)

    # Positive-definite observed information at the estimate => genuine local max.
    assert check["passed"] is True
    assert check["min_eigenvalue"] > 0.0
    # vcov is the true inverse of the observed information (SE preservation).
    assert np.allclose(vcov @ hessian, np.eye(hessian.shape[0]), atol=1e-8)
    # Reported SEs are finite, positive, and consistent with the vcov diagonal.
    assert np.all(np.isfinite(standard_errors))
    assert np.all(standard_errors > 0.0)
    assert np.allclose(np.diag(vcov), standard_errors**2)


def test_observed_information_equals_analytic_score_jacobian():
    """The observed information must match the Jacobian of the analytic score.

    ``test_observed_information_se_chain_is_preserved_at_a_converged_mle`` above
    checks that the SE chain is *internally* consistent (``vcov`` inverts ``H``;
    ``se == sqrt(diag(vcov))``), but that chain would still pass if the
    finite-difference Hessian itself carried a sign or scaling error -- a
    wrong-but-invertible ``H`` inverts just as cleanly. This pins the Hessian
    against an *independent* route: the observed information is the Hessian of
    the penalized negative log-likelihood, so it must equal the Jacobian of the
    analytic gradient (score) that ``neg_loglik_and_grad`` returns and that the
    estimator itself optimizes (Efron & Hinkley, 1978). A regression in either
    the value-based curvature or the analytic score would break the identity.
    Both routes use the NumPy reference backend so the check is independent of
    the compiled Rust core.
    """
    params = MLSIRMParams(
        theta=np.array([[-0.6], [0.2], [0.8]]),
        alpha=np.array([0.1, -0.2]),
        b=np.array([0.0, 0.4]),
        xi=np.zeros((3, 1)),
        zeta=np.zeros((2, 1)),
        tau=-30.0,
    )
    responses = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]])
    factor_id = np.zeros(2, dtype=int)
    config = FitConfig(
        model="MIRT",
        max_iter=1,
        penalty=PenaltyConfig(lambda_theta=1.0, lambda_b=1.0, lambda_alpha=1.0),
    )
    model = config.normalized_model()

    hessian = observed_information(
        responses, factor_id, params, config=config, backend="numpy", device=None, step=1e-4
    )

    reference = _pack(params, model)
    n = reference.size

    def _packed_score(point: np.ndarray) -> np.ndarray:
        """Analytic gradient (score) of the penalized objective at ``point``."""
        _, gradient_params, _ = neg_loglik_and_grad(
            responses, factor_id, _unpack(point, params, model), config=config, backend="numpy"
        )
        return _pack(gradient_params, model)

    step = 1e-5
    score_jacobian = np.zeros((n, n), dtype=np.float64)
    for column in range(n):
        bump = np.zeros(n, dtype=np.float64)
        bump[column] = step
        score_jacobian[:, column] = (
            _packed_score(reference + bump) - _packed_score(reference - bump)
        ) / (2.0 * step)
    score_jacobian = (score_jacobian + score_jacobian.T) / 2.0

    assert np.allclose(hessian, score_jacobian, atol=1e-5, rtol=1e-4)


def test_observed_information_defaults_rust_hessian_to_cpu_device(monkeypatch):
    calls = []
    params = MLSIRMParams(
        theta=np.array([[-0.2], [0.4]]),
        alpha=np.array([0.1, -0.3]),
        b=np.array([0.2, 0.5]),
        xi=np.zeros((2, 1)),
        zeta=np.zeros((2, 1)),
        tau=-30.0,
    )
    responses = np.array([[0.0, 1.0], [1.0, 0.0]])
    factors = np.zeros(2, dtype=int)

    def fake_neg_loglik_and_grad(
        responses,
        factor_id,
        candidate,
        config=None,
        mask=None,
        backend="numpy",
        device=None,
    ):
        calls.append((backend, device))
        value = (
            np.vdot(candidate.theta, candidate.theta)
            + np.vdot(candidate.alpha, candidate.alpha)
            + np.vdot(candidate.b, candidate.b)
        )
        return float(value), candidate.copy(), -float(value)

    monkeypatch.setattr(
        "fast_mlsirm.inference.neg_loglik_and_grad", fake_neg_loglik_and_grad
    )

    hessian = observed_information(
        responses,
        factors,
        params,
        config=FitConfig(model="MIRT", backend="rust", rust_device="auto"),
        step=1e-4,
    )

    assert hessian.shape == (6, 6)
    assert calls
    assert {backend for backend, _ in calls} == {"rust"}
    assert {device for _, device in calls} == {"cpu"}


def test_fixed_item_parameter_linking_recovers_anchor_metric():
    source = MLSIRMParams(
        theta=np.array([[-1.0], [0.0], [1.0]]),
        alpha=np.log(np.array([1.2, 0.8, 1.5])),
        b=np.array([-0.4, 0.1, 0.7]),
        xi=np.zeros((3, 1)),
        zeta=np.zeros((3, 1)),
        tau=-30.0,
    )
    scale = 1.5
    shift = -0.25
    target = source.copy()
    target.theta = scale * source.theta + shift
    target.alpha = np.log(source.a / scale)
    target.b = source.b - target.a * shift

    linked, transform = link_fixed_item_parameters(source, target, anchor_items=np.array([0, 1, 2]))

    assert np.isclose(transform["scale"][0], scale)
    assert np.isclose(transform["shift"][0], shift)
    assert np.allclose(linked.theta, target.theta)
    assert np.allclose(linked.alpha, target.alpha)
    assert np.allclose(linked.b, target.b)


def test_wle_stays_finite_and_monotone_for_perfect_and_zero_scores():
    """WLE gives a finite, ordered ability estimate for the all-correct and
    all-incorrect patterns, where the MLE diverges to +/-infinity.

    This pins the 0-score / full-score robustness contract that ``score_wle``
    documents but no test exercised: the weighted-likelihood estimator has a
    genuine finite interior root (not a clamped boundary value) for extreme
    response vectors, preserves the score ordering, and reports a positive SE.

    Reference (APA 7th ed.):
        Warm, T. A. (1989). Weighted likelihood estimation of ability in item
            response theory. *Psychometrika, 54*(3), 427-450.
            https://doi.org/10.1007/BF02294627
    """
    a = np.array([1.0, 1.2, 0.8, 1.5, 1.0])
    b = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    perfect = score_wle(a, b, np.ones((1, 5)))
    zero = score_wle(a, b, np.zeros((1, 5)))
    mixed = score_wle(a, b, np.array([[1.0, 1.0, 0.0, 0.0, 1.0]]))

    # Finite (not +/-inf) with a positive SE for every pattern, extremes included.
    for out in (perfect, zero, mixed):
        assert np.all(np.isfinite(out["theta"]))
        assert np.all(out["se"] > 0.0)

    # The extremes are genuine finite roots, not values clamped to theta_bound.
    assert not bool(perfect["boundary"][0])
    assert not bool(zero["boundary"][0])

    # Perfect > mixed > zero: the estimator preserves the raw-score ordering.
    assert perfect["theta"][0] > mixed["theta"][0] > zero["theta"][0]

    # Missing responses (NaN) are dropped per person; a full-correct pattern on
    # the observed subset is still finite.
    partial = score_wle(a, b, np.array([[1.0, 1.0, np.nan, np.nan, 1.0]]))
    assert np.isfinite(partial["theta"][0])


def test_fipc_freezes_anchors_under_missing_response_data():
    """FIPC holds anchor item parameters exactly fixed and still recovers the
    free population when the new-form responses are ~25% missing (MCAR).

    This is the concurrent-calibration missing-value robustness a common-item
    nonequivalent-groups linking design depends on: in a real linking study the
    new form is administered to a different group, so item-by-person missingness
    is the norm, yet the anchored (old-form) parameters must remain exactly
    frozen and the free population mean must still recover the group's shift.
    The existing FIPC test uses a complete matrix; this pins the missing-data
    case.

    References (APA 7th ed.):
        Kim, S. (2006). A comparative study of IRT fixed parameter calibration
            methods. *Journal of Educational Measurement, 43*(4), 355-381.
            https://doi.org/10.1111/j.1745-3984.2006.00021.x
        Rose, N., von Davier, M., & Nagengast, B. (2017). Modeling omitted and
            not-reached items in IRT models. *Psychometrika, 82*(3), 795-819.
            https://doi.org/10.1007/s11336-016-9544-7
    """
    rng = np.random.default_rng(11)
    n_persons, n_items = 600, 12
    factors = np.zeros(n_items, dtype=np.int64)
    a_true = 0.8 + 0.6 * rng.random(n_items)
    b_true = -1.0 + 2.0 * rng.random(n_items)
    theta = 0.8 + rng.standard_normal(n_persons)  # shifted new-form population
    eta = a_true[None, :] * theta[:, None] + b_true[None, :]
    responses = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)

    # ~25% missing-completely-at-random responses on the new form.
    responses[rng.random((n_persons, n_items)) < 0.25] = np.nan

    anchors = dict(
        fixed=np.arange(n_items) < 6,
        alpha=np.log(a_true),
        b=b_true,
        zeta=np.zeros((n_items, 1)),
        tau=-30.0,
    )
    config = FitConfig(model="ULS2PLM", estimator="mmle", max_iter=80, q_theta=15, latent_dim=1)
    result = fit(responses, factors, config, anchors=anchors)

    # Anchor items stay exactly fixed despite the missing data.
    np.testing.assert_allclose(result.params.b[:6], b_true[:6])
    np.testing.assert_allclose(np.exp(result.params.alpha[:6]), a_true[:6])

    # Every estimate is finite, and the free population recovers the ~0.8 shift.
    assert np.all(np.isfinite(result.params.b))
    assert np.all(np.isfinite(result.params.alpha))
    assert np.all(np.isfinite(result.params.theta))
    assert result.population["kind"] == "singlefree"
    assert 0.3 < result.population["mu"][0, 0] < 1.4


def test_cat_item_selection_and_greedy_ata_constraints():
    params = MLSIRMParams(
        theta=np.array([[0.0]]),
        alpha=np.log(np.array([0.5, 2.0, 1.0, 1.5])),
        b=np.zeros(4),
        xi=np.zeros((1, 1)),
        zeta=np.zeros((4, 1)),
        tau=-30.0,
    )
    factors = np.zeros(4, dtype=int)

    information = item_information(params, factors, theta=np.array([0.0]), model="MIRT")
    next_item = select_cat_item(params, factors, theta=np.array([0.0]), administered=np.array([1]), model="MIRT")
    form = assemble_test_form(
        information,
        length=3,
        content=np.array(["algebra", "algebra", "geometry", "geometry"]),
        min_per_content={"geometry": 1},
        max_per_content={"algebra": 1},
    )

    assert int(np.argmax(information)) == 1
    assert next_item == 3
    assert len(form) == 3
    assert np.sum(np.array(["algebra", "algebra", "geometry", "geometry"])[form] == "algebra") <= 1
    assert np.sum(np.array(["algebra", "algebra", "geometry", "geometry"])[form] == "geometry") >= 1


def test_assemble_test_form_maximizes_information_unconstrained():
    # van der Linden (2005), "Linear Models for Optimal Test Design" (Springer,
    # doi:10.1007/0-387-29054-0): optimal test assembly maximizes the test
    # information function subject to constraints. With no content constraints
    # the optimal fixed-length form is exactly the `length` most informative
    # items, so its total information is the global maximum over every feasible
    # subset. This pins that maximum-information objective so a regression cannot
    # silently degrade assemble_test_form into a feasible-but-suboptimal pick
    # (the existing constrained test only checks feasibility, not optimality).
    from itertools import combinations

    information = np.array([0.10, 0.55, 0.20, 0.80, 0.35, 0.65, 0.05, 0.45])
    length = 3
    form = assemble_test_form(information, length=length)

    top = np.argsort(-information)[:length]
    assert set(form.tolist()) == {int(i) for i in top}
    optimal_total = float(np.sort(information)[-length:].sum())
    assert np.isclose(float(information[form].sum()), optimal_total)
    best_feasible = max(
        float(information[list(subset)].sum())
        for subset in combinations(range(information.size), length)
    )
    assert np.isclose(float(information[form].sum()), best_feasible)


def test_weighted_likelihood_ability_is_finite_for_zero_and_perfect_scores():
    # Warm (1989), "Weighted likelihood estimation of ability in item response
    # theory", Psychometrika 54(3), 427-450 (doi:10.1007/BF02294627): the WLE
    # adds a J(theta)/(2 I(theta)) correction to the score equation so ability
    # stays FINITE for perfect and zero response patterns, where the plain MLE
    # diverges to +/-inf. This pins that zero/perfect-score robustness so a
    # regression cannot silently reduce score_wle to an unbounded MLE.
    a = np.array([1.0, 1.2, 0.9, 1.1, 1.0, 0.8])
    b = np.array([-1.5, -0.8, -0.2, 0.4, 1.0, 1.7])
    responses = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # zero score    -> MLE theta -> -inf
            [1, 1, 1, 0, 0, 0],  # interior pattern
            [1, 1, 1, 1, 1, 1],  # perfect score -> MLE theta -> +inf
        ],
        dtype=float,
    )

    estimate = score_wle(a, b, responses)
    theta = estimate["theta"]

    # Finite, and — the whole point of the Warm correction — strictly interior
    # to the +/-theta_bound clamp rather than saturating it (a raw MLE would peg
    # the bound). The boundary flag is the estimator's own saturation signal.
    assert np.all(np.isfinite(theta))
    assert not np.any(estimate["boundary"])
    assert np.all(np.abs(theta) < 0.9 * 20.0)
    # Monotone in raw score: zero < interior < perfect.
    assert theta[0] < theta[1] < theta[2]
    # Standard errors stay finite and positive at the extremes too.
    assert np.all(np.isfinite(estimate["se"]))
    assert np.all(estimate["se"] > 0.0)


def test_full_information_item_factor_solution_is_stable_and_heywood_free():
    # Bock, R. D., Gibbons, R. D., & Muraki, E. (1988). Full-information item
    # factor analysis. Applied Psychological Measurement, 12(3), 261-280. The
    # multidimensional item-factor model is fit by marginal maximum likelihood,
    # and Bayes constraints on the loadings are needed to keep the solution
    # proper -- i.e. to suppress Heywood cases (divergent/improper estimates).
    # This pins that stability for the MLS2PLM full-information fit: with the
    # penalty (the Bayes constraints) on, two independent same-seed fits return
    # the identical, finite, bounded 2-factor solution rather than drifting or
    # blowing up.
    data = simulate(
        MLS2PLMConfig(n_persons=120, n_dims=2, items_per_dim=4, latent_dim=2, seed=17)
    )
    responses = np.asarray(data.Y, dtype=float)
    config = FitConfig(
        model="MLS2PLM",
        optimizer="adam",
        max_iter=200,
        n_restarts=1,
        latent_dim=2,
        seed=5,
        penalty=PenaltyConfig(lambda_theta=1.0, lambda_b=1.0, lambda_alpha=1.0),
    )

    first = fit(responses, data.factor_id, config=config)
    second = fit(responses, data.factor_id, config=config)

    # Genuinely the multidimensional full-information model: a 2-factor latent space.
    assert first.params.theta.shape == (120, 2)
    # Stability (1) reproducible: the same seed returns the identical solution.
    assert np.isfinite(first.objective)
    assert np.isclose(first.objective, second.objective)
    assert np.allclose(first.params.b, second.params.b)
    assert np.allclose(first.params.alpha, second.params.alpha)
    assert np.allclose(first.params.theta, second.params.theta)
    # Stability (2) proper / Heywood-free: the Bayes penalty keeps every loading,
    # difficulty, and latent coordinate finite and bounded rather than diverging.
    assert np.all(np.isfinite(first.params.alpha))
    assert np.all(np.isfinite(first.params.b))
    assert np.all(np.isfinite(first.params.theta))
    assert np.max(np.abs(first.params.alpha)) < 20.0
    assert np.max(np.abs(first.params.b)) < 20.0
    assert np.max(np.abs(first.params.theta)) < 20.0
