"""Literature-grounded parameter-recovery and estimator-stability tests.

These tests exercise the public estimation surface (``simulate`` ->
``fit`` -> ``recovery_report``) end to end. They add coverage that the
existing suite does not have: ``test_irt_stability.py`` only checks the
*identity* recovery ``recovery_report(truth, truth)`` and the objective on
tiny fixtures, so an actual "generate from known parameters, re-estimate,
compare" recovery study and a run-to-run estimator-stability check were
missing. Nothing here changes any model formula, objective, gradient, or
Rust/NumPy numeric path (AGENTS.md "Formula Scope"); the tests only assert
properties of the existing estimator.

The recovery design is the standard Monte-Carlo IRT recovery study: generate
binary responses from known item/person parameters, re-estimate, and report
correlation and bias between true and estimated parameters (Harwell, Stone,
Hsu, & Kirisci, 1996; Reckase, 2009). Because joint MLE of item parameters is
inconsistent under the incidental-parameters problem, the reliable and honest
recovery estimator here is marginal maximum likelihood (the latent trait is
integrated out over its population distribution by Gauss-Hermite quadrature and
maximised by EM), which is consistent for the item parameters (Bock & Aitkin,
1981). Marginal ML is also the estimation principle of full-information item
factor analysis, whose identified solution is unique up to the fixed latent
metric, so repeated fits are stable (Bock, Gibbons, & Muraki, 1988).

Data are generated with the latent-space interaction disabled (``gamma=0``), so
the generating model is an ordinary two-parameter logistic (2PL) item model and
the unidimensional marginal 2PL estimator (``model="ULS2PLM"``,
``estimator="mmle"``) is correctly specified for it. Tolerances are deliberately
loose (correlation floors and bias ceilings with wide headroom over the observed
values), not tight equalities, because finite-sample recovery is stochastic.

References
----------
Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
    item parameters: Application of an EM algorithm. *Psychometrika, 46*(4),
    443-459. https://doi.org/10.1007/BF02293801
Bock, R. D., Gibbons, R., & Muraki, E. (1988). Full-information item factor
    analysis. *Applied Psychological Measurement, 12*(3), 261-280.
    https://doi.org/10.1177/014662168801200305
Harwell, M., Stone, C. A., Hsu, T.-C., & Kirisci, L. (1996). Monte Carlo
    studies in item response theory. *Applied Psychological Measurement,
    20*(2), 101-125. https://doi.org/10.1177/014662169602000201
Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
    https://doi.org/10.1007/978-0-387-89976-3
"""

from __future__ import annotations

import numpy as np

from fast_mlsirm import FitConfig, MLS2PLMConfig, fit, recovery_report, simulate


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flattened arrays."""
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def test_marginal_estimator_recovers_generating_2pl_item_parameters():
    """simulate -> fit -> recovery_report recovers the generating item and
    person parameters within a documented, honest tolerance.

    Grounded in the Monte-Carlo recovery-study design (Harwell et al., 1996;
    Reckase, 2009) and the consistency of marginal ML for item parameters
    (Bock & Aitkin, 1981). Asserts correlation floors and bias ceilings, not
    tight equalities, because finite-sample recovery is stochastic.
    """
    # 2PL-generated data (latent-space interaction off) so the unidimensional
    # marginal 2PL estimator is correctly specified for the generating model.
    data = simulate(
        MLS2PLMConfig(
            n_persons=600, n_dims=1, items_per_dim=20, gamma=0.0, seed=40404
        )
    )

    result = fit(
        data.Y.astype(float),
        data.factor_id,
        FitConfig(model="ULS2PLM", estimator="mmle", max_iter=500, tolerance=1e-6),
    )
    assert result.convergence_status == "converged"

    report = recovery_report(data.truth, result.params)

    # Item discrimination (a) and difficulty (b): high rank/linear agreement.
    assert report.metrics["a_corr"] > 0.82
    assert report.metrics["b_corr"] > 0.92
    # Bias (mean of estimate - truth) stays small (well inside the observed
    # |bias| < 0.15; ceilings carry wide headroom for platform variance).
    assert abs(report.metrics["a_bias"]) < 0.5
    assert abs(report.metrics["b_bias"]) < 0.4
    # Person ability (theta) is recovered in rank.
    assert _corr(data.truth.theta, result.params.theta) > 0.75
    # Every recovered parameter is finite.
    assert np.all(np.isfinite(result.params.a))
    assert np.all(np.isfinite(result.params.b))
    assert np.all(np.isfinite(result.params.theta))


def test_marginal_item_factor_solution_is_stable_across_run_configurations():
    """Repeated marginal-ML fits of the same data under different iteration
    budgets and convergence tolerances converge to the same identified item
    solution.

    Full-information item factor analysis maximises the marginal likelihood,
    whose solution is unique up to the fixed latent metric (Bock, Gibbons, &
    Muraki, 1988; Bock & Aitkin, 1981). For the unidimensional 2PL there is no
    rotational freedom and the N(0, 1) prior fixes location/scale/sign, so the
    optimum is a single point that any converged run must reach. Distinct
    ``seed`` values are passed to document start-configuration independence.
    """
    data = simulate(
        MLS2PLMConfig(
            n_persons=500, n_dims=1, items_per_dim=16, gamma=0.0, seed=555
        )
    )
    y = data.Y.astype(float)

    fit_a = fit(
        y,
        data.factor_id,
        FitConfig(model="ULS2PLM", estimator="mmle", max_iter=300, tolerance=1e-6, seed=1),
    )
    fit_b = fit(
        y,
        data.factor_id,
        FitConfig(model="ULS2PLM", estimator="mmle", max_iter=900, tolerance=1e-8, seed=77),
    )

    assert fit_a.convergence_status == "converged"
    assert fit_b.convergence_status == "converged"

    # Same identified optimum: near-perfect mutual correlation and negligible
    # element-wise difference in both item-parameter vectors.
    assert _corr(fit_a.params.a, fit_b.params.a) > 0.999
    assert _corr(fit_a.params.b, fit_b.params.b) > 0.999
    assert float(np.max(np.abs(fit_a.params.a - fit_b.params.a))) < 5e-3
    assert float(np.max(np.abs(fit_a.params.b - fit_b.params.b))) < 5e-3
    # Equivalent maximised marginal log-likelihood (same optimum value).
    assert abs(fit_a.loglik_trace[-1] - fit_b.loglik_trace[-1]) < 1e-3
