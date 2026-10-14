"""Python binding contract for stage-3 Oakes standard errors
(ContextualWisdomLab/fast-mlsirm#1912).

Covers ``fast_mlsirm.bifactor_grm.bifactor_oakes_se``: shapes and the
vcov/SE contract at a fitted MLE, the non-positive-definite flag path
(``None`` SEs with a reason, never substitutes), and caller-argument
validation.

References
----------
Oakes, D. (1999). Direct calculation of the information matrix via the EM
algorithm. *Journal of the Royal Statistical Society Series B: Statistical
Methodology, 61*(2), 479-482. https://doi.org/10.1111/1467-9868.00188

Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
Full-information item bifactor analysis of graded response data. *Applied
Psychological Measurement, 31*(1), 4-19.
https://doi.org/10.1177/0146621606289485
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.bifactor_grm import bifactor_oakes_se, fit_bifactor_grm

N_PERSONS = 300
N_ITEMS = 6
N_SPECIFIC = 2
N_CAT = 4
SEED = 20260916
SPECIFIC_MAP = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
TRUE_A_G = np.array([1.4, -1.1, 1.0, 1.2, 0.9, 1.1])
TRUE_A_S = np.array([1.0, 0.9, 1.1, 1.0, 0.8, 0.9])
TRUE_D = np.array(
    [
        [1.2, 0.0, -1.2],
        [1.0, -0.1, -1.3],
        [1.3, 0.2, -1.0],
        [1.1, 0.1, -1.1],
        [0.9, -0.2, -1.4],
        [1.2, 0.0, -1.2],
    ]
)


def _simulate(seed: int, n: int = N_PERSONS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta_g = rng.normal(0.0, 1.0, n)
    theta_s = rng.normal(0.0, 1.0, (n, N_SPECIFIC))
    y = np.zeros((n, N_ITEMS), dtype=np.int64)
    for i in range(N_ITEMS):
        base = TRUE_A_G[i] * theta_g + TRUE_A_S[i] * theta_s[:, SPECIFIC_MAP[i]]
        cum = 1.0 / (1.0 + np.exp(-(base[:, None] + TRUE_D[i][None, :])))
        probs = np.concatenate(
            [1.0 - cum[:, [0]], -np.diff(cum, axis=1), cum[:, [-1]]], axis=1
        )
        draws = rng.random(n)
        y[:, i] = (draws[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    return y


def _fit(y: np.ndarray):
    return fit_bifactor_grm(
        y,
        SPECIFIC_MAP,
        N_CAT,
        N_SPECIFIC,
        q_general=7,
        q_specific=7,
        max_iter=500,
        tol=1e-5,
        n_starts=1,
        seed=SEED,
    )


def _se_at_fit(fit, y: np.ndarray):
    return bifactor_oakes_se(
        fit.a_general,
        fit.a_specific,
        fit.threshold,
        y,
        SPECIFIC_MAP,
        N_CAT,
        N_SPECIFIC,
        q_general=7,
        q_specific=7,
        fd_step=1e-5,
    )


def test_se_returns_matching_vcov_and_se_at_fitted_mle() -> None:
    y = _simulate(SEED)
    fit = _fit(y)
    assert fit.converged, fit.termination_reason
    res = _se_at_fit(fit, y)
    assert res.positive_definite, res.non_pd_reason
    assert res.non_pd_reason is None
    k = N_ITEMS * (2 + N_CAT - 1)
    assert len(res.labels) == k
    assert res.information.shape == (k, k)
    assert res.vcov is not None and res.vcov.shape == (k, k)
    assert res.se is not None and res.se.shape == (k,)
    assert bool(np.all(np.isfinite(res.se)) and np.all(res.se > 0))
    # Symmetry of information and vcov; se == sqrt(diag(vcov)) exactly.
    np.testing.assert_allclose(res.information, res.information.T, rtol=1e-12)
    np.testing.assert_allclose(res.vcov, res.vcov.T, rtol=1e-9)
    np.testing.assert_allclose(
        res.se, np.sqrt(np.diag(res.vcov)), rtol=1e-12
    )
    # Labels follow the free order: per item [a_general, a_specific, d...].
    assert res.labels[0] == "a_general:0"
    assert res.labels[1] == "a_specific:0"
    assert res.labels[2] == "d:0:0"


def test_non_pd_information_is_flagged_not_substituted() -> None:
    # Four respondents cannot identify 24 item parameters; every category
    # is observed (so validation passes) yet the information is singular.
    y = np.array(
        [
            [0, 1, 2, 3, 0, 1],
            [1, 2, 3, 0, 1, 2],
            [2, 3, 0, 1, 2, 3],
            [3, 0, 1, 2, 3, 0],
        ],
        dtype=np.int64,
    )
    res = bifactor_oakes_se(
        TRUE_A_G,
        TRUE_A_S,
        TRUE_D,
        y,
        SPECIFIC_MAP,
        N_CAT,
        N_SPECIFIC,
        q_general=7,
        q_specific=7,
        fd_step=1e-5,
    )
    assert not res.positive_definite
    assert res.vcov is None
    assert res.se is None
    assert res.non_pd_reason is not None and len(res.non_pd_reason) > 0
    assert res.information.shape == (N_ITEMS * (2 + N_CAT - 1),) * 2


def test_rejects_out_of_range_caller_arguments() -> None:
    y = _simulate(SEED)
    fit = _fit(y)
    # #1929: no node-count cap; q_general=5 is now accepted, only < 1 is not.
    with pytest.raises(ValueError):
        bifactor_oakes_se(
            fit.a_general, fit.a_specific, fit.threshold, y, SPECIFIC_MAP,
            N_CAT, N_SPECIFIC, q_general=0, q_specific=7, fd_step=1e-5,
        )
    with pytest.raises(ValueError):
        bifactor_oakes_se(
            fit.a_general, fit.a_specific, fit.threshold, y, SPECIFIC_MAP,
            N_CAT, N_SPECIFIC, q_general=7, q_specific=7, fd_step=0.0,
        )
    with pytest.raises(ValueError):
        bifactor_oakes_se(
            fit.a_general, fit.a_specific, fit.threshold[:, :1], y,
            SPECIFIC_MAP, N_CAT, N_SPECIFIC, q_general=7, q_specific=7,
            fd_step=1e-5,
        )


def test_core_binding_rejects_negative_y_without_mask() -> None:
    # The PyO3 binding must not silently coerce negative categories to 0
    # when no observed mask is given (masked-out cells may carry negative
    # placeholders; unmasked cells must be non-negative).
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    assert core is not None and hasattr(core, "bifactor_oakes_se")
    y = _simulate(SEED)
    bad = y.astype(np.int64).reshape(-1).copy()
    bad[0] = -1
    with pytest.raises(ValueError):
        core.bifactor_oakes_se(
            TRUE_A_G.astype(np.float64),
            TRUE_A_S.astype(np.float64),
            TRUE_D.astype(np.float64).reshape(-1),
            bad,
            None,
            SPECIFIC_MAP.astype(np.int64),
            N_PERSONS,
            N_ITEMS,
            N_SPECIFIC,
            N_CAT,
            7,
            7,
            1e-5,
        )
