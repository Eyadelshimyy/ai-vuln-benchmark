"""Concurrent (multigroup) MMLE calibration robustness under missing responses.

Concurrent calibration estimates item parameters on a single common metric while
freeing each non-reference group's ability population, so a shared item bank links
groups that took overlapping-but-incomplete item sets (Kolen & Brennan, 2014,
*Test Equating, Scaling, and Linking*, 3rd ed.). Missing responses are the norm in
that design (each group answers a subset), so the marginal-maximum-likelihood EM
(Bock & Aitkin, 1981, *Psychometrika*, 46, 443-459) must integrate over the
missing cells and still return finite item/ability estimates on the common metric.

``tests/test_estimator_mmle.py`` covers single-group MMLE under missing data, and
the only existing ``group_id`` + ``mask`` combinations are validation tests that
raise before fitting. This pins the fully-exercised concurrent path: a two-group
fit with missing responses stays finite, keeps the EM log-likelihood monotone,
uses the multigroup population structure, and still recovers the item metric.
"""

from __future__ import annotations

import numpy as np

from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import fit


def _simulate_two_group(
    n_per_group=250, n_items=12, group1_shift=0.6, missing=0.2, seed=3
):
    """Return a two-group 2PL response set with a shifted group and missing cells.

    Group 0 is the N(0, 1) reference; group 1's ability distribution is shifted by
    ``group1_shift``. Both groups share the same item bank so the concurrent fit
    must place every item on one common metric.
    """
    rng = np.random.default_rng(seed)
    a = 0.7 + 1.3 * rng.random(n_items)
    b = -1.5 + 3.0 * rng.random(n_items)
    n = 2 * n_per_group
    group_id = np.concatenate(
        [np.zeros(n_per_group, dtype=np.int64), np.ones(n_per_group, dtype=np.int64)]
    )
    theta = np.concatenate(
        [
            rng.standard_normal(n_per_group),
            group1_shift + rng.standard_normal(n_per_group),
        ]
    )
    logit = a[None, :] * theta[:, None] + b[None, :]
    y = (rng.random((n, n_items)) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    mask = rng.random((n, n_items)) >= missing
    factors = np.zeros(n_items, dtype=np.int64)
    return y, factors, mask, group_id, a, b, theta


def test_concurrent_multigroup_calibration_is_robust_to_missing():
    """Two-group MMLE with missing cells stays finite and uses the multigroup metric."""
    y, factors, mask, group_id, a, b, theta = _simulate_two_group()

    result = fit(
        y,
        factors,
        FitConfig(model="ULS2PLM", estimator="mmle", max_iter=150, q_theta=21),
        mask=mask,
        group_id=group_id,
    )

    # Robustness: every estimate is finite despite two groups + 25% missing.
    assert np.all(np.isfinite(result.params.a))
    assert np.all(np.isfinite(result.params.b))
    assert np.all(np.isfinite(result.params.theta))
    assert np.isfinite(result.objective)

    # The concurrent path engaged the multigroup population structure: group 0 is
    # the fixed N(0, 1) reference and group 1's ability metric is freed.
    assert result.population is not None
    assert result.population["kind"] == "multigroup"
    mu = np.asarray(result.population["mu"]).ravel()
    sigma = np.asarray(result.population["sigma"]).ravel()
    assert mu.shape == (2,) and sigma.shape == (2,)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(np.asarray(result.population["theta_sd"])))
    assert np.isclose(mu[0], 0.0) and np.isclose(sigma[0], 1.0)  # reference fixed
    assert sigma[1] > 0.0
    assert mu[1] > 0.0  # recovers the positive group shift under missing data

    # EM stability holds under missing + multigroup (monotone non-decreasing).
    trace = np.asarray(result.loglik_trace)
    assert np.all(np.diff(trace) >= -1e-6)

    # Item parameters are still recovered on the common metric.
    assert np.corrcoef(result.params.a, a)[0, 1] > 0.6
    assert np.corrcoef(result.params.b, b)[0, 1] > 0.8
