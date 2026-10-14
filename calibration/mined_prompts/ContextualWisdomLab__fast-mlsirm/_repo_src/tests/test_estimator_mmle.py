"""MMLE estimator axis: routing, validation, and missing-data recovery."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import fit


def _simulate_2pl(n_persons=600, n_items=15, missing=0.30, seed=0):
    rng = np.random.default_rng(seed)
    a = 0.7 + 1.3 * rng.random(n_items)
    b = -1.5 + 3.0 * rng.random(n_items)
    theta = rng.standard_normal(n_persons)
    logit = a[None, :] * theta[:, None] + b[None, :]
    y = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    mask = rng.random((n_persons, n_items)) >= missing
    factors = np.zeros(n_items, dtype=np.int64)
    return y, factors, mask, a, b, theta


def test_mmle_recovers_item_params_under_30pct_missing():
    y, factors, mask, a, b, theta = _simulate_2pl()
    result = fit(
        y, factors, FitConfig(model="ULS2PLM", estimator="mmle", max_iter=300), mask=mask
    )
    assert result.convergence_status == "converged"
    assert result.optimizer.startswith("mmle_em/")
    assert np.corrcoef(result.params.a, a)[0, 1] > 0.8
    assert np.corrcoef(result.params.b, b)[0, 1] > 0.9
    assert np.corrcoef(result.params.theta[:, 0], theta)[0, 1] > 0.8


def test_mmle_loglik_is_monotone_nondecreasing():
    y, factors, mask, *_ = _simulate_2pl()
    result = fit(
        y, factors, FitConfig(model="ULS2PLM", estimator="mmle", max_iter=200), mask=mask
    )
    trace = np.asarray(result.loglik_trace)
    assert np.all(np.diff(trace) >= -1e-6)


@pytest.mark.parametrize("estimator", ["em", "bayes"])
def test_unimplemented_estimators_are_rejected_during_validation(estimator):
    """Reserved estimator names must fail at configuration validation, not fitting."""
    with pytest.raises(ValueError, match="estimator must be one of"):
        FitConfig(model="ULS2PLM", estimator=estimator).validate()


def test_mmle_spatial_models_route_to_marginal_estimator():
    y, factors, mask, *_ = _simulate_2pl(n_persons=60, n_items=6)
    result = fit(
        y,
        factors,
        FitConfig(model="MLS2PLM", estimator="mmle", max_iter=5, q_theta=7, q_xi=7),
        mask=mask,
    )
    assert result.optimizer.startswith("mmle_marginal_em/")
    assert result.population is not None and result.population["kind"] == "single"


def test_mmle_population_structures_require_mmle_estimator():
    y, factors, mask, *_ = _simulate_2pl(n_persons=50, n_items=5)
    with pytest.raises(ValueError, match="require estimator='mmle'"):
        fit(
            y,
            factors,
            FitConfig(model="ULS2PLM", estimator="jmle"),
            mask=mask,
            group_id=np.zeros(50, dtype=np.int64),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit(
            y,
            factors,
            FitConfig(model="ULS2PLM", estimator="mmle"),
            mask=mask,
            group_id=np.zeros(50, dtype=np.int64),
            cluster_id=np.zeros(50, dtype=np.int64),
        )


def test_invalid_estimator_rejected_by_validate():
    with pytest.raises(ValueError, match="estimator must be one of"):
        FitConfig(estimator="nope").validate()


def test_default_estimator_is_jmle():
    assert FitConfig().estimator == "jmle"
