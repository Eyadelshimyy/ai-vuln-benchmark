"""Fail-first Rust ownership contract for multilevel M2 (issue #627)."""

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats


def _multilevel_case():
    """Return a bounded random-intercept MIRT case with enough clusters."""
    rng = np.random.default_rng(6271)
    n_items = 6
    n_clusters = 40
    cluster_size = 3
    cluster_id = np.repeat(np.arange(n_clusters, dtype=np.int64), cluster_size)
    intercept = np.linspace(-0.9, 0.9, n_items)
    random_intercept = np.repeat(0.5 * rng.standard_normal(n_clusters), cluster_size)
    probability = 1.0 / (
        1.0 + np.exp(-(random_intercept[:, None] + intercept[None, :]))
    )
    responses = (rng.random(probability.shape) < probability).astype(np.float64)
    params = SimpleNamespace(
        alpha=np.zeros(n_items),
        b=intercept,
        zeta=np.zeros((n_items, 1)),
        tau=0.0,
    )
    return responses, np.zeros(n_items, dtype=np.int64), params, cluster_id


def _run_multilevel(case):
    """Call the public multilevel M2 path with bounded quadrature."""
    responses, factor_id, params, cluster_id = case
    return fitstats.m2_multilevel(
        responses,
        factor_id,
        params,
        "MIRT",
        cluster_id,
        0.5,
        q_theta=7,
        q_u=7,
        q_xi=7,
    )


def test_multilevel_m2_rejects_missing_core_before_reference_projection(monkeypatch):
    """Public multilevel M2 fails closed without Rust projection ownership."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)
    monkeypatch.setattr(
        fitstats,
        "_projected_m2_numpy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public multilevel M2 entered NumPy projection arithmetic")
        ),
    )

    with pytest.raises(
        RuntimeError, match="fit statistics require the compiled Rust core"
    ):
        _run_multilevel(_multilevel_case())


def test_multilevel_m2_delegates_target_and_null_projection_to_rust(monkeypatch):
    """Both multilevel target and independence projections cross the Rust boundary."""
    calls = []
    moment_calls = []
    covariance_calls = []

    class ProjectionCore:
        """Provide only the Rust multilevel projection surface used here."""

        @staticmethod
        def chi2_sf(_x, _df):
            """Return a finite p-value for the delegated projection result."""
            return 0.5

        @staticmethod
        def projected_m2(residual, delta, xi, n):
            """Record projection inputs and return distinct target/null values."""
            calls.append(
                (
                    np.asarray(residual).shape,
                    np.asarray(delta).shape,
                    np.asarray(xi).shape,
                    float(n),
                )
            )
            return 8.0 if len(calls) == 1 else 12.0

        @staticmethod
        def factorized_multilevel_moments_stat(
            _probs,
            _cluster_weights,
            _trait_weights,
            _space_weights,
            _q_u,
            _q_theta,
            _factor_id,
            _item_values,
            item_offsets,
        ):
            """Return bounded fake Rust moments while recording the ownership call."""
            moment_calls.append(len(item_offsets) - 1)
            return np.full(len(item_offsets) - 1, 0.25)

        @staticmethod
        def cluster_moment_covariance_stat(
            _z_rows,
            _model_moments,
            _cluster_id,
            _n_rows,
            n_moments,
            _n_clusters,
        ):
            """Return an identity covariance while recording the ownership call."""
            covariance_calls.append(n_moments)
            return np.eye(n_moments).ravel()

    monkeypatch.setattr(fitstats, "_core_module", lambda: ProjectionCore())
    monkeypatch.setattr(
        fitstats,
        "_factorized_multilevel_moments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public multilevel M2 entered NumPy moment arithmetic")
        ),
    )
    monkeypatch.setattr(
        fitstats,
        "_projected_m2_numpy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public multilevel M2 entered NumPy projection arithmetic")
        ),
    )

    result = _run_multilevel(_multilevel_case())

    assert result.m2 == 8.0
    assert result.null_m2 == 12.0
    assert len(calls) == 2
    assert moment_calls
    assert len(covariance_calls) == 2
    for residual_shape, delta_shape, xi_shape, n in calls:
        assert residual_shape == (21,)
        assert xi_shape == (21, 21)
        assert n == 120.0
    assert calls[0][1] == (21, 13)
    assert calls[1][1] == (21, 6)


def test_native_multilevel_moments_match_the_reference_reduction():
    """Rust multilevel moment integration matches the paper-backed reference path."""
    core = fitstats._core_module()
    if core is None:
        pytest.skip("compiled Rust core is unavailable in this test environment")

    probs = np.array(
        [
            [[[0.2], [0.4]], [[0.3], [0.5]]],
            [[[0.4], [0.6]], [[0.5], [0.7]]],
        ],
        dtype=np.float64,
    )
    cluster_weights = np.array([0.4, 0.6])
    trait_weights = np.array([0.25, 0.75])
    space_weights = np.array([1.0])
    factor_id = np.array([0, 0], dtype=np.int64)
    item_sets = [[0], [1], [0, 1]]

    actual = fitstats._rust_factorized_m2_moments(
        probs,
        trait_weights,
        space_weights,
        factor_id,
        item_sets,
        cluster_weights,
    )
    expected = fitstats._factorized_multilevel_moments(
        probs,
        cluster_weights,
        trait_weights,
        space_weights,
        factor_id,
        item_sets,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_native_cluster_covariance_matches_explicit_cluster_reference():
    """Rust cluster covariance preserves finite-cluster correction and centering."""
    core = fitstats._core_module()
    if core is None:
        pytest.skip("compiled Rust core is unavailable in this test environment")

    rows = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    )
    model_moments = np.array([0.5, 0.5])
    cluster_id = np.array([0, 0, 1, 1, 2, 2])
    actual, n_clusters = fitstats._cluster_moment_covariance(
        rows, model_moments, cluster_id
    )
    totals = np.add.reduceat(rows - model_moments, [0, 2, 4])
    centered = totals - totals.mean(axis=0)
    expected = (3.0 / 2.0) * (centered.T @ centered) / rows.shape[0]

    assert n_clusters == 3
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
