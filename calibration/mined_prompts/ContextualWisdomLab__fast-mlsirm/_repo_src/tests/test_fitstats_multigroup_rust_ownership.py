"""Fail-first Rust ownership contract for multigroup M2 (issue #627)."""

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats


def _multigroup_case():
    """Return a small connected two-group MIRT case for ownership tests."""
    rng = np.random.default_rng(627)
    n_persons = 120
    n_items = 6
    group_id = np.repeat(np.arange(2, dtype=np.int64), n_persons // 2)
    factor_id = np.zeros(n_items, dtype=np.int64)
    group_mean = np.array([[0.0], [0.35]], dtype=np.float64)
    group_sd = np.ones((2, 1), dtype=np.float64)
    theta = rng.standard_normal(n_persons) + group_mean[group_id, 0]
    intercept = np.linspace(-0.9, 0.9, n_items)
    probability = 1.0 / (1.0 + np.exp(-(theta[:, None] + intercept[None, :])))
    responses = (rng.random((n_persons, n_items)) < probability).astype(np.float64)
    params = SimpleNamespace(
        alpha=np.zeros(n_items),
        b=intercept,
        zeta=np.zeros((n_items, 1)),
        tau=-30.0,
    )
    return responses, factor_id, params, group_id, group_mean, group_sd


def _reject_reference_projection(*_args, **_kwargs):
    """Fail if a public ownership test reaches the NumPy reference kernel."""
    raise AssertionError("public multigroup M2 entered NumPy projection arithmetic")


def _run_multigroup(case):
    """Call the public multigroup M2 path with bounded quadrature."""
    responses, factor_id, params, group_id, group_mean, group_sd = case
    return fitstats.m2_multigroup(
        responses,
        factor_id,
        params,
        "MIRT",
        group_id,
        group_mean,
        group_sd,
        q_theta=7,
        q_xi=7,
    )


def test_multigroup_m2_rejects_missing_core_before_reference_projection(monkeypatch):
    """Public multigroup M2 must fail closed when the compiled core is absent."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)
    monkeypatch.setattr(fitstats, "_projected_m2_numpy", _reject_reference_projection)

    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        _run_multigroup(_multigroup_case())


def test_multigroup_m2_requires_rust_projection_entrypoint(monkeypatch):
    """A compiled core without the projection entrypoint is still non-passing."""
    core_without_projection = SimpleNamespace(chi2_sf=lambda _x, _df: 0.5)
    monkeypatch.setattr(fitstats, "_core_module", lambda: core_without_projection)
    monkeypatch.setattr(fitstats, "_projected_m2_numpy", _reject_reference_projection)

    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        _run_multigroup(_multigroup_case())


def test_multigroup_m2_delegates_target_and_null_projection_to_rust(monkeypatch):
    """Both public multigroup projection calls must cross the Rust boundary."""
    calls = []

    class ProjectionCore:
        """Provide only the Rust projection surface used by this test."""

        @staticmethod
        def factorized_trait_moments_stat(
            _probs, _trait_weights, _space_weights, _q_theta, _factor_id, _item_values, item_offsets
        ):
            """Provide a deterministic Rust-boundary moment result for projection tests."""
            return np.full(len(item_offsets) - 1, 0.25, dtype=np.float64)

        @staticmethod
        def projected_m2(residual, delta, xi, n):
            """Record projection inputs and return distinct target/null values."""
            residual = np.asarray(residual)
            delta = np.asarray(delta)
            xi = np.asarray(xi)
            calls.append((residual.shape, delta.shape, xi.shape, float(n)))
            # Keep target/null finite and distinct while exercising downstream indices.
            return 8.0 if len(calls) == 1 else 12.0

        @staticmethod
        def chi2_sf(_x, _df):
            """Return a finite p-value for the delegated projection result."""
            return 0.5

    monkeypatch.setattr(fitstats, "_core_module", lambda: ProjectionCore())
    monkeypatch.setattr(
        fitstats,
        "_factorized_trait_moments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public multigroup M2 entered NumPy moment arithmetic")
        ),
    )
    monkeypatch.setattr(fitstats, "_projected_m2_numpy", _reject_reference_projection)

    result = _run_multigroup(_multigroup_case())

    assert np.isfinite(result.m2)
    assert result.m2 == 8.0
    assert result.null_m2 == 12.0
    assert len(calls) == 2
    for residual_shape, delta_shape, xi_shape, n in calls:
        assert residual_shape == (42,)
        assert delta_shape[0] == residual_shape[0]
        assert xi_shape == (42, 42)
        assert n == 1.0


@pytest.mark.parametrize("target", ["residual", "delta", "xi", "n"])
def test_projected_m2_native_rejects_nonfinite_inputs(target):
    """The PyO3 boundary must reject non-finite values before Rust arithmetic."""
    core = fitstats._core_module()
    if core is None:
        pytest.skip("compiled Rust core is unavailable in this test environment")

    residual = np.array([0.25, -0.25], dtype=np.float64)
    delta = np.array([[1.0], [0.5]], dtype=np.float64)
    xi = np.eye(2, dtype=np.float64)
    n = 100.0
    if target == "residual":
        residual[0] = np.nan
    elif target == "delta":
        delta[0, 0] = np.inf
    elif target == "xi":
        xi[0, 0] = np.nan
    else:
        n = np.inf

    with pytest.raises(ValueError, match="finite"):
        core.projected_m2(residual, delta, xi, n)


def test_projected_m2_native_rejects_oversized_broadcast_before_copy():
    """Logical broadcast shapes must hit the resource guard before materialization."""
    core = fitstats._core_module()
    if core is None:
        pytest.skip("compiled Rust core is unavailable in this test environment")

    size = 2049
    residual = np.zeros(size, dtype=np.float64)
    delta = np.broadcast_to(np.array([[1.0]], dtype=np.float64), (size, 1))
    xi = np.broadcast_to(np.array([[1.0]], dtype=np.float64), (size, size))

    assert not xi.flags.owndata
    with pytest.raises(ValueError, match="workspace.*supported element budget"):
        core.projected_m2(residual, delta, xi, 1.0)
