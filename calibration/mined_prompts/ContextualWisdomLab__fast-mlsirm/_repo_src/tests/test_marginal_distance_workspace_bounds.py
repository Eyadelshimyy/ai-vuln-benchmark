"""Resource and numerical contracts for marginal latent-space distances."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import fast_mlsirm.estimators.marginal as marginal

_FLOAT64_BYTES = np.dtype(np.float64).itemsize


def test_pairwise_distance_matches_explicit_broadcast() -> None:
    """The bounded helper must reproduce the direct Euclidean equation."""
    left = np.array([[0.0, 0.0], [1.5, -0.5], [-2.0, 3.0]], dtype=np.float64)
    right = np.array(
        [[0.25, -0.75], [1.0, 2.0], [-2.0, 3.0], [4.0, -1.0]],
        dtype=np.float64,
    )
    expected = np.sqrt(
        1e-8 + np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    )

    actual = marginal._pairwise_euclidean_distances(
        left,
        right,
        eps_distance=1e-8,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("left_rows,right_rows", [(2, 5), (5, 2)])
def test_pairwise_distance_is_translation_stable_for_large_common_offsets(
    left_rows: int,
    right_rows: int,
) -> None:
    """Large shared offsets must not erase close Euclidean separations."""
    left_bank = np.array(
        [[0.0, 0.0], [3.0, 4.0], [-5.0, 7.0], [11.0, -13.0], [17.0, 19.0]],
        dtype=np.float64,
    )
    right_bank = np.array(
        [[1.0, 2.0], [-2.0, 6.0], [8.0, -9.0], [14.0, 15.0], [-21.0, 5.0]],
        dtype=np.float64,
    )
    left = np.ascontiguousarray(left_bank[:left_rows])
    right = np.ascontiguousarray(right_bank[:right_rows])
    expected = np.sqrt(
        1e-12 + np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    )
    offset = np.float64(2**40)
    translated_left = np.ascontiguousarray(left + offset)
    translated_right = np.ascontiguousarray(right + offset)

    actual = marginal._pairwise_euclidean_distances(
        translated_left,
        translated_right,
        eps_distance=1e-12,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_pairwise_distance_handles_zero_and_roundoff() -> None:
    """Identical and nearly identical rows remain finite and non-negative."""
    left = np.array([[1.0, 2.0, 3.0], [1e8, 1e8, 1e8]], dtype=np.float64)
    right = left.copy()

    actual = marginal._pairwise_euclidean_distances(
        left,
        right,
        eps_distance=1e-12,
    )

    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0.0)
    np.testing.assert_allclose(np.diag(actual), np.sqrt(1e-12), rtol=1e-12)


def test_pairwise_distance_rejects_nonfinite_matrices_before_kernel() -> None:
    """NaN and infinity never enter the pairwise arithmetic boundary."""
    valid = np.zeros((2, 2), dtype=np.float64)
    for invalid_value in (float("nan"), float("inf"), -float("inf")):
        invalid = valid.copy()
        invalid[0, 0] = invalid_value
        with pytest.raises(ValueError, match="finite"):
            marginal._pairwise_euclidean_distances(
                invalid,
                valid,
                eps_distance=1e-8,
            )
        with pytest.raises(ValueError, match="finite"):
            marginal._pairwise_euclidean_distances(
                valid,
                invalid,
                eps_distance=1e-8,
            )


def test_checked_workspace_bytes_uses_float64_byte_accounting() -> None:
    """Distance resource limits are byte contracts, not element-count aliases."""
    assert marginal._checked_marginal_workspace_bytes(
        "test workspace",
        3,
        5,
        itemsize=_FLOAT64_BYTES,
        limit_bytes=3 * 5 * _FLOAT64_BYTES,
    ) == 3 * 5 * _FLOAT64_BYTES

    with pytest.raises(ValueError, match="test workspace"):
        marginal._checked_marginal_workspace_bytes(
            "test workspace",
            3,
            5,
            itemsize=_FLOAT64_BYTES,
            limit_bytes=(3 * 5 * _FLOAT64_BYTES) - 1,
        )


def test_distance_workspace_guard_fails_before_pairwise_allocation(monkeypatch) -> None:
    """Oversized output-plus-scratch work fails before asking NumPy for matrices."""
    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 1_000)

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._validate_marginal_distance_workspaces(
            n_items=13,
            n_x=10,
            latent_dim=3,
            uses_space=True,
        )


def test_distance_workspace_guard_fails_before_gradient_allocation(monkeypatch) -> None:
    """The intentional node-by-dimension derivative workspace is byte-bounded."""
    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 1_000)

    with pytest.raises(ValueError, match="item distance gradient workspace"):
        marginal._validate_marginal_distance_workspaces(
            n_items=2,
            n_x=63,
            latent_dim=2,
            uses_space=True,
        )


def test_public_estimator_rejects_pairwise_budget_before_node_allocation(monkeypatch) -> None:
    """A pairwise-over-budget spatial request fails before `_xi_nodes` allocates nodes."""
    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 100)

    def forbidden_xi_nodes(*_args, **_kwargs):
        raise AssertionError("_xi_nodes must not run before distance preflight")

    monkeypatch.setattr(marginal, "_xi_nodes", forbidden_xi_nodes)
    y = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    observed = np.ones_like(y, dtype=bool)
    factor_id = np.zeros(2, dtype=np.int64)

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal.fit_marginal_numpy(
            y,
            observed,
            factor_id,
            model="MLSRM",
            n_dims=1,
            latent_dim=1,
            q_theta=7,
            q_xi=7,
            max_iter=1, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_public_estimator_rejects_gradient_budget_before_node_allocation(monkeypatch) -> None:
    """A node-by-dimension-over-budget request fails before `_xi_nodes` allocation."""
    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 200)

    def forbidden_xi_nodes(*_args, **_kwargs):
        raise AssertionError("_xi_nodes must not run before gradient workspace preflight")

    monkeypatch.setattr(marginal, "_xi_nodes", forbidden_xi_nodes)
    y = np.array([[0.0], [1.0]], dtype=np.float64)
    observed = np.ones_like(y, dtype=bool)
    factor_id = np.zeros(1, dtype=np.int64)

    # Pairwise peak for 1 item x 10 nodes fits within 200 bytes, while the
    # intentional 10 x 3 float64 item-gradient workspace requires 240 bytes.
    with pytest.raises(ValueError, match="item distance gradient workspace"):
        marginal.fit_marginal_numpy(
            y,
            observed,
            factor_id,
            model="MLSRM",
            n_dims=1,
            latent_dim=3,
            q_theta=7,
            xi_rule="qmc",
            xi_points=10,
            max_iter=1, q_xi=11, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_seed=0
        )


@pytest.mark.parametrize(
    "dimensions",
    [(-1, 2), (True, 2), (2.5, 2), (2, False)],
)
def test_checked_workspace_bytes_rejects_invalid_dimensions(dimensions) -> None:
    """Malformed dimensions never reach multiplication or allocation."""
    with pytest.raises(ValueError, match="non-negative integers"):
        marginal._checked_marginal_workspace_bytes(
            "test workspace",
            *dimensions,
            itemsize=_FLOAT64_BYTES,
            limit_bytes=1_000,
        )


@pytest.mark.parametrize("invalid_value", [True, 0, -1, 1.5])
def test_checked_workspace_bytes_rejects_invalid_limits(invalid_value) -> None:
    """Malformed item sizes or ceilings cannot weaken the resource gate."""
    with pytest.raises(ValueError, match="itemsize"):
        marginal._checked_marginal_workspace_bytes(
            "test workspace",
            2,
            3,
            itemsize=invalid_value,
            limit_bytes=1_000,
        )
    with pytest.raises(ValueError, match="workspace byte limit"):
        marginal._checked_marginal_workspace_bytes(
            "test workspace",
            2,
            3,
            itemsize=_FLOAT64_BYTES,
            limit_bytes=invalid_value,
        )


def test_checked_workspace_bytes_fails_before_unbounded_integer_product() -> None:
    """Huge dimensions are rejected by checked division rather than giant products."""
    with pytest.raises(ValueError, match="test workspace"):
        marginal._checked_marginal_workspace_bytes(
            "test workspace",
            10**200,
            10**200,
            itemsize=_FLOAT64_BYTES,
            limit_bytes=128 * 1024 * 1024,
        )


def test_non_spatial_models_skip_distance_workspace_contract() -> None:
    """MIRT does not incur latent-space distance workspaces."""
    marginal._validate_marginal_distance_workspaces(
        n_items=10**9,
        n_x=10**9,
        latent_dim=3,
        uses_space=False,
    )


@pytest.mark.parametrize("invalid_flag", [0, 1, "false", None])
def test_distance_workspace_contract_requires_a_real_boolean(invalid_flag) -> None:
    """Integer and text truthiness cannot change whether resource checks run."""
    with pytest.raises(ValueError, match="uses_space"):
        marginal._validate_marginal_distance_workspaces(
            n_items=2,
            n_x=3,
            latent_dim=2,
            uses_space=invalid_flag,
        )


def test_distance_helpers_reject_invalid_shapes_dtype_and_epsilon() -> None:
    """Direct helper use fails closed for malformed numerical contracts."""
    with pytest.raises(ValueError, match="two-dimensional"):
        marginal._pairwise_euclidean_distances(
            np.zeros(2, dtype=np.float64),
            np.zeros((2, 1), dtype=np.float64),
            eps_distance=1e-8,
        )
    with pytest.raises(ValueError, match="share one latent dimension"):
        marginal._pairwise_euclidean_distances(
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            eps_distance=1e-8,
        )
    with pytest.raises(ValueError, match="float64"):
        marginal._pairwise_euclidean_distances(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((3, 2), dtype=np.float64),
            eps_distance=1e-8,
        )
    for invalid_epsilon in (True, 0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="eps_distance"):
            marginal._pairwise_euclidean_distances(
                np.zeros((2, 2), dtype=np.float64),
                np.zeros((3, 2), dtype=np.float64),
                eps_distance=invalid_epsilon,
            )


def test_pairwise_helper_source_uses_coordinate_subtraction_and_one_scratch() -> None:
    """The stable kernel must use direct coordinate subtraction without 3-D broadcast."""
    source = inspect.getsource(marginal._pairwise_euclidean_distances)
    assert "left @ right.T" not in source
    assert "np.matmul(" not in source
    assert "np.einsum(" not in source
    assert "np.subtract(" in source and "out=scratch" in source
    assert "np.square(" in source and "out=scratch" in source
    assert "np.sqrt(" in source and "out=distances" in source
    assert "left[:, None, :]" not in source
    assert "right[None, :, :]" not in source


def test_covariate_path_has_no_three_dimensional_distance_broadcast() -> None:
    """The removed item-by-node-by-dimension temporary cannot silently return."""
    source = inspect.getsource(marginal.fit_marginal_numpy)
    assert "x_grid[None, :, :] - zeta[:, None, :]" not in source
