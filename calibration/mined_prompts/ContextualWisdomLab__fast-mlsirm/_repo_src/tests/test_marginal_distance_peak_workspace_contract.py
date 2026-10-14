"""Peak-allocation contracts for bounded marginal pairwise distances."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.estimators.marginal as marginal

_FLOAT64_BYTES = np.dtype(np.float64).itemsize


def test_distance_preflight_counts_output_and_reusable_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An output that fits alone fails when the same-sized scratch is also live."""

    # Output: 12 * 10 * 8 = 960 bytes. The translation-stable kernel keeps one
    # same-shaped float64 scratch live while accumulating coordinate squares, so
    # its arithmetic phase peaks at 1,920 bytes.
    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 1_000)

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._validate_marginal_distance_workspaces(
            n_items=12,
            n_x=10,
            latent_dim=2,
            uses_space=True,
        )


def test_output_only_boundary_is_rejected_when_scratch_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stable kernel cannot authorize an output without its live scratch."""

    output_bytes = 12 * 10 * _FLOAT64_BYTES
    monkeypatch.setattr(
        marginal,
        "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES",
        output_bytes,
    )

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._validate_marginal_distance_workspaces(
            n_items=12,
            n_x=10,
            latent_dim=2,
            uses_space=True,
        )


def test_input_finiteness_mask_is_preflighted_before_numpy_mask_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wide direct-helper input fails before ``np.isfinite`` can allocate its mask."""

    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 100)
    left = np.zeros((1, 101), dtype=np.float64)
    right = np.zeros((1, 101), dtype=np.float64)

    def forbidden_isfinite(*_args: object, **_kwargs: object) -> np.ndarray:
        """Prove resource rejection happens before temporary-producing finiteness checks."""

        raise AssertionError("np.isfinite must not run before workspace preflight")

    monkeypatch.setattr(marginal.np, "isfinite", forbidden_isfinite)
    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._pairwise_euclidean_distances(
            left,
            right,
            eps_distance=1e-8,
        )


def test_pairwise_helper_applies_the_same_peak_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct helper use cannot bypass the estimator's peak preflight."""

    monkeypatch.setattr(marginal, "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES", 1_000)
    left = np.zeros((12, 2), dtype=np.float64)
    right = np.zeros((10, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._pairwise_euclidean_distances(
            left,
            right,
            eps_distance=1e-8,
        )


def test_pairwise_peak_equal_to_limit_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The output-plus-scratch byte ceiling is inclusive at the exact boundary."""

    output_bytes = 12 * 10 * _FLOAT64_BYTES
    peak_bytes = 2 * output_bytes
    monkeypatch.setattr(
        marginal,
        "MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES",
        peak_bytes,
    )
    left = np.zeros((12, 2), dtype=np.float64)
    right = np.zeros((10, 2), dtype=np.float64)

    result = marginal._pairwise_euclidean_distances(
        left,
        right,
        eps_distance=1e-8,
    )

    assert result.shape == (12, 10)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, np.sqrt(1e-8))
