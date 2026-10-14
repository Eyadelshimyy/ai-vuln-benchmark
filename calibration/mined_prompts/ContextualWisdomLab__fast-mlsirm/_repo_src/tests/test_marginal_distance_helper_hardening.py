"""Adversarial type and branch contracts for marginal distance helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fast_mlsirm.estimators.marginal as marginal


class DerivedArray(np.ndarray):
    """Untrusted NumPy subclass that must not cross the BLAS boundary."""


def _derived_array(shape: tuple[int, int]) -> DerivedArray:
    """Return one float64 ndarray subclass with the requested shape."""

    return np.zeros(shape, dtype=np.float64).view(DerivedArray)


def test_workspace_helper_rejects_empty_operation_name() -> None:
    """Resource errors require one stable nonempty operation name."""

    with pytest.raises(ValueError, match="name"):
        marginal._checked_marginal_workspace_bytes(
            "",
            2,
            3,
            itemsize=np.dtype(np.float64).itemsize,
            limit_bytes=1_000,
        )


def test_workspace_helper_accepts_zero_product_without_large_multiplication() -> None:
    """A zero-sized dimension short-circuits later astronomical dimensions safely."""

    assert (
        marginal._checked_marginal_workspace_bytes(
            "zero workspace",
            5,
            0,
            10**200,
            itemsize=np.dtype(np.float64).itemsize,
            limit_bytes=1_000,
        )
        == 0
    )


def test_pairwise_helper_rejects_nonarrays_and_ndarray_subclasses() -> None:
    """Only exact package-controlled ndarray storage reaches matrix multiplication."""

    valid = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="array"):
        marginal._pairwise_euclidean_distances(
            [[0.0, 0.0]],  # type: ignore[arg-type]
            valid,
            eps_distance=1e-8,
        )
    with pytest.raises(ValueError, match="exact NumPy arrays"):
        marginal._pairwise_euclidean_distances(
            _derived_array((2, 2)),
            valid,
            eps_distance=1e-8,
        )
    with pytest.raises(ValueError, match="exact NumPy arrays"):
        marginal._pairwise_euclidean_distances(
            valid,
            _derived_array((2, 2)),
            eps_distance=1e-8,
        )


def test_pairwise_helper_rejects_non_c_contiguous_inputs_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strided or Fortran input is rejected before any budgeted numeric workspace."""

    valid = np.arange(4, dtype=np.float64).reshape(2, 2)
    strided = np.arange(8, dtype=np.float64).reshape(2, 4)[:, ::2]
    fortran = np.asfortranarray(valid)
    assert valid.flags.c_contiguous
    assert not strided.flags.c_contiguous
    assert not fortran.flags.c_contiguous

    def forbidden_preflight(*_args: object, **_kwargs: object) -> None:
        """Prove layout rejection precedes the pairwise workspace boundary."""

        raise AssertionError("workspace preflight must not run for rejected layout")

    monkeypatch.setattr(
        marginal,
        "_validate_pairwise_distance_workspace",
        forbidden_preflight,
    )
    with pytest.raises(ValueError, match="C-contiguous"):
        marginal._pairwise_euclidean_distances(
            strided,
            valid,
            eps_distance=1e-8,
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        marginal._pairwise_euclidean_distances(
            valid,
            fortran,
            eps_distance=1e-8,
        )


def test_pairwise_helper_rejects_zero_latent_width() -> None:
    """A zero-width latent space cannot produce authoritative distances."""

    with pytest.raises(ValueError, match="latent dimension"):
        marginal._pairwise_euclidean_distances(
            np.zeros((2, 0), dtype=np.float64),
            np.zeros((3, 0), dtype=np.float64),
            eps_distance=1e-8,
        )


def test_pairwise_helper_rejects_nonfinite_computed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-finite post-accumulation result fails before estimator use."""

    real_sqrt = np.sqrt

    def corrupting_sqrt(values: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """Corrupt the in-place square-root result without changing inputs."""

        result = real_sqrt(values, *args, **kwargs)
        output = kwargs.get("out")
        target = output if output is not None else result
        if isinstance(target, tuple):
            target = target[0]
        target[...] = np.inf
        return target

    monkeypatch.setattr(marginal.np, "sqrt", corrupting_sqrt)
    with pytest.raises(ValueError, match="finite"):
        marginal._pairwise_euclidean_distances(
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((3, 2), dtype=np.float64),
            eps_distance=1e-8,
        )


def test_pairwise_helper_returns_owned_c_contiguous_float64() -> None:
    """The helper returns one deterministic owned matrix for downstream mutation."""

    result = marginal._pairwise_euclidean_distances(
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
        np.array([[1.0, 0.0], [3.0, 2.0]], dtype=np.float64),
        eps_distance=1e-8,
    )

    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert result.flags.owndata


@pytest.mark.parametrize("n_left,n_right", [(2, 3), (3, 2)])
def test_pairwise_helper_is_stable_under_large_common_translation(
    n_left: int,
    n_right: int,
) -> None:
    """Large exact common offsets must not erase small Euclidean separations."""

    left_bank = np.array(
        [[1.0, 2.0], [3.0, -4.0], [-5.0, 6.0]],
        dtype=np.float64,
    )[:n_left]
    right_bank = np.array(
        [[-2.0, 5.0], [6.0, 1.0], [3.0, 7.0]],
        dtype=np.float64,
    )[:n_right]
    offset = float(2**50)
    translated_left = np.ascontiguousarray(left_bank + offset)
    translated_right = np.ascontiguousarray(right_bank + offset)
    eps_distance = 1e-8

    direct = np.sqrt(
        eps_distance
        + np.sum(
            (left_bank[:, None, :] - right_bank[None, :, :]) ** 2,
            axis=2,
        )
    )
    translated = marginal._pairwise_euclidean_distances(
        translated_left,
        translated_right,
        eps_distance=eps_distance,
    )
    baseline = marginal._pairwise_euclidean_distances(
        left_bank,
        right_bank,
        eps_distance=eps_distance,
    )

    np.testing.assert_array_equal(translated, direct)
    np.testing.assert_array_equal(translated, baseline)
