"""Trust-boundary regressions for public RSM response evidence."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.rsm import fit_rsm


class _FloatTrap:
    """Fail if package admission converts one caller-owned object cell."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller element conversion executed")


class _ArrayTrap:
    """Fail if package admission invokes a caller-owned array protocol."""

    callbacks = 0

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        type(self).callbacks += 1
        raise AssertionError("caller array protocol executed")


class _FakeCore:
    """Minimal Rust-boundary recorder for valid response compatibility."""

    def __init__(self) -> None:
        self.yy: np.ndarray | None = None
        self.observed: np.ndarray | None = None

    def fit_rsm(
        self,
        yy,
        observed,
        n_persons,
        n_items,
        n_cat,
        q_theta,
        max_iter,
        tol,
    ):
        self.yy = np.asarray(yy)
        self.observed = np.asarray(observed)
        return {
            "item_location": np.zeros(n_items),
            "thresholds": np.zeros(n_cat - 1),
            "theta": np.zeros(n_persons),
            "loglik_trace": np.zeros(1),
            "n_iter": 1,
            "converged": True,
            "n_parameters": n_items + n_cat - 2,
        }


def _native_must_not_run():
    raise AssertionError("compiled-core discovery must not run for rejected evidence")


def test_array_provider_fails_without_protocol_or_native_discovery() -> None:
    _ArrayTrap.callbacks = 0
    with patch("fast_mlsirm.fitstats._core_module", side_effect=_native_must_not_run):
        with pytest.raises(ValueError, match="responses must be a real numeric array"):
            fit_rsm(_ArrayTrap(), n_cat=3, q_theta=41, max_iter=500, tol=1e-6)
    assert _ArrayTrap.callbacks == 0


def test_complex_responses_fail_before_lossy_cast_or_native_discovery() -> None:
    responses = np.array([[0.0 + 1.0j, 1.0], [1.0, 2.0]], dtype=np.complex128)
    with patch("fast_mlsirm.fitstats._core_module", side_effect=_native_must_not_run):
        with pytest.raises(ValueError, match="responses must be a real numeric array"):
            fit_rsm(responses, n_cat=3, q_theta=41, max_iter=500, tol=1e-6)


def test_object_response_cells_fail_without_numeric_callbacks_or_native_discovery() -> None:
    _FloatTrap.callbacks = 0
    responses = np.array([[0, 1], [2, _FloatTrap()]], dtype=object)
    with patch("fast_mlsirm.fitstats._core_module", side_effect=_native_must_not_run):
        with pytest.raises(ValueError, match="responses must be a real numeric array"):
            fit_rsm(responses, n_cat=3, q_theta=41, max_iter=500, tol=1e-6)
    assert _FloatTrap.callbacks == 0


def test_text_response_storage_is_not_reinterpreted_as_categories() -> None:
    responses = np.array([["0", "1"], ["1", "2"]])
    with patch("fast_mlsirm.fitstats._core_module", side_effect=_native_must_not_run):
        with pytest.raises(ValueError, match="responses must be a real numeric array"):
            fit_rsm(responses, n_cat=3, q_theta=41, max_iter=500, tol=1e-6)


def test_real_numeric_and_nan_responses_keep_existing_rust_payload_semantics() -> None:
    core = _FakeCore()
    responses = np.array([[0, 1], [2, 0], [1, np.nan]], dtype=np.float64)
    with patch("fast_mlsirm.fitstats._core_module", return_value=core):
        fit_rsm(responses, n_cat=3, q_theta=41, max_iter=500, tol=1e-6)
    assert core.yy is not None
    assert core.observed is not None
    np.testing.assert_array_equal(core.yy, np.array([0, 1, 2, 0, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(
        core.observed,
        np.array([True, True, True, True, True, False]),
    )


def test_builtin_and_exact_numpy_rows_preserve_response_compatibility() -> None:
    inputs = [
        [[np.int8(0), np.uint8(1)], (np.float32(2.0), 0), [1, np.nan]],
        [
            np.array([0, 1], dtype=np.int16),
            np.array([2, 0], dtype=np.uint8),
            np.array([1.0, np.nan], dtype=np.float32),
        ],
    ]

    for responses in inputs:
        core = _FakeCore()
        with patch("fast_mlsirm.fitstats._core_module", return_value=core):
            fit_rsm(responses, n_cat=3, q_theta=41, max_iter=500, tol=1e-6)
        assert core.yy is not None
        assert core.observed is not None
        np.testing.assert_array_equal(
            core.yy,
            np.array([0, 1, 2, 0, 1, 0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            core.observed,
            np.array([True, True, True, True, True, False]),
        )


def test_builtin_rows_preserve_exact_numpy_boolean_cell_compatibility() -> None:
    core = _FakeCore()
    responses = [
        [np.bool_(False), np.bool_(True)],
        [np.bool_(True), np.bool_(False)],
        [np.bool_(False), np.bool_(True)],
    ]

    with patch("fast_mlsirm.fitstats._core_module", return_value=core):
        fit_rsm(responses, n_cat=2, q_theta=41, max_iter=500, tol=1e-6)

    assert core.yy is not None
    assert core.observed is not None
    np.testing.assert_array_equal(core.yy, np.array([0, 1, 1, 0, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(core.observed, np.ones(6, dtype=bool))
