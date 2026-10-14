"""Regression coverage for callback-safe IRT response admission."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.irt_contract import (
    fit_irt_experiment,
    validate_irt_response_matrix,
)


class _ArrayProvider:
    """Hostile array provider that must never run during admission."""

    calls = 0

    def __array__(self, dtype=None):
        """Fail loudly if NumPy protocol dispatch occurs before trust admission."""
        del dtype
        type(self).calls += 1
        raise AssertionError("caller array protocol executed")


class _FloatCell:
    """Hostile object-array cell that must never receive numeric conversion."""

    calls = 0

    def __float__(self):
        """Fail loudly if object storage reaches numeric coercion."""
        type(self).calls += 1
        raise AssertionError("caller numeric conversion executed")


class _BoolCell:
    """Hostile object-array cell that must never receive truth conversion."""

    calls = 0

    def __bool__(self):
        """Fail loudly if object storage reaches Boolean coercion."""
        type(self).calls += 1
        raise AssertionError("caller truth conversion executed")


def _ready_binary_responses() -> np.ndarray:
    """Return a minimal readiness-valid dichotomous response matrix."""
    return np.array(
        [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
        dtype=float,
    )


def test_response_matrix_rejects_array_provider_before_callback() -> None:
    """Arbitrary array providers must fail closed without protocol execution."""
    _ArrayProvider.calls = 0

    with pytest.raises(ValueError, match="responses must be a real numeric matrix"):
        validate_irt_response_matrix(_ArrayProvider(), "dichotomous")

    assert _ArrayProvider.calls == 0


def test_response_matrix_rejects_object_cells_before_numeric_callback() -> None:
    """Object storage must be rejected before per-cell numeric conversion."""
    _FloatCell.calls = 0
    responses = np.array([[_FloatCell(), 1], [0, 1]], dtype=object)

    with pytest.raises(ValueError, match="responses must be a real numeric matrix"):
        validate_irt_response_matrix(responses, "dichotomous")

    assert _FloatCell.calls == 0


def test_response_matrix_rejects_textual_categories() -> None:
    """Text that merely looks numeric must not become category evidence."""
    responses = np.array([["0", "1"], ["1", "0"]])

    with pytest.raises(ValueError, match="responses must be a real numeric matrix"):
        validate_irt_response_matrix(responses, "dichotomous")


def test_response_matrix_rejects_complex_before_lossy_float_cast() -> None:
    """Imaginary response evidence must never be projected onto the real axis."""
    responses = np.array(
        [[0.0 + 1.0j, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="responses must be a real numeric matrix"):
        validate_irt_response_matrix(responses, "dichotomous")


def test_fit_rejects_complex_before_fit_callable() -> None:
    """Production readiness must reject lossy evidence before numerical fitting."""
    responses = np.array(
        [
            [0.0 + 1.0j, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    calls = 0

    def _fit(_matrix, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("fit callable executed")

    with pytest.raises(ValueError, match="responses must be a real numeric matrix"):
        fit_irt_experiment(_fit, responses, "dichotomous")

    assert calls == 0


def test_fit_rejects_mask_array_provider_before_callback() -> None:
    """Mask admission must not dispatch arbitrary NumPy array providers."""
    _ArrayProvider.calls = 0
    fit_calls = 0

    def _fit(_matrix, **_kwargs):
        nonlocal fit_calls
        fit_calls += 1
        raise AssertionError("fit callable executed")

    with pytest.raises(ValueError, match="mask must be Boolean or numeric evidence"):
        fit_irt_experiment(
            _fit,
            _ready_binary_responses(),
            "dichotomous",
            mask=_ArrayProvider(),
        )

    assert _ArrayProvider.calls == 0
    assert fit_calls == 0


def test_fit_rejects_object_mask_cells_before_truth_callback() -> None:
    """Object-backed masks must fail before per-cell truth conversion."""
    _BoolCell.calls = 0
    fit_calls = 0
    mask = np.full((5, 2), True, dtype=object)
    mask[0, 0] = _BoolCell()

    def _fit(_matrix, **_kwargs):
        nonlocal fit_calls
        fit_calls += 1
        raise AssertionError("fit callable executed")

    with pytest.raises(ValueError, match="mask must be Boolean or numeric evidence"):
        fit_irt_experiment(
            _fit,
            _ready_binary_responses(),
            "dichotomous",
            mask=mask,
        )

    assert _BoolCell.calls == 0
    assert fit_calls == 0


def test_fit_validates_item_contract_before_response_work() -> None:
    """Invalid model semantics must fail before caller response materialization."""
    _ArrayProvider.calls = 0

    with pytest.raises(ValueError, match="item_type"):
        fit_irt_experiment(lambda *_args, **_kwargs: None, _ArrayProvider(), "ordinal")
    assert _ArrayProvider.calls == 0

    with pytest.raises(ValueError, match="n_categories"):
        fit_irt_experiment(
            lambda *_args, **_kwargs: None,
            _ArrayProvider(),
            "polytomous",
            n_categories=1,
        )
    assert _ArrayProvider.calls == 0


def test_fit_preserves_trusted_numeric_mask_compatibility() -> None:
    """Trusted exact numeric masks retain historical truth-value semantics."""
    mask = np.array(
        [[1, 1], [1, 1], [1, 1], [1, 1], [0, 1]],
        dtype=np.int8,
    )

    result = fit_irt_experiment(
        lambda matrix, **_kwargs: matrix,
        _ready_binary_responses(),
        "dichotomous",
        factor_ids=(0, 0),
        mask=mask,
    )

    assert np.isnan(result[4, 0])
    assert result[4, 1] == 1.0


def test_builtin_matrix_with_numpy_scalars_remains_supported() -> None:
    """Ordinary nested containers retain supported concrete NumPy scalar cells."""
    responses = [
        [np.int16(0), np.float32(1.0)],
        [np.uint8(1), np.float64(0.0)],
    ]

    matrix = validate_irt_response_matrix(responses, "dichotomous")

    assert matrix.dtype == np.float64
    assert matrix.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_builtin_matrix_with_exact_numpy_rows_remains_supported() -> None:
    """Built-in matrices may contain exact real-numeric ndarray rows."""
    responses = [
        np.array([0, 1], dtype=np.int16),
        np.array([1.0, 0.0], dtype=np.float32),
    ]

    matrix = validate_irt_response_matrix(responses, "dichotomous")

    assert matrix.dtype == np.float64
    assert matrix.tolist() == [[0.0, 1.0], [1.0, 0.0]]
