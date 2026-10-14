"""Regression coverage for multigroup population-label narrowing safety."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fast_mlsirm.fit import _compact_population_labels


def test_population_labels_reject_unsigned_int64_narrowing_overflow() -> None:
    """Unsigned labels above INT64_MAX must not wrap into the reference group."""
    labels = np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64)

    with pytest.raises(ValueError, match="signed 64-bit"):
        _compact_population_labels(labels, 2, "group_id")


def test_population_labels_reject_float_int64_narrowing_overflow_without_warning() -> None:
    """Integral floats at 2**63 must fail closed rather than warn and wrap."""
    labels = np.array([0.0, float(2**63)], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="signed 64-bit"):
            _compact_population_labels(labels, 2, "group_id")


def test_population_labels_reject_integer_sequence_at_signed_boundary() -> None:
    """A Python sequence containing 2**63 must fail regardless of NumPy promotion."""
    labels = [0, 2**63]

    with pytest.raises(ValueError, match="signed 64-bit"):
        _compact_population_labels(labels, 2, "group_id")


def test_population_labels_preserve_signed_int64_upper_boundary() -> None:
    """The largest valid signed label remains admissible and order preserving."""
    labels = np.array([0, np.iinfo(np.int64).max], dtype=np.int64)

    ids, n_populations = _compact_population_labels(labels, 2, "group_id")

    assert n_populations == 2
    assert ids.tolist() == [0, 1]


def test_population_labels_preserve_extended_precision_int64_upper_boundary() -> None:
    """A wider real dtype must preserve an exactly representable INT64_MAX label."""
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("np.longdouble has no additional precision on this platform")

    labels = np.array(
        [np.longdouble(0), np.longdouble(np.iinfo(np.int64).max)],
        dtype=np.longdouble,
    )

    ids, n_populations = _compact_population_labels(labels, 2, "group_id")

    assert n_populations == 2
    assert ids.tolist() == [0, 1]


def test_population_labels_preserve_float16_without_boundary_warning() -> None:
    """Small floating dtypes must not overflow merely constructing the int64 bound."""
    labels = np.array([0.0, 1.0], dtype=np.float16)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ids, n_populations = _compact_population_labels(labels, 2, "group_id")

    assert n_populations == 2
    assert ids.tolist() == [0, 1]
