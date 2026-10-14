"""Direct Rust-delegation contracts for paired rating-range evidence."""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from fast_mlsirm import RatingRangeEvidence, paired_rating_range_evidence
from fast_mlsirm._rating_range_core_loader import rating_range_core


def test_every_public_field_matches_the_raw_rust_payload() -> None:
    """Python validates and marshals but never recomputes a numerical field."""

    automated = np.array([1, 1, 2, 3, 3, 2], dtype=np.uint32)
    reference = np.array([0, 1, 2, 3, 4, 4], dtype=np.uint32)

    result = paired_rating_range_evidence(
        automated,
        reference,
        category_count=5,
    )
    raw = rating_range_core().paired_rating_range_evidence(
        automated,
        reference,
        5,
    )

    assert isinstance(result, RatingRangeEvidence)
    assert {field.name for field in fields(RatingRangeEvidence)} == set(raw)
    for field in fields(RatingRangeEvidence):
        actual = getattr(result, field.name)
        expected = raw[field.name]
        if isinstance(actual, float):
            assert actual == float(expected)
        elif actual is None:
            assert expected is None
        else:
            assert actual == expected


def test_python_wrapper_does_not_mutate_caller_arrays() -> None:
    """Delegating to Rust leaves the exact caller-owned rating buffers unchanged."""

    automated = np.array([1, 2, 3, 2], dtype=np.uint32)
    reference = np.array([0, 2, 4, 3], dtype=np.uint32)
    automated_before = automated.copy()
    reference_before = reference.copy()

    paired_rating_range_evidence(
        automated,
        reference,
        category_count=5,
    )

    np.testing.assert_array_equal(automated, automated_before)
    np.testing.assert_array_equal(reference, reference_before)
