"""RED contracts for paired automated/reference rating-range evidence."""

from __future__ import annotations

import math

import numpy as np
import pytest

import fast_mlsirm.validation as validation


def test_range_compressed_scorer_returns_hand_calculated_paired_evidence() -> None:
    """A scorer restricted to the middle categories exposes exact descriptive evidence."""
    automated = np.array([1, 1, 2, 3, 3], dtype=np.int64)
    reference = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    result = validation.paired_rating_range_evidence(
        automated,
        reference,
        category_count=5,
    )

    assert isinstance(result, validation.RatingRangeEvidence)
    assert result.sample_size == 5
    assert (result.automated_min, result.automated_max) == (1, 3)
    assert (result.reference_min, result.reference_max) == (0, 4)
    assert result.automated_distinct_categories == 3
    assert result.reference_distinct_categories == 5
    assert result.automated_span == 2
    assert result.reference_span == 4
    assert result.automated_sd == pytest.approx(math.sqrt(0.8), rel=1e-14)
    assert result.reference_sd == pytest.approx(math.sqrt(2.0), rel=1e-14)
    assert result.span_ratio == pytest.approx(0.5)
    assert result.distinct_category_ratio == pytest.approx(0.6)
    assert result.sd_ratio == pytest.approx(math.sqrt(0.4), rel=1e-14)
    assert result.lower_endpoint_gap == 1
    assert result.upper_endpoint_gap == 1
    assert result.narrower_observed_support is True
    assert result.central_tendency_signal is True


def test_identical_full_range_scorer_is_not_signaled() -> None:
    """Matching paired category use has unit ratios, zero gaps, and no range signal."""
    reference = np.array([0, 1, 2, 3, 4, 0, 4], dtype=np.int64)

    result = validation.paired_rating_range_evidence(
        reference.copy(),
        reference,
        category_count=5,
    )

    assert result.span_ratio == pytest.approx(1.0)
    assert result.distinct_category_ratio == pytest.approx(1.0)
    assert result.sd_ratio == pytest.approx(1.0)
    assert result.lower_endpoint_gap == 0
    assert result.upper_endpoint_gap == 0
    assert result.narrower_observed_support is False
    assert result.central_tendency_signal is False


def test_upper_end_truncation_is_visible_without_central_tendency_claim() -> None:
    """Preserving the lower endpoint while losing the upper endpoint is not central tendency."""
    automated = np.array([0, 1, 2, 3, 3], dtype=np.int64)
    reference = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    result = validation.paired_rating_range_evidence(
        automated,
        reference,
        category_count=5,
    )

    assert result.lower_endpoint_gap == 0
    assert result.upper_endpoint_gap == 1
    assert result.automated_span == 3
    assert result.reference_span == 4
    assert result.automated_distinct_categories == 4
    assert result.reference_distinct_categories == 5
    assert result.narrower_observed_support is True
    assert result.central_tendency_signal is False


def test_same_span_with_fewer_internal_categories_does_not_trigger_combined_signal() -> None:
    """Sparse internal category use alone cannot be mislabeled as narrower observed support."""
    automated = np.array([0, 0, 2, 4, 4], dtype=np.int64)
    reference = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    result = validation.paired_rating_range_evidence(
        automated,
        reference,
        category_count=5,
    )

    assert result.automated_span == result.reference_span == 4
    assert result.automated_distinct_categories == 3
    assert result.reference_distinct_categories == 5
    assert result.distinct_category_ratio == pytest.approx(0.6)
    assert result.narrower_observed_support is False
    assert result.central_tendency_signal is False


def test_degenerate_reference_returns_unavailable_relative_ratios() -> None:
    """A zero-range reference never creates NaN or infinity as relative evidence."""
    automated = np.array([1, 2, 2, 3], dtype=np.int64)
    reference = np.array([2, 2, 2, 2], dtype=np.int64)

    result = validation.paired_rating_range_evidence(
        automated,
        reference,
        category_count=5,
    )

    assert result.reference_span == 0
    assert result.reference_sd == 0.0
    assert result.span_ratio is None
    assert result.sd_ratio is None
    assert result.distinct_category_ratio == pytest.approx(3.0)
    for value in (
        result.automated_sd,
        result.reference_sd,
        result.distinct_category_ratio,
    ):
        assert math.isfinite(value)
    assert result.narrower_observed_support is False
    assert result.central_tendency_signal is False


@pytest.mark.parametrize(
    ("automated", "reference", "message"),
    (
        (np.zeros((2, 2), dtype=np.int64), np.zeros(4, dtype=np.int64), "1-D"),
        (np.array([0, 1, 2]), np.array([0, 1]), "length"),
        (np.array([0]), np.array([0]), "at least two"),
        (np.array([False, True]), np.array([0, 1]), "boolean"),
        (np.array([0.0, 1.5]), np.array([0, 1]), "integer"),
        (np.array([-1, 1]), np.array([0, 1]), "non-negative"),
        (np.array([0, 5]), np.array([0, 1]), "0..category_count-1"),
    ),
)
def test_public_wrapper_rejects_malformed_paired_labels(
    automated: np.ndarray,
    reference: np.ndarray,
    message: str,
) -> None:
    """Malformed paired labels fail before uint32 conversion can coerce them."""
    with pytest.raises(ValueError, match=message):
        validation.paired_rating_range_evidence(
            automated,
            reference,
            category_count=5,
        )


@pytest.mark.parametrize("category_count", (True, 1, 1.5, 1001))
def test_public_wrapper_rejects_invalid_category_count(category_count: object) -> None:
    """Category-count validation is exact and bounded like the existing judge API."""
    with pytest.raises(ValueError, match="category_count"):
        validation.paired_rating_range_evidence(
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            category_count=category_count,  # type: ignore[arg-type]
        )


def test_public_result_is_an_immutable_copy_of_rust_evidence() -> None:
    """Mutating caller arrays after evaluation cannot mutate the typed audit evidence."""
    automated = np.array([1, 1, 2, 3, 3], dtype=np.int64)
    reference = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    result = validation.paired_rating_range_evidence(
        automated,
        reference,
        category_count=5,
    )
    snapshot = result

    automated[:] = 4
    reference[:] = 0

    assert result == snapshot
    with pytest.raises((AttributeError, TypeError)):
        result.sample_size = 99  # type: ignore[misc]
