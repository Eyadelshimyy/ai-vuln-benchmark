from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.rating_range as rating_range


class _HostileArrayProvider:
    """Fail if package validation executes caller-owned NumPy conversion."""

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("caller __array__ callback executed")


class _ArraySubclass(np.ndarray):
    """Caller-defined ndarray identity that must not cross the trust boundary."""


def _native_must_not_run() -> object:
    raise AssertionError("Rust rating-range core discovered before evidence admission")


def test_automated_array_provider_is_rejected_before_callback_or_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rating_range, "rating_range_core", _native_must_not_run)

    with pytest.raises(ValueError, match="automated must be a NumPy array"):
        rating_range.paired_rating_range_evidence(
            _HostileArrayProvider(),  # type: ignore[arg-type]
            np.array([0, 1], dtype=np.int64),
            category_count=2,
        )


def test_reference_array_provider_is_rejected_before_callback_or_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rating_range, "rating_range_core", _native_must_not_run)

    with pytest.raises(ValueError, match="reference must be a NumPy array"):
        rating_range.paired_rating_range_evidence(
            np.array([0, 1], dtype=np.int64),
            _HostileArrayProvider(),  # type: ignore[arg-type]
            category_count=2,
        )


def test_ndarray_subclass_is_rejected_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rating_range, "rating_range_core", _native_must_not_run)
    automated = np.array([0, 1], dtype=np.int64).view(_ArraySubclass)

    with pytest.raises(ValueError, match="automated must be a NumPy array"):
        rating_range.paired_rating_range_evidence(
            automated,
            np.array([0, 1], dtype=np.int64),
            category_count=2,
        )


@pytest.mark.parametrize("dtype", [np.int16, np.uint32, np.float32])
def test_exact_numeric_ndarrays_keep_existing_marshalling(dtype: type[np.generic]) -> None:
    values = np.array([0, 1, 1, 0], dtype=dtype)

    admitted = rating_range._rating_array(values, "automated", category_count=2)

    assert admitted.dtype == np.uint32
    assert admitted.flags.c_contiguous
    np.testing.assert_array_equal(admitted, np.array([0, 1, 1, 0], dtype=np.uint32))
