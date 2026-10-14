"""Fail-first resource-safety contracts for bifactor advertised shapes."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from fast_mlsirm import assess_bifactor_scoreability


class _GuardedShape:
    """Finite shape probe that fails if validation requests one value too many."""

    def __init__(self, values: tuple[int, ...], request_limit: int) -> None:
        self._values = values
        self._request_limit = request_limit
        self.requests = 0

    def __iter__(self) -> Iterator[int]:
        for value in self._values:
            self.requests += 1
            if self.requests > self._request_limit:
                raise AssertionError("shape validation requested too many dimensions")
            yield value
        self.requests += 1
        if self.requests > self._request_limit:
            raise AssertionError("shape validation requested too many dimensions")


class _AdvertisedArrayLike:
    """Array-like probe whose payload must never be materialized after bad shape."""

    def __init__(self, shape: object) -> None:
        self.shape = shape

    def __array__(self, dtype=None, copy=None):  # noqa: ANN001, ANN204
        del dtype, copy
        raise AssertionError("invalid advertised shape must fail before array conversion")


def _valid_loadings() -> np.ndarray:
    """Return a small standardized strict-bifactor loading matrix."""
    return np.asarray(
        [
            [0.70, 0.40, 0.00],
            [0.70, 0.30, 0.00],
            [0.70, 0.00, 0.50],
            [0.70, 0.00, 0.60],
        ],
        dtype=np.float64,
    )


def _valid_uniquenesses() -> np.ndarray:
    """Return residual variances matching :func:`_valid_loadings`."""
    return np.asarray([0.35, 0.42, 0.26, 0.15], dtype=np.float64)


def test_loadings_shape_rejects_after_third_dimension_without_fourth_request() -> None:
    """A matrix shape needs at most three requests to prove it is not two-dimensional."""
    shape = _GuardedShape((4, 3, 2), request_limit=3)

    with pytest.raises(ValueError, match="2-D item-by-factor matrix"):
        assess_bifactor_scoreability(
            _AdvertisedArrayLike(shape),
            _valid_uniquenesses(),
            zero_tolerance=0.0,
        )

    assert shape.requests == 3


def test_uniqueness_shape_rejects_after_second_dimension_without_third_request() -> None:
    """A vector shape needs at most two requests to prove it is not one-dimensional."""
    shape = _GuardedShape((4, 1), request_limit=2)

    with pytest.raises(ValueError, match="1-D item vector"):
        assess_bifactor_scoreability(
            _valid_loadings(),
            _AdvertisedArrayLike(shape),
            zero_tolerance=0.0,
        )

    assert shape.requests == 2


class _FailingShape:
    """Shape iterable that raises caller-controlled text during inspection."""

    def __iter__(self) -> Iterator[int]:
        yield 4
        raise RuntimeError("CALLER_CONTROLLED_SHAPE_SENTINEL")


def test_shape_iteration_failure_is_stable_and_non_reflective() -> None:
    """Ordinary shape-iteration failures become bounded package errors."""
    with pytest.raises(ValueError) as exc_info:
        assess_bifactor_scoreability(
            _AdvertisedArrayLike(_FailingShape()),
            _valid_uniquenesses(),
            zero_tolerance=0.0,
        )

    message = str(exc_info.value)
    assert "2-D item-by-factor matrix" in message
    assert "CALLER_CONTROLLED_SHAPE_SENTINEL" not in message


class _InterruptingShape:
    """Shape iterable that raises a process-control signal."""

    def __iter__(self) -> Iterator[int]:
        yield 4
        raise KeyboardInterrupt


def test_shape_inspection_preserves_process_control_signals() -> None:
    """Bounded validation must not swallow process-control signals."""
    with pytest.raises(KeyboardInterrupt):
        assess_bifactor_scoreability(
            _AdvertisedArrayLike(_InterruptingShape()),
            _valid_uniquenesses(),
            zero_tolerance=0.0,
        )
