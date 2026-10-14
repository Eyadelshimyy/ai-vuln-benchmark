"""Fail-closed public input contracts for the Rust ATA gain binding."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm._ata_core_loader import ata_core


def _valid_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return one small valid request accepted by the compiled ATA boundary."""
    return (
        np.array([[1.0, 2.0, 4.0], [0.5, 2.0, 3.0]], dtype=np.float64),
        np.array([0, 2], dtype=np.int64),
        np.array([4.0, 3.0], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
    )


@pytest.mark.parametrize(
    ("argument_index", "replacement", "message"),
    [
        (0, np.ones((2, 3), dtype=np.float32), "information_matrix"),
        (1, [0, 2], "candidates"),
        (2, np.ones(2, dtype=np.float32), "target_info"),
        (3, np.ones(2, dtype=np.float32), "accumulated"),
    ],
)
def test_binding_normalizes_wrong_types_to_value_error(
    argument_index: int,
    replacement: object,
    message: str,
) -> None:
    """Wrong dtype and non-array inputs must not leak PyO3 ``TypeError``."""
    arguments = list(_valid_inputs())
    arguments[argument_index] = replacement

    with pytest.raises(ValueError, match=message):
        ata_core().target_information_gains(*arguments)


@pytest.mark.parametrize(
    ("argument_index", "replacement", "message"),
    [
        (0, _valid_inputs()[0][:, ::-1], "information_matrix"),
        (1, np.arange(4, dtype=np.int64)[::2], "candidates"),
        (2, np.arange(4, dtype=np.float64)[::2], "target_info"),
        (3, np.arange(4, dtype=np.float64)[::2], "accumulated"),
    ],
)
def test_binding_normalizes_non_contiguous_inputs_to_value_error(
    argument_index: int,
    replacement: np.ndarray,
    message: str,
) -> None:
    """Every borrowed array must fail with a stable package-owned error."""
    arguments = list(_valid_inputs())
    arguments[argument_index] = replacement

    with pytest.raises(ValueError, match=message):
        ata_core().target_information_gains(*arguments)


def test_candidate_count_is_bounded_before_candidate_slice_access() -> None:
    """An oversized strided candidate view must fail on count, not layout."""
    matrix, _, target, accumulated = _valid_inputs()
    oversized_non_contiguous = np.arange(8, dtype=np.int64)[::2]
    assert oversized_non_contiguous.size > matrix.shape[1]
    assert not oversized_non_contiguous.flags.c_contiguous

    with pytest.raises(ValueError, match="candidate count"):
        ata_core().target_information_gains(
            matrix,
            oversized_non_contiguous,
            target,
            accumulated,
        )


def test_empty_matrix_is_rejected_before_candidate_conversion() -> None:
    """The historical empty-matrix error has precedence over candidate layout."""
    empty_matrix = np.empty((0, 3), dtype=np.float64)
    oversized_non_contiguous = np.arange(8, dtype=np.int64)[::2]

    with pytest.raises(ValueError, match="non-empty"):
        ata_core().target_information_gains(
            empty_matrix,
            oversized_non_contiguous,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
