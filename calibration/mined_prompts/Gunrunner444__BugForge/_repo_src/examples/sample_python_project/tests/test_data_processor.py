"""Tests for data_processor — several are EXPECTED TO FAIL due to seeded bugs."""
import pytest
from src.data_processor import chunk_list, filter_positive, find_maximum, normalize


# --- Passing tests ---

def test_filter_positive_all_positive() -> None:
    assert filter_positive([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_find_maximum_normal() -> None:
    assert find_maximum([3.0, 1.0, 4.0, 1.0, 5.0]) == 5.0


def test_normalize_basic() -> None:
    result = normalize([0.0, 5.0, 10.0])
    assert result == [0.0, 0.5, 1.0]


def test_chunk_list_exact_multiple() -> None:
    assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


# --- FAILING tests (BUG: filter_positive includes zero) ---

def test_filter_positive_excludes_zero() -> None:
    """BUG: zero is included because the condition is >= 0."""
    assert 0.0 not in filter_positive([0.0, 1.0, -1.0])  # FAILS


def test_filter_positive_excludes_negatives() -> None:
    assert filter_positive([-3.0, -1.0, 0.0, 2.0]) == [2.0]  # FAILS — returns [0.0, 2.0]


# --- FAILING tests (BUG: find_maximum on empty list) ---

def test_find_maximum_empty_raises() -> None:
    """BUG: raises ValueError from max() rather than a descriptive error."""
    with pytest.raises(ValueError, match="empty"):
        find_maximum([])  # FAILS — wrong exception message


# --- FAILING tests (BUG: normalize identical values) ---

def test_normalize_identical_values() -> None:
    """BUG: ZeroDivisionError when all values are the same."""
    with pytest.raises(ZeroDivisionError):
        normalize([5.0, 5.0, 5.0])  # FAILS with ZeroDivisionError (wrong exception type)


# --- FAILING tests (BUG: chunk_list off-by-one) ---

def test_chunk_list_remainder() -> None:
    """BUG: the last chunk [5] is dropped."""
    result = chunk_list([1, 2, 3, 4, 5], 2)
    assert result == [[1, 2], [3, 4], [5]]  # FAILS — returns [[1, 2], [3, 4]]


def test_chunk_list_single_element() -> None:
    assert chunk_list([42], 1) == [[42]]  # FAILS — returns []


def test_chunk_invalid_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_list([1, 2, 3], 0)
