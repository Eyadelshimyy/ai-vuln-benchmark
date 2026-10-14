"""Tests for string_utils — several are EXPECTED TO FAIL due to seeded bugs."""
import pytest
from src.string_utils import count_words, first_char, is_palindrome, repeat, truncate


# --- Passing tests ---

def test_is_palindrome_true() -> None:
    assert is_palindrome("racecar") is True


def test_is_palindrome_false() -> None:
    assert is_palindrome("hello") is False


def test_is_palindrome_with_spaces() -> None:
    assert is_palindrome("a man a plan a canal panama") is True


def test_repeat() -> None:
    assert repeat("ab", 3) == "ababab"


def test_count_words_normal() -> None:
    assert count_words("hello world") == 2


# --- FAILING tests (BUG: truncate result length is max_length + 3) ---

def test_truncate_short_string_unchanged() -> None:
    assert truncate("hi", 10) == "hi"


def test_truncate_respects_max_length() -> None:
    """BUG: result is 13 chars instead of max_length=10."""
    result = truncate("Hello, world!", 10)
    assert len(result) <= 10  # FAILS — len is 13


def test_truncate_adds_ellipsis() -> None:
    result = truncate("Hello, world!", 10)
    assert result.endswith("...")


# --- FAILING tests (BUG: count_words raises on None) ---

def test_count_words_none_raises() -> None:
    """BUG: raises AttributeError instead of a controlled error."""
    with pytest.raises((TypeError, AttributeError)):
        count_words(None)  # type: ignore[arg-type]  # FAILS with AttributeError


# --- FAILING tests (BUG: first_char raises on empty string) ---

def test_first_char_normal() -> None:
    assert first_char("abc") == "a"


def test_first_char_empty_raises() -> None:
    """BUG: raises IndexError instead of a controlled ValueError."""
    with pytest.raises(ValueError):
        first_char("")  # FAILS with IndexError
