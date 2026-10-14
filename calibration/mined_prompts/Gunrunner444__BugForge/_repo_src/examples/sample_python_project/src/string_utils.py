"""
String utility functions — contains intentional bugs for BugForge to discover.

Seeded bugs:
  1. truncate()   — result is max_length + 3 chars (ellipsis not subtracted)
  2. count_words()— AttributeError when text is None
  3. first_char() — IndexError on empty string
"""
from __future__ import annotations


def truncate(text: str, max_length: int) -> str:
    """Shorten text to max_length characters, appending '...' if truncated.

    BUG: result is max_length + 3 characters, not max_length.
    Correct: return text[: max_length - 3] + "..." when len > max_length.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."  # BUG: exceeds max_length by 3


def count_words(text: str) -> int:
    """Return the number of whitespace-separated words in text.

    BUG: raises AttributeError when text is None.
    """
    return len(text.split())  # BUG: no None guard


def is_palindrome(text: str) -> bool:
    """Return True if text reads the same forward and backward. (correct)"""
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]


def first_char(text: str) -> str:
    """Return the first character of text.

    BUG: raises IndexError on an empty string.
    """
    return text[0]  # BUG: should guard for empty string


def repeat(text: str, times: int) -> str:
    """Return text repeated times times. (correct)"""
    return text * times
