"""
Data-processing helpers — contains intentional bugs for BugForge to discover.

Seeded bugs:
  1. filter_positive() — includes zero (uses >= 0 instead of > 0)
  2. find_maximum()    — raises ValueError on an empty list
  3. normalize()       — ZeroDivisionError when all values are identical
  4. chunk_list()      — off-by-one, last chunk is silently dropped
"""
from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def filter_positive(numbers: list[float]) -> list[float]:
    """Return only the positive numbers (strictly > 0).

    BUG: uses >= 0, which lets zero through.
    """
    return [n for n in numbers if n >= 0]  # BUG: should be > 0


def find_maximum(data: list[float]) -> float:
    """Return the largest value in data.

    BUG: raises ValueError on an empty list instead of a descriptive error.
    """
    return max(data)  # BUG: no empty-list guard


def normalize(values: list[float]) -> list[float]:
    """Return values scaled to [0, 1].

    BUG: ZeroDivisionError when all values are the same (max == min).
    """
    min_val = min(values)
    max_val = max(values)
    return [(v - min_val) / (max_val - min_val) for v in values]  # BUG: division by zero


def chunk_list(lst: list[T], chunk_size: int) -> list[list[T]]:
    """Split lst into chunks of chunk_size.

    BUG: off-by-one in the range stop — the last chunk is dropped when
    len(lst) is not an exact multiple of chunk_size.
    Correct: range(0, len(lst), chunk_size).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks: list[list[T]] = []
    for i in range(0, len(lst) - chunk_size + 1, chunk_size):  # BUG: wrong stop
        chunks.append(lst[i : i + chunk_size])
    return chunks
