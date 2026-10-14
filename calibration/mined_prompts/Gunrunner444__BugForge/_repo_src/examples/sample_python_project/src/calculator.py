"""
Calculator module — contains intentional bugs for BugForge to discover.

Seeded bugs:
  1. divide()        — no zero-division guard → ZeroDivisionError
  2. average()       — empty-list returns 0 instead of raising ValueError
  3. factorial()     — off-by-one in range (range(1, n) instead of range(1, n+1))
  4. percentage()    — missing × 100 in the formula
"""
from __future__ import annotations


class Calculator:
    def add(self, a: float, b: float) -> float:
        """Return the sum of a and b."""
        return a + b

    def subtract(self, a: float, b: float) -> float:
        """Return a minus b."""
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """Return the product of a and b."""
        return a * b

    def divide(self, a: float, b: float) -> float:
        """Return a divided by b.

        BUG: raises ZeroDivisionError when b == 0 instead of a controlled error.
        """
        return a / b  # BUG: no zero-division guard

    def average(self, numbers: list[float]) -> float:
        """Return the arithmetic mean of numbers.

        BUG: returns 0.0 for an empty list instead of raising ValueError.
        """
        if not numbers:
            return 0.0  # BUG: should raise ValueError("average of empty sequence")
        return sum(numbers) / len(numbers)

    def factorial(self, n: int) -> int:
        """Return n!

        BUG: off-by-one — range(1, n) misses the final factor n.
        Correct would be range(1, n + 1).
        """
        if n < 0:
            raise ValueError("factorial is undefined for negative numbers")
        result = 1
        for i in range(1, n):  # BUG: should be range(1, n + 1)
            result *= i
        return result

    def power(self, base: float, exp: int) -> float:
        """Return base raised to exp. (correct)"""
        return base**exp

    def percentage(self, value: float, total: float) -> float:
        """Return what percentage value is of total.

        BUG: missing multiplication by 100; returns a fraction, not a percentage.
        """
        if total == 0:
            raise ValueError("total cannot be zero")
        return value / total  # BUG: should be (value / total) * 100
