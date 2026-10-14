"""Tests for Calculator — several are EXPECTED TO FAIL due to seeded bugs."""
import pytest
from src.calculator import Calculator


@pytest.fixture
def calc() -> Calculator:
    return Calculator()


# --- Passing tests ---

def test_add(calc: Calculator) -> None:
    assert calc.add(2, 3) == 5.0


def test_subtract(calc: Calculator) -> None:
    assert calc.subtract(10, 4) == 6.0


def test_multiply(calc: Calculator) -> None:
    assert calc.multiply(3, 4) == 12.0


def test_divide_normal(calc: Calculator) -> None:
    assert calc.divide(10, 2) == 5.0


def test_power(calc: Calculator) -> None:
    assert calc.power(2, 8) == 256.0


# --- FAILING tests (BUG: no zero-division guard) ---

def test_divide_by_zero_raises(calc: Calculator) -> None:
    """BUG: should raise a controlled exception, not ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        calc.divide(10, 0)  # FAILS — no guard


# --- FAILING tests (BUG: average of empty returns 0) ---

def test_average_empty_raises(calc: Calculator) -> None:
    """BUG: returns 0 instead of raising ValueError."""
    with pytest.raises(ValueError):
        calc.average([])  # FAILS


def test_average_single(calc: Calculator) -> None:
    assert calc.average([42.0]) == 42.0


def test_average_multiple(calc: Calculator) -> None:
    assert calc.average([1.0, 2.0, 3.0]) == 2.0


# --- FAILING tests (BUG: off-by-one in factorial) ---

def test_factorial_zero(calc: Calculator) -> None:
    """BUG: range(1, 0) is empty so result stays 1, which is accidentally correct."""
    assert calc.factorial(0) == 1  # Passes accidentally


def test_factorial_one(calc: Calculator) -> None:
    """BUG: range(1, 1) is empty so result stays 1 — accidentally correct."""
    assert calc.factorial(1) == 1


def test_factorial_five(calc: Calculator) -> None:
    """BUG: range(1, 5) computes 4! = 24 instead of 5! = 120."""
    assert calc.factorial(5) == 120  # FAILS — returns 24


def test_factorial_negative_raises(calc: Calculator) -> None:
    with pytest.raises(ValueError):
        calc.factorial(-1)


# --- FAILING tests (BUG: percentage missing × 100) ---

def test_percentage(calc: Calculator) -> None:
    """BUG: returns 0.5 instead of 50.0."""
    assert calc.percentage(50, 100) == 50.0  # FAILS — returns 0.5


def test_percentage_zero_total_raises(calc: Calculator) -> None:
    with pytest.raises(ValueError):
        calc.percentage(10, 0)
