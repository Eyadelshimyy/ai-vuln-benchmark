"""Money is rendered from integer minor units, exactly."""

import pytest

from gvas.domain.money import UnsupportedCurrencyError, format_money, minor_unit_digits

LARGE_MINOR = 99999999999999999


@pytest.mark.parametrize(
    ("minor", "expected"),
    [(0, "USD 0.00"), (5, "USD 0.05"), (12500, "USD 125.00"), (12550, "USD 125.50")],
)
def test_minor_units_render_without_scaling_error(minor: int, expected: str) -> None:
    assert format_money(minor, "USD") == expected


def test_a_large_amount_keeps_every_digit_a_float_would_lose() -> None:
    """``minor / 100`` rounds this up to a whole million; the owner's price is exact."""

    assert format_money(LARGE_MINOR, "USD") == "USD 999999999999999.99"
    assert f"{LARGE_MINOR / 100:.2f}" == "1000000000000000.00"


def test_a_currency_with_unknown_minor_units_is_refused_not_guessed() -> None:
    """JPY has no minor unit, so a two-decimal assumption would misprice it 100x."""

    with pytest.raises(UnsupportedCurrencyError):
        format_money(20000, "JPY")
    with pytest.raises(UnsupportedCurrencyError):
        minor_unit_digits("EUR")


def test_currency_codes_are_case_insensitive() -> None:
    assert format_money(12500, "usd") == "USD 125.00"
