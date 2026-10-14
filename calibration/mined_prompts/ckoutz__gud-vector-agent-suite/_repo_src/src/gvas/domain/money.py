"""Exact money formatting.

Prices are carried as integer minor units everywhere, so rendering them must
stay integral too: ``amount_minor / 100`` would move an owner's price through a
binary float. How many minor units a major unit holds is a property of the
currency (USD has 100, JPY has 1), so a formatter cannot assume two decimals
either. Only the currencies this deployment can price are listed; an unknown
code is refused rather than formatted with a guessed exponent.
"""

from typing import Final

#: ISO 4217 minor-unit exponents for the currencies the pilot supports.
MINOR_UNIT_DIGITS: Final = {"USD": 2}


class UnsupportedCurrencyError(ValueError):
    """Raised for a currency whose minor-unit exponent is not known here."""


def supported_currencies() -> tuple[str, ...]:
    return tuple(sorted(MINOR_UNIT_DIGITS))


def minor_unit_digits(currency: str) -> int:
    code = currency.upper()
    digits = MINOR_UNIT_DIGITS.get(code)
    if digits is None:
        raise UnsupportedCurrencyError(
            f"{code} is not supported; supported currencies: {', '.join(supported_currencies())}"
        )
    return digits


def minor_unit_scale(currency: str) -> int:
    """How many minor units make one major unit of ``currency``."""

    scale: int = 10 ** minor_unit_digits(currency)
    return scale


def format_money(amount_minor: int, currency: str) -> str:
    """Render minor units exactly, using integer arithmetic only."""

    code = currency.upper()
    digits = minor_unit_digits(code)
    sign = "-" if amount_minor < 0 else ""
    if digits == 0:
        return f"{code} {sign}{abs(amount_minor)}"
    major, minor = divmod(abs(amount_minor), minor_unit_scale(code))
    return f"{code} {sign}{major}.{minor:0{digits}d}"
