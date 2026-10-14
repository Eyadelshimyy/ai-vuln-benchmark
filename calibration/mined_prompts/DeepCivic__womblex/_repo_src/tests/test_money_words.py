"""Tests for financial values expressed in narrative structure.

The forms prose uses that a digits-and-symbol scan does not see: amounts
written out in words, thousands grouped with spaces, the Australian
``$US``/``$A`` symbol order, a true minus sign, and the drafting idiom that
writes one amount twice (``one million dollars ($1,000,000)``).

Each case is asserted as an exact ``Decimal``: the failure mode that matters
here is a value wrong by a power of ten, not a missing span — ``$10 000`` read
as ``$10`` is worse than ``$10 000`` missed.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from womblex.process.money import MoneyOptions, find_money, has_amount_signal
from womblex.process.money_words import find_worded_amounts, parse_number_words


def _one(text: str, options: MoneyOptions | None = None):
    spans = find_money(text, options)
    assert len(spans) == 1, f"expected exactly one span in {text!r}, got {spans}"
    return spans[0]


# ---------------------------------------------------------------------------
# Pattern 11 — worded amounts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "value", "currency"), [
    ("two million dollars", Decimal(2_000_000), "AUD"),
    ("five hundred thousand dollars", Decimal(500_000), "AUD"),
    ("twenty-five thousand dollars", Decimal(25_000), "AUD"),
    ("one million two hundred and fifty thousand dollars", Decimal(1_250_000), "AUD"),
    ("a million dollars", Decimal(1_000_000), "AUD"),
    ("one hundred and fifty dollars", Decimal(150), "AUD"),
    ("ten dollars", Decimal(10), "AUD"),
    ("fifty cents", Decimal("0.5"), "AUD"),
    ("two hundred Australian dollars", Decimal(200), "AUD"),
    ("five hundred euros", Decimal(500), "EUR"),
    ("half a million dollars", Decimal(500_000), "AUD"),
    ("one and a half million dollars", Decimal(1_500_000), "AUD"),
    ("three quarters of a million dollars", Decimal(750_000), "AUD"),
])
def test_worded_amounts(text, value, currency):
    span = _one(f"The department paid {text} in that year.")
    assert span.value == value
    assert span.currency == currency
    assert span.evidence == "p11"
    assert span.text == text


@pytest.mark.parametrize(("text", "value", "modifier"), [
    ("a sum not exceeding five hundred thousand dollars", Decimal(500_000),
     "not exceeding"),
    ("a fee not to exceed two thousand dollars", Decimal(2000), "not to exceed"),
    ("a penalty not exceeding $50 000", Decimal(50_000), "not exceeding"),
    ("grants up to a maximum of $100,000", Decimal(100_000), "up to a maximum of"),
])
def test_drafting_qualifiers_are_stored_separately(text, value, modifier):
    """A delegation limit or a penalty is written `a sum not exceeding …`. The
    bound qualifies the amount; it is never folded into the value."""
    span = _one(text)
    assert span.value == value
    assert span.modifier == modifier


def test_worded_amount_reports_its_magnitude_in_the_canonical_lane():
    assert _one("a grant of two million dollars").multiplier == "million"
    assert _one("a fee of fifty cents").multiplier == "cents"


@pytest.mark.parametrize("text", [
    # The measured shape in this corpus: a worded number that is not money.
    "There are more than one million Australians overseas at any time.",
    "Twenty-five projects were funded in the period.",
    "The five hundred participants were surveyed.",
    # `a` is filler, not one: `a dollar` must not become an amount.
    "The report gives a dollar figure for each project.",
    "Every dollar of expenditure is accounted for.",
])
def test_a_worded_number_without_a_currency_word_is_not_money(text):
    assert find_money(text) == []


def test_worded_phrase_does_not_cross_the_element_join():
    """`reassemble_narrative` joins elements with `\\n\\n`; a phrase spanning
    that join would bind two unrelated paragraphs into one amount."""
    assert find_money("payment of two million\n\ndollars owing") == []
    assert len(find_money("payment of two million\ndollars owing")) == 1


@pytest.mark.parametrize(("phrase", "value"), [
    ("two million", Decimal(2_000_000)),
    ("nineteen", Decimal(19)),
    ("ninety nine", Decimal(99)),
    ("one hundred and five", Decimal(105)),
    ("three billion", Decimal(3_000_000_000)),
])
def test_number_word_parser(phrase, value):
    parsed = parse_number_words(phrase)
    assert parsed is not None and parsed[0] == value


@pytest.mark.parametrize("phrase", ["a", "and", "of a", "million thousand million"])
def test_number_word_parser_declines_what_it_cannot_read(phrase):
    assert parse_number_words(phrase) is None


def test_zero_is_an_absence_not_an_amount():
    assert find_worded_amounts("zero dollars were spent") == []


# ---------------------------------------------------------------------------
# Grammar — phrases English does not write as one number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    # A range, not a sum: adding the endpoints reports thirty dollars.
    "quotes of between ten and twenty dollars",
    "quotes of ten and twenty dollars",
    "quotes of ten-twenty dollars",
    "quotes of ten–twenty dollars",
    "fifty and one hundred dollars",
    # A year, and two numbers run together.
    "the nineteen fifty dollars figure",
    "five six dollars",
])
def test_two_numbers_run_together_are_declined(text):
    """The failure this guards is a *wrong* value, not a missing one — the
    phrase parses arithmetically and reports money the document never wrote."""
    assert find_money(text) == []


@pytest.mark.parametrize("text", [
    "Amounts are in million dollars.",
    "The table shows thousand dollars.",
    "Figures are hundred dollars.",
])
def test_a_bare_scale_word_is_a_unit_declaration_not_an_amount(text):
    """`in million dollars` names the table's unit. Only an article or a number
    makes it an amount: `a million dollars` is one million."""
    assert find_money(text) == []


@pytest.mark.parametrize(("text", "value"), [
    ("a hundred dollars", Decimal(100)),
    ("fifteen hundred dollars", Decimal(1500)),
    ("one hundred fifty five dollars", Decimal(155)),
    ("two thousand and five dollars", Decimal(2005)),
    ("ninety nine cents", Decimal("0.99")),
])
def test_grammar_still_admits_what_english_writes(text, value):
    assert _one(f"a charge of {text} applies").value == value


# ---------------------------------------------------------------------------
# Restatement — `one million dollars ($1,000,000)`
# ---------------------------------------------------------------------------


def test_parenthesised_restatement_is_not_an_accounting_negative():
    """Drafting writes the amount twice. Read as a bracketed negative the
    sentence yielded −1,000,000; counted twice it doubled the money."""
    span = _one("The contract value is one million dollars ($1,000,000).")
    assert span.value == Decimal(1_000_000)
    assert span.negative is False
    assert span.text == "one million dollars"


def test_restatement_collapses_in_either_order():
    span = _one("A fee of $1,000,000 (one million dollars) applies.")
    assert span.value == Decimal(1_000_000)
    assert span.text == "$1,000,000"


def test_two_digit_amounts_are_not_a_restatement():
    """`5,000 (5,000)` in a financial statement is this year and last, not one
    amount written twice — collapsing it would discard a real negative."""
    spans = find_money("Total expenditure of $5,000 (5,000) for the statement period.")
    assert [s.value for s in spans] == [Decimal(5000), Decimal(-5000)]


# ---------------------------------------------------------------------------
# Space-grouped thousands (AGPS convention)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "value"), [
    ("$10 000", Decimal(10_000)),
    ("$50 000", Decimal(50_000)),
    ("$1 500 000", Decimal(1_500_000)),
    ("$10 000", Decimal(10_000)),      # no-break space from a PDF text layer
    ("$1 234 567.89", Decimal("1234567.89")),
])
def test_space_grouped_thousands(text, value):
    """Measured on an ACT regulatory notice: `Penalty: $10 000` was read as
    `$10`, storing a value wrong by 10³."""
    assert _one(f"Penalty: {text}, in the case of an individual.").value == value


def test_space_grouped_thousands_after_a_currency_word():
    assert _one("a payment of 500 000 dollars").value == Decimal(500_000)


def test_a_bare_space_grouped_number_is_still_not_money():
    assert find_money("The register lists 10 000 entries.") == []


def test_a_space_group_needs_exactly_three_digits():
    """Otherwise `$5` followed by a year binds two unrelated numbers."""
    assert _one("$5 2020 review").value == Decimal(5)


# ---------------------------------------------------------------------------
# Symbol forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "value", "currency"), [
    ("$US655.5m", Decimal(655500000), "USD"),
    ("$US617.7m", Decimal(617700000), "USD"),
    ("$A250,000", Decimal(250_000), "AUD"),
    ("$AUD1.2m", Decimal(1_200_000), "AUD"),
    ("$NZ40,000", Decimal(40_000), "NZD"),
    ("US$30,709,575.00", Decimal("30709575.00"), "USD"),
])
def test_symbol_then_letters_forms(text, value, currency):
    """The ANAO Major Projects Report writes foreign-military-sales case values
    `$US655.5m` throughout; all three were lost before."""
    span = _one(f"reducing the case value to {text}. The Amendment")
    assert span.value == value
    assert span.currency == currency


def test_cent_symbol_is_a_sub_unit():
    assert _one("A charge of 50¢ applies.").value == Decimal("0.5")


# ---------------------------------------------------------------------------
# Signs and brackets
# ---------------------------------------------------------------------------


def test_true_minus_sign_is_a_negative():
    span = _one("The balance was −$5.2 million at year end.")
    assert span.value == Decimal(-5_200_000)
    assert span.negative is True


def test_en_dash_is_still_a_range_not_a_negative():
    spans = find_money("Estimates of $10–20 million were provided.")
    assert [s.value for s in spans] == [Decimal(10_000_000), Decimal(20_000_000)]
    assert {s.range_role for s in spans} == {"lower", "upper"}


def test_symbol_outside_the_bracket_is_an_accounting_negative():
    span = _one("The adjustment is $(1,234.50) for the year.")
    assert span.value == Decimal("-1234.50")
    assert span.negative is True


# ---------------------------------------------------------------------------
# Guards that must survive the additions
# ---------------------------------------------------------------------------


def test_a_second_dotted_group_is_declined_not_guessed_at():
    """`$3.219.3m` is a real ANAO typo for `$3,219.3m`. Extracting the readable
    prefix reports three dollars for a $3.2 billion budget; repairing it would
    be a guess. Declining is the only honest outcome."""
    assert find_money("Total Approved Budget (Current) $3.219.3m") == []


@pytest.mark.parametrize("text", [
    "The road is 100m long.",
    "a 50m radius around the site",
    "Funding of $5 million covers 100m of cable.",
])
def test_measurements_are_still_blocked(text):
    assert all(s.value != Decimal(100) for s in find_money(text))
    assert "100m" not in [s.text for s in find_money(text)]


def test_has_amount_signal_pre_filter():
    assert has_amount_signal("paid $50")
    assert has_amount_signal("two million dollars")
    assert not has_amount_signal("no amount here")
