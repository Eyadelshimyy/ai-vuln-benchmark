"""Tests for the money op's pure core.

The extraction-pattern and false-positive tables in ``docs/money-extraction.md`` are the
specification; each row below is one of their examples. Values are asserted as
``Decimal`` — a silent 10⁶ error on a magnitude expression is the failure mode
that matters here, not a missing span.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from womblex.process.money import (
    MoneyOptions,
    blocked_spans,
    find_money,
    parse_number,
    resolve_iso,
)
from womblex.process.money_columns import (
    ColumnOptions,
    cell_amount,
    classify_column,
    extract_column,
    fold_header_continuation,
    header_currency,
    header_scale,
    is_null_marker,
)


def _one(text: str, options: MoneyOptions | None = None):
    spans = find_money(text, options)
    assert len(spans) == 1, f"expected exactly one span in {text!r}, got {spans}"
    return spans[0]


# ---------------------------------------------------------------------------
# Extraction patterns (docs/money-extraction.md table)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "value", "currency", "evidence"), [
    ("$100", Decimal(100), "AUD", "p1"),
    ("-$250", Decimal(-250), "AUD", "p1"),
    ("A$500", Decimal(500), "AUD", "p1"),
    ("AU$5 million", Decimal(5000000), "AUD", "p6"),
    ("AUD 100", Decimal(100), "AUD", "p2"),
    ("USD 50", Decimal(50), "USD", "p2"),
    ("EUR 1000", Decimal(1000), "EUR", "p2"),
    ("100 AUD", Decimal(100), "AUD", "p3"),
    ("500 USD", Decimal(500), "USD", "p3"),
    ("100 dollars", Decimal(100), "AUD", "p4"),
    ("250 Australian dollars", Decimal(250), "AUD", "p4"),
    ("50 cents", Decimal("0.5"), "AUD", "p4"),
    ("100$", Decimal(100), "AUD", "p5"),
    ("50€", Decimal(50), "EUR", "p5"),
    ("$5 million", Decimal(5000000), "AUD", "p6"),
    ("AUD 12 billion", Decimal(12000000000), "AUD", "p6"),
    ("$4.2bn", Decimal(4200000000), "AUD", "p6"),
    ("$500k", Decimal(500000), "AUD", "p6"),
    ("$78.7bn", Decimal(78700000000), "AUD", "p6"),
    ("$33.1 million", Decimal(33100000), "AUD", "p6"),
    ("AUD$21.9 million", Decimal(21900000), "AUD", "p6"),
])
def test_pattern_table(text, value, currency, evidence):
    span = _one(text)
    assert span.value == value
    assert span.currency == currency
    assert span.evidence == evidence
    assert span.text in text


def test_original_text_is_preserved():
    span = _one("Funding allocation was $5.2 million.")
    assert span.text == "$5.2 million"
    assert span.value == Decimal(5200000)
    assert span.multiplier == "million"


@pytest.mark.parametrize(("text", "canonical"), [
    ("$1.2m", "million"), ("$1.2 mn", "million"), ("$1.2 million", "million"),
    ("$1.2bn", "billion"), ("$1.2 b", "billion"), ("$1.2 billion", "billion"),
    ("$5k", "thousand"), ("$5 thousand", "thousand"),
    ("$1.5t", "trillion"), ("$1.5 trillion", "trillion"),
])
def test_multiplier_is_one_lane_per_magnitude(text, canonical):
    """Recognition is broad; what gets written is not.

    Persisting the document's own token split a single magnitude across
    `m` / `mn` / `million` in `money_spans.multiplier`, so a downstream
    group-by fragmented. The column path already emitted the long form.
    """
    assert _one(text).multiplier == canonical


@pytest.mark.parametrize(("text", "value", "canonical"), [
    ("USD 6.6Mn", Decimal(6600000), "million"),
    ("USD 6.6Bn", Decimal(6600000000), "billion"),
    ("USD 10K", Decimal(10000), "thousand"),
    ("USD 6.6M", Decimal(6600000), "million"),
    ("6.6Mn USD", Decimal(6600000), "million"),
    ("10K USD", Decimal(10000), "thousand"),
])
def test_capitalised_scale_reads_behind_an_iso_code(text, value, canonical):
    """A capitalised scale token must survive the case-sensitive ISO patterns.

    p2 / p3 are compiled without `re.IGNORECASE` so `[A-Z]{3}` cannot match a
    lowercase word, and that reached the shared scale tail: `USD 6.6Mn` matched
    only `USD 6.6` and stored 6.6 at 0.99 confidence — silently wrong by 10**6,
    the failure class typed ground truth exists to catch — while `6.6Mn USD`
    missed outright. The symbol patterns were never affected.
    """
    span = _one(text)
    assert span.value == value
    assert span.multiplier == canonical


def test_scale_letter_does_not_eat_an_iso_codes_first_character():
    """The guard that keeps the case-insensitive scale tail honest.

    `MXN` opens with a magnitude letter; reading it as `M` + `XN` would turn
    100 Mexican pesos into 100 million of nothing.
    """
    span = _one("The invoice was paid in 100 MXN for the transfer fee.")
    assert (span.value, span.currency, span.multiplier) == (Decimal(100), "MXN", None)


def test_values_are_exact_decimals_not_floats():
    span = _one("$0.10")
    assert span.value == Decimal("0.10")
    assert isinstance(span.value, Decimal)
    total = sum((_one(f"${i}.10").value for i in range(3)), Decimal(0))
    assert total == Decimal("3.30")  # float would drift


# ---------------------------------------------------------------------------
# Magnitude gate — the `100m road` rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["100m road", "50m radius", "20m hose", "100 km", "5 ha"])
def test_bare_letter_scale_needs_a_currency_marker(text):
    assert find_money(text) == []


def test_bare_letter_scale_licensed_by_symbol():
    assert _one("$100m").value == Decimal(100000000)


# ---------------------------------------------------------------------------
# Ranges and qualifiers
# ---------------------------------------------------------------------------


def test_range_keeps_both_endpoints_and_shares_scale():
    lo, hi = find_money("$10–20 million")
    assert (lo.value, hi.value) == (Decimal(10000000), Decimal(20000000))
    assert lo.range_group == hi.range_group
    assert (lo.range_role, hi.range_role) == ("lower", "upper")


def test_range_with_symbol_on_both_endpoints():
    lo, hi = find_money("$100-$150")
    assert (lo.value, hi.value) == (Decimal(100), Decimal(150))


def test_between_and_range():
    lo, hi = find_money("grants of between $5,000 and $10,000 were made")
    assert (lo.value, hi.value) == (Decimal(5000), Decimal(10000))


def test_range_needs_evidence_on_an_endpoint():
    assert find_money("10-20 million") == []


@pytest.mark.parametrize(("text", "modifier"), [
    ("about $100", "about"),
    ("~$50", "~"),
    ("up to $50,000", "up to"),
    ("approximately $500", "approximately"),
    ("at least $1,000", "at least"),
])
def test_qualifier_stored_separately_never_folded_into_value(text, modifier):
    span = _one(text)
    assert span.modifier == modifier
    assert span.value > 0


# ---------------------------------------------------------------------------
# Accounting negatives — gated
# ---------------------------------------------------------------------------


def test_bracketed_amount_with_symbol_is_negative():
    span = _one("($100)")
    assert span.value == Decimal(-100)
    assert span.negative is True
    assert span.evidence == "p9"


def test_bracketed_amount_after_iso_code_is_negative():
    span = _one("AUD (500)")
    assert span.value == Decimal(-500)


def test_bracketed_amount_under_accounting_context():
    span = _one("Total expenditure was (6,550.1) for the year")
    assert span.value == Decimal("-6550.1")
    assert span.confidence < 0.9  # context-gated, not marker-gated


@pytest.mark.parametrize("text", [
    "s167(1)", "(02) 6203 7300", "(2018)",
    "Total expenditure in (2018) was reported",
])
def test_unanchored_bracketed_numbers_are_not_amounts(text):
    assert find_money(text) == []


# ---------------------------------------------------------------------------
# Australian false positives (docs/money-extraction.md table)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "01/07/2025", "2025-07-01", "1 July 2025", "2024-25",
    "10:30", "14:45", "0930 hrs",
    "02 6123 4567", "0412 345 678", "1800 123 456", "(02) 6203 7300",
    "12 345 678 901", "123 456 789",
    "Lot 5", "DP12345", "SP4567",
    "Section 10", "Clause 12", "Schedule 3", "Division 2",
    "INC123456", "F2024/12345", "IR000456",
    "50m", "100 km", "20 kg", "5 ha", "10 MW", "250 ML", "40°C",
    "10%", "15.5%", "100 percent",
])
def test_false_positive_classes_are_rejected(text):
    assert find_money(text) == []


def test_postcode_rejected_only_with_address_context():
    assert find_money("PO Box 100, Canberra ACT 2600",
                      MoneyOptions(implicit_context=True, min_confidence=0.3)) == []
    assert any(s.value == Decimal(2600) for s in find_money(
        "the fee is 2600", MoneyOptions(implicit_context=True, min_confidence=0.3)))


def test_blocked_spans_skip_measurements_behind_a_currency_marker():
    names = {n for _, _, n in blocked_spans("$100 m allocated")}
    assert "measurement" not in names
    assert "measurement" in {n for _, _, n in blocked_spans("100 m of pipe")}


# ---------------------------------------------------------------------------
# Currency model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["ABC 500", "XYZ 100", "500 ABC"])
def test_three_uppercase_letters_are_not_a_currency(text):
    assert find_money(text) == []


def test_iso_membership():
    assert resolve_iso("AUD") == "AUD"
    assert resolve_iso("RMB") == "CNY"
    assert resolve_iso("ABC") is None


def test_tier3_currency_needs_context_and_scores_lower():
    """Several ISO codes are ordinary English words in caps. Tier 3 is admitted
    only when the surroundings reinforce it — otherwise `TOP 10 projects` is
    ten Tongan paʻanga."""
    tier1 = _one("AUD 100")
    tier3 = _one("a grant of PGK 100 was paid")
    assert tier3.currency == "PGK"
    assert tier3.confidence < tier1.confidence
    assert find_money("PGK 100") == []          # bare tier 3: no reinforcement


@pytest.mark.parametrize("text", [
    "TOP 10 projects were funded",
    "ALL 25 projects reported on time",
    "TRY 3 times before escalating",
    "CUP 4 finals",
])
def test_english_words_that_are_iso_codes_are_not_currencies(text):
    assert find_money(text) == []


def test_ambiguous_currency_word_leaves_currency_unresolved():
    span = _one("500 pesos")
    assert span.currency is None
    assert span.currency_source == "word"


# ---------------------------------------------------------------------------
# Number format / locale
# ---------------------------------------------------------------------------


def test_australian_number_forms():
    assert parse_number("1,000.50") == Decimal("1000.50")
    assert parse_number(".50") == Decimal("0.50")
    assert parse_number("100") == Decimal(100)


def test_comma_decimal_not_read_as_australian():
    assert find_money("€1.000,50") == []


@pytest.mark.parametrize("text", [
    "1.000,50 EUR",       # ISO suffix (p3)
    "1.000,50€",          # symbol suffix (p5)
    "1.000,50 dollars",   # currency word (p4)
    "Paid 1.234,56 EUR to the vendor.",
])
def test_continental_tail_is_not_extracted_as_its_own_amount(text):
    """A declined continental number must not leak its decimal tail.

    Declining the candidate that starts at the run is not sufficient: the tail
    (`,56 EUR`) is itself a complete match for the suffix patterns, so this
    used to return `56 EUR` — wrong by 10³. Only the *prefix*-marker forms were
    ever safe, which is why `€1.000,50` above passed while these did not.
    """
    assert find_money(text) == []


def test_malformed_thousands_group_blocks_the_whole_run():
    assert find_money("$1,23") == []
    assert find_money("1,23 dollars") == []


def test_international_mode_accepts_comma_decimals():
    opts = MoneyOptions(international_numbers=True)
    values = [s.value for s in find_money("€1.000,50 and $10.000.000,00", opts)]
    assert values == [Decimal("1000.50"), Decimal("10000000.00")]


# ---------------------------------------------------------------------------
# Implicit financial context (pattern 10) — off by default
# ---------------------------------------------------------------------------


def test_implicit_context_off_by_default():
    assert find_money("The estimated cost is 250.") == []


def test_implicit_context_when_enabled_scores_lowest():
    opts = MoneyOptions(implicit_context=True, min_confidence=0.3)
    span = _one("The estimated cost is 250.", opts)
    assert span.value == Decimal(250)
    assert span.evidence == "p10"
    assert span.confidence < 0.5


def test_implicit_context_does_not_license_a_bare_letter_scale():
    # `100k` under a trigger word is dropped entirely rather than read as 100:
    # the letter is unlicensed, and a bare 100 would be the wrong answer.
    opts = MoneyOptions(implicit_context=True, min_confidence=0.3)
    assert find_money("the fee of 100k", opts) == []


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------


def test_longer_match_wins_at_equal_priority():
    span = _one("AUD$21.9 million")  # not `$21.9`
    assert span.value == Decimal(21900000)


def test_paragraph_extraction_end_to_end():
    para = (
        "The Department received an appropriation of $33.1 million in 2024-25, "
        "of which approximately $4.2m was allocated under s167(1). Contact the "
        "registry on (02) 6203 7300 or write to PO Box 100, Canberra ACT 2600."
    )
    spans = find_money(para)
    assert [s.value for s in spans] == [Decimal(33100000), Decimal(4200000)]
    assert spans[1].modifier == "approximately"
    assert para[spans[0].start:spans[0].end] == spans[0].text


# ---------------------------------------------------------------------------
# Boundaries: newlines, malformed numbers, compact ISO forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "value", "currency"), [
    ("USD100", Decimal(100), "USD"),
    ("AUD1500", Decimal(1500), "AUD"),
    ("EUR2000", Decimal(2000), "EUR"),
])
def test_compact_iso_form_is_not_an_incident_reference(text, value, currency):
    """`USD100` looks like a reference number to the FP filter; the ISO
    membership check is what tells the two apart."""
    span = _one(text)
    assert (span.value, span.currency) == (value, currency)


def test_no_pattern_crosses_the_element_join():
    """`\\n\\n` joins two elements in the reassembled narrative. A range binding
    across it would fabricate a relationship between unrelated paragraphs."""
    spans = find_money("Payment of $100\n\n-$200 was made")
    assert [s.value for s in spans] == [Decimal(100), Decimal(-200)]
    assert all(s.range_group is None for s in spans)

    spans = find_money("Line one $100\n\nto $200 line two")
    assert [s.value for s in spans] == [Decimal(100), Decimal(200)]
    assert all(s.range_role is None for s in spans)


def test_magnitude_survives_a_single_line_wrap():
    # PDF text layers wrap mid-phrase; one newline is still one paragraph.
    assert _one("allocated $5\nmillion to the program").value == Decimal(5000000)
    # But two are an element boundary, so the scale does not reach across.
    assert _one("allocated $5\n\nmillion to the program").value == Decimal(5)


def test_malformed_thousands_group_is_declined():
    # `$1,23` would otherwise report one dollar — wrong by two orders.
    assert find_money("$1,23") == []
    assert _one("$1,234").value == Decimal(1234)


def test_large_document_stays_linear():
    """Guards the interval index: a linear rescan here was 3s on 300 KB."""
    import time

    unit = "The Department paid $1,234.56 on 1 July 2025 under s167(1). "
    text = unit * 3000  # ~180 KB
    t0 = time.perf_counter()
    spans = find_money(text)
    elapsed = time.perf_counter() - t0
    assert len(spans) == 3000
    assert elapsed < 5.0, f"quadratic regression: {elapsed:.1f}s on {len(text)} chars"


def test_comma_dense_ocr_noise_does_not_backtrack():
    import time

    t0 = time.perf_counter()
    find_money("$" + ",".join(["1"] * 4000))
    assert time.perf_counter() - t0 < 2.0


# ---------------------------------------------------------------------------
# Column-evidenced path
# ---------------------------------------------------------------------------


def test_number_format_is_definitive():
    verdict = classify_column("Value (AUD)", ["50000", "125000", "7500"],
                              number_format="$#,##0.00")
    assert verdict.is_money
    assert verdict.evidence == "number_format"
    assert verdict.currency == "AUD"


def test_money_header_plus_numeric_cells():
    verdict = classify_column("Value", ["50000", "125000", "7500"],
                              number_format="#,##0.00")
    assert verdict.is_money
    assert verdict.evidence == "header+numeric"


def test_numeric_cells_alone_never_promote_a_column():
    verdict = classify_column("Serial", ["50000", "125000", "7500"])
    assert not verdict.is_money
    assert verdict.verdict == "insufficient"


def test_no_header_leaves_bare_cells_alone():
    verdict = classify_column("", ["1,500", "2,700", "300"])
    assert not verdict.is_money


@pytest.mark.parametrize("header", ["Postcode", "ABN", "Count", "Phone", "Year",
                                    "Percent", "Rate", "Latitude", "FTE", "Age"])
def test_veto_terms_suppress_numeric_columns(header):
    verdict = classify_column(header, ["1,000", "2,000", "3,000"])
    assert verdict.verdict == "vetoed"
    assert verdict.veto_term is not None


def test_explicit_header_currency_outranks_an_incidental_veto_term():
    """DocLayNet `dense_text_548`: `Grant Date Fair Value ... ($)` is a money
    column that happens to contain the word "date". Vetoing on that loses every
    amount beneath it — the `($)` is the header describing itself."""
    values = ["8,453,500", "568,690", "345,825", "445,730", "338,140"]
    verdict = classify_column("Grant Date Fair Value of Stock and Option Awards ($)", values)
    assert verdict.is_money
    assert verdict.veto_term == "date"  # recorded, but overridden — still auditable
    assert len(extract_column(values, verdict)) == 5


def test_unit_marker_columns_are_not_money():
    """The same page carries `Threshold ($)` and `Threshold (#)` — dollars and
    unit counts, distinguished only by the marker. A tokeniser that drops `#`
    reads the count column as money and invents five amounts."""
    counts = ["320,833", "21,583", "13,125", "16,917", "12,833"]
    dollars = ["32,031", "9,180", "5,439", "9,287", "5,625"]
    assert classify_column("Threshold (#)", counts).verdict == "vetoed"
    assert classify_column("Target (#)", counts).verdict == "vetoed"
    assert classify_column("Threshold ($)", dollars).is_money
    assert classify_column("Target ($)", dollars).is_money


def test_veto_still_wins_without_an_explicit_currency_marker():
    # Count columns on the same page carry `(#)`, not `($)`, and stay vetoed.
    assert classify_column(
        "All Other Option Awards: Number of Securities Underlying Options (#)",
        ["550,000", "37,000", "22,500"]).verdict == "vetoed"
    assert classify_column("Date", ["20,000", "30,000", "40,000"]).verdict == "vetoed"


def test_veto_matching_is_whole_word():
    verdict = classify_column("Average Cost", ["1,200", "1,300", "1,100"])
    assert verdict.is_money, "`age` must not veto `Average Cost`"


def test_percent_number_format_vetoes():
    verdict = classify_column("Share", ["10", "20", "30"], number_format="0.00%")
    assert verdict.verdict == "vetoed"


def test_null_markers_excluded_from_the_numeric_fraction():
    # The DocLayNet compensation-table case: em-dashes are absent values, and
    # counting them as non-numeric would suppress a genuine money column.
    values = ["1,000", "—", "2,000", "–", "3,000", "-"]
    verdict = classify_column("Threshold ($)", values)
    assert verdict.numeric_fraction == 1.0
    assert verdict.is_money
    assert len(extract_column(values, verdict)) == 3


@pytest.mark.parametrize("marker", ["—", "–", "-", "n/a", "nil", "none", ""])
def test_null_marker_recognition(marker):
    assert is_null_marker(marker)


def test_header_supplies_the_column_scale():
    values = ["1.5", "2.7", "0.3"]
    verdict = classify_column("Approved Budget $m", values)
    assert verdict.scale == "million"
    assert [v for _, v, _ in extract_column(values, verdict)] == [
        Decimal("1500000.0"), Decimal("2700000.0"), Decimal("300000.0")]


@pytest.mark.parametrize("header", [
    "Budget 2000", "Grants over $10,000", "Threshold $5,000", "Payments 2019-20",
])
def test_a_number_in_the_header_is_not_a_thousands_scale(header):
    """`Grants over $10,000` must not declare a thousands scale off the `000`
    inside its own number — that multiplies every cell below it by 1,000."""
    assert header_scale(header) is None
    values = ["1,200", "3,400", "2,750"]
    verdict = classify_column(header, values)
    if verdict.is_money:
        assert [v for _, v, _ in extract_column(values, verdict)] == [
            Decimal(1200), Decimal(3400), Decimal(2750)]


@pytest.mark.parametrize(("header", "scale"), [
    ("Expenditure $'000", "thousand"), ("Value '000", "thousand"),
    ("In $000s", "thousand"), ("Amount ($m)", "million"),
    ("Cost ($ million)", "million"), ("Approved Budget $m", "million"),
])
def test_genuine_header_scales_still_read(header, scale):
    assert header_scale(header) == scale


def test_header_scale_and_currency_reading():
    assert header_scale("Expenditure $'000") == "thousand"
    assert header_scale("Cost ($ million)") == "million"
    assert header_scale("Value") is None
    assert header_currency("Value (AUD)", "AUD") == ("AUD", "column_header")
    assert header_currency("Value (USD)", "AUD") == ("USD", "column_header")
    assert header_currency("Value", "AUD") == (None, None)


def test_brackets_in_a_money_column_are_accounting_negatives():
    values = ["1,500", "(300)", "2,700"]
    verdict = classify_column("Expenditure", values)
    extracted = extract_column(values, verdict)
    assert [v for _, v, _ in extracted] == [
        Decimal(1500), Decimal(-300), Decimal(2700)]


def test_cell_amount_parsing():
    assert cell_amount("$1,234.50") == (Decimal("1234.50"), False)
    assert cell_amount("(1,234.50)") == (Decimal("-1234.50"), True)
    assert cell_amount("50000") == (Decimal(50000), False)
    assert cell_amount("15%") is None
    assert cell_amount("Pending") is None
    assert cell_amount("—") is None


def test_header_continuation_row_is_folded_into_the_header():
    """The ANAO Major Projects Report case: a header wrapped across two rows,
    only the first of which the extractor declares. Without folding, the unit
    (`$m`) and the vocabulary (`Budget`) are both invisible and 27 real amounts
    are left un-extracted."""
    body = ["Budget $m", "16,631.3", "9108.9", "6291.8", "5925.8", "78,699.2"]
    header, values = fold_header_continuation("Approved", body)
    assert header == "Approved Budget $m"
    assert values == body[1:]

    verdict = classify_column(header, values)
    assert verdict.is_money
    assert verdict.scale == "million"
    assert next(v for _, v, _ in extract_column(values, verdict)) == Decimal("16631300000.0")


def test_header_continuation_does_not_eat_a_data_row():
    # A column of text values is not a money column with a stray header.
    header, values = fold_header_continuation("Project", ["Wedgetail", "Hornet", "Tiger"])
    assert (header, values) == ("Project", ["Wedgetail", "Hornet", "Tiger"])
    # Nor is a numeric first row a header.
    header, values = fold_header_continuation("Value", ["100", "200", "300"])
    assert (header, values) == ("Value", ["100", "200", "300"])
    # Too few data cells below to tell — leave it alone.
    header, values = fold_header_continuation("Amount", ["$m", "100"])
    assert header == "Amount"


def test_footnote_marker_does_not_fabricate_a_value():
    """`24.2 5` is a value with a footnote marker, not 24.25 — closing the gap
    would invent a number that is in no document."""
    assert cell_amount("24.2 5") is None
    assert cell_amount("1 234 567") == (Decimal(1234567), False)  # spaced thousands
    assert cell_amount("24.25") == (Decimal("24.25"), False)


def test_extra_veto_terms_from_config():
    opts = ColumnOptions(extra_veto_terms=frozenset({"widgets"}))
    verdict = classify_column("Widgets Value", ["1,000", "2,000", "3,000"], options=opts)
    assert verdict.verdict == "vetoed"
