"""Column-evidenced monetary amounts (pure core).

The structural path: a bare number is money because of the *column* it sits
in, not because of anything in the cell. This carries the overwhelming
majority of the corpus's amounts — a 48,997-row grant register recording
$22.7bn contains exactly one ``$`` — so symbol-keyed detection alone reaches
almost none of it (``docs/money-extraction.md``).

A column is classified **once** and every cell beneath inherits the verdict.
Evidence, strongest first:

1. a number format carrying a currency symbol (``$#,##0.00``) — definitive;
2. money-vocabulary header + predominantly numeric cells;
3. predominantly numeric cells — supporting only, never promoting on its own
   (identifiers, counts and postcodes are numerically indistinguishable).

Vetoes suppress a column outright, and matching is whole-word so ``age`` vetoes
an ``Age`` column while ``Average Cost`` survives. Null markers (``—``, ``n/a``,
``nil``) are *absent values*, excluded from the numeric fraction rather than
counted against it. Where no header is recoverable, bare cells are left alone:
under-counting is the correct failure mode here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from womblex.process.money import parse_number
from womblex.process.money_vocab import (
    HEADER_MONEY_TERMS,
    HEADER_VETO_TERMS,
    ISO_4217,
    NULL_MARKERS,
    SCALES,
    SYMBOL_TO_CODE,
    currency_tier,
    header_tokens,
)

# ---------------------------------------------------------------------------
# Options / verdicts
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ColumnOptions:
    """Column-path knobs. Mirrors ``MoneyColumnsConfig``."""

    default_currency: str = "AUD"
    numeric_fraction_min: float = 0.7
    min_cells: int = 3
    extra_header_terms: frozenset[str] = field(default_factory=frozenset)
    extra_veto_terms: frozenset[str] = field(default_factory=frozenset)
    international_numbers: bool = False


@dataclass(slots=True)
class ColumnVerdict:
    """One column's classification, persisted as the money-column audit row."""

    verdict: str            # money | vetoed | insufficient
    evidence: str           # number_format | header+numeric | header_currency | none
    header_text: str
    currency: str | None = None
    scale: str | None = None
    number_format: str | None = None
    numeric_fraction: float = 0.0
    null_fraction: float = 0.0
    veto_term: str | None = None
    confidence: float = 0.0
    cells_total: int = 0

    @property
    def is_money(self) -> bool:
        return self.verdict == "money"


# ---------------------------------------------------------------------------
# Header reading
# ---------------------------------------------------------------------------

_CURRENCY_SYMBOLS = tuple(sorted(SYMBOL_TO_CODE, key=len, reverse=True))

_HEADER_SCALE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # The lookbehind is load-bearing: without it the `000` inside a number in
    # the header matches, so `Grants over $10,000` declares a thousands scale
    # and multiplies every cell beneath it by 1,000.
    (re.compile(r"(?<![\d,.])\$?\s*'?000s?\b"), "thousand"),
    (re.compile(r"\bthousands?\b", re.IGNORECASE), "thousand"),
    (re.compile(r"\bmillions?\b", re.IGNORECASE), "million"),
    (re.compile(r"\bbillions?\b", re.IGNORECASE), "billion"),
    (re.compile(r"\$\s?m\b", re.IGNORECASE), "million"),
    (re.compile(r"\$\s?bn?\b", re.IGNORECASE), "billion"),
    (re.compile(r"\$\s?k\b", re.IGNORECASE), "thousand"),
)

# Excel currency formats: a bare symbol (`$#,##0.00`) or a locale-tagged one
# (`[$$-en-AU]#,##0.00`, `[$£-en-GB]#,##0`).
_FORMAT_LOCALE_RE = re.compile(r"\[\$(?P<sym>[^\-\]]*)")

# `Value (AUD)` — the column naming its own currency.
_PAREN_CODE_RE = re.compile(r"\(\s*\$?\s*([A-Z]{3})\s*\)")


def header_currency(header: str, default: str) -> tuple[str | None, str | None]:
    """Currency named by a header. Returns ``(code, source)``.

    ``Value (AUD)`` states its currency; ``Value`` does not and takes the
    document default at the point the verdict is built, not here.
    """
    if not header:
        return None, None
    # A parenthesised code is the column naming its currency. A bare one is
    # only trusted for tier 1/2, because several ISO codes are ordinary words
    # in caps — `ALL OTHER COMPENSATION ($)` is a dollar column, not Albanian
    # lek, and the header's own `$` should win.
    for token in _PAREN_CODE_RE.findall(header):
        if token in ISO_4217:
            return token, "column_header"
    for token in re.findall(r"\b[A-Z]{3}\b", header):
        if token in ISO_4217 and currency_tier(token) < 3:
            return token, "column_header"
    for sym in _CURRENCY_SYMBOLS:
        if sym in header:
            return SYMBOL_TO_CODE[sym], "column_header"
    return None, None


def header_scale(header: str) -> str | None:
    """Multiplier declared in a header (``$m``, ``$'000``), else ``None``.

    Financial tables put the unit in the header and leave the cells bare, so
    the header supplies the multiplier for every cell beneath it.
    """
    if not header:
        return None
    for pattern, scale in _HEADER_SCALE_PATTERNS:
        if pattern.search(header):
            return scale
    return None


def format_currency(number_format: str | None) -> str | None:
    """Currency implied by a cell number format, else ``None``.

    The strongest available signal for the column-evidenced path: a register's
    money column is frequently identifiable *only* from its format.
    """
    if not number_format:
        return None
    locale = _FORMAT_LOCALE_RE.search(number_format)
    if locale:
        sym = locale.group("sym").strip()
        for candidate in _CURRENCY_SYMBOLS:
            if candidate in sym:
                return SYMBOL_TO_CODE[candidate]
    for sym in _CURRENCY_SYMBOLS:
        if sym in number_format:
            return SYMBOL_TO_CODE[sym]
    return None


def _veto_term(header: str, extra: frozenset[str]) -> str | None:
    tokens = header_tokens(header)
    vetoes = HEADER_VETO_TERMS | extra
    for token in tokens:
        if token in vetoes:
            return token
    return None


def _money_term(header: str, extra: frozenset[str]) -> str | None:
    tokens = header_tokens(header)
    terms = HEADER_MONEY_TERMS | extra
    for token in tokens:
        if token in terms:
            return token
    # Multi-word terms ("contract value", "sum insured").
    lowered = (header or "").lower()
    for term in terms:
        if " " in term and term in lowered:
            return term
    return None


# ---------------------------------------------------------------------------
# Cell parsing
# ---------------------------------------------------------------------------

_TRAILING_JUNK_RE = re.compile(r"[\s ]+")


def is_null_marker(text: str | None) -> bool:
    """True for the absent-value markers financial tables are full of."""
    if text is None:
        return True
    return _TRAILING_JUNK_RE.sub("", text).strip().lower() in NULL_MARKERS


def cell_amount(
    text: str | None, *, international: bool = False,
) -> tuple[Decimal, bool] | None:
    """Parse one cell to ``(value, negative)``; ``None`` when it isn't a number.

    Bracketed values are accounting negatives here without further gating —
    inside a classified money column the column itself satisfies the gate that
    narrative text cannot (``docs/money-extraction.md``).
    """
    if text is None:
        return None
    raw = _TRAILING_JUNK_RE.sub(" ", str(text)).strip()
    if not raw or is_null_marker(raw):
        return None
    if "%" in raw:
        return None

    negative = False
    if raw.startswith("(") and raw.endswith(")"):
        negative = True
        raw = raw[1:-1].strip()
    if raw.startswith("-"):
        negative = True
        raw = raw[1:].strip()
    elif raw.endswith("-"):
        negative = True
        raw = raw[:-1].strip()

    for sym in _CURRENCY_SYMBOLS:
        if raw.startswith(sym):
            raw = raw[len(sym):].strip()
            break
    raw = re.sub(r"\b[A-Z]{3}\b", "", raw).strip()
    if not raw or not re.fullmatch(r"[\d,. ]+", raw):
        return None
    # An internal space only survives as a thousands separator. `24.2 5` is a
    # value with a footnote marker attached, and closing the gap would invent
    # 24.25 — deleting spaces unconditionally fabricates numbers.
    if " " in raw and not re.fullmatch(r"\d{1,3}(?: \d{3})+(?:\.\d+)?", raw):
        return None

    value = parse_number(raw.replace(" ", ""), international=international)
    if value is None:
        return None
    return (-value if negative else value), negative


def scaled(value: Decimal, scale: str | None) -> Decimal:
    """Apply a column scale (``$m`` → ×10⁶) to a cell value."""
    if not scale:
        return value
    factor = SCALES.get(scale.lower())
    return value if factor is None else value * factor


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def fold_header_continuation(
    header: str,
    values: Sequence[str | None],
    *,
    options: ColumnOptions | None = None,
) -> tuple[str, list[str | None]]:
    """Absorb a header row the extractor left sitting in the body.

    PDF financial tables wrap their headers across two lines — ``Approved`` /
    ``Budget $m`` — and the extractor declares only the first row a header. The
    unit and the money vocabulary both live in the second, so without folding
    the column reads as a nameless run of bare numbers and is (correctly, given
    what it can see) left alone. Measured on the ANAO Major Projects Report,
    this is the difference between recovering the 27 approved-budget amounts
    and recovering none of them.

    Only one row is folded, only when it is non-numeric text, and only when the
    rest of the column is numeric enough to be a data column — so a genuine
    text data row is never eaten.
    """
    opts = options or ColumnOptions()
    if not values:
        return header, list(values)
    first, rest = values[0], list(values[1:])
    if is_null_marker(first) or cell_amount(first, international=opts.international_numbers):
        return header, list(values)

    present = [v for v in rest if not is_null_marker(v)]
    if len(present) < opts.min_cells:
        return header, list(values)
    numeric = sum(
        1 for v in present
        if cell_amount(v, international=opts.international_numbers) is not None
    )
    if numeric / len(present) < opts.numeric_fraction_min:
        return header, list(values)
    return f"{header} {first}".strip(), rest


def classify_column(
    header: str,
    values: Sequence[str | None],
    *,
    number_format: str | None = None,
    options: ColumnOptions | None = None,
) -> ColumnVerdict:
    """Classify one column as money / vetoed / insufficient."""
    opts = options or ColumnOptions()
    header = header or ""
    total = len(values)

    present = [v for v in values if not is_null_marker(v)]
    nulls = total - len(present)
    numeric = sum(
        1 for v in present
        if cell_amount(v, international=opts.international_numbers) is not None
    )
    numeric_fraction = (numeric / len(present)) if present else 0.0
    null_fraction = (nulls / total) if total else 0.0

    verdict = ColumnVerdict(
        verdict="insufficient", evidence="none", header_text=header,
        number_format=number_format,
        numeric_fraction=round(numeric_fraction, 4),
        null_fraction=round(null_fraction, 4),
        cells_total=total,
    )

    scale = header_scale(header)
    hdr_currency, _ = header_currency(header, opts.default_currency)
    fmt_currency = format_currency(number_format)

    # A veto term suppresses a column that would otherwise be promoted on
    # vocabulary alone. It does *not* override the column declaring its own
    # currency: `Grant Date Fair Value of Stock and Option Awards ($)` is a
    # money column that happens to contain the word "date", and vetoing it
    # loses every amount beneath it. The `($)` is the header describing itself,
    # in the same string as the incidental term.
    veto = _veto_term(header, opts.extra_veto_terms)
    if veto:
        verdict.veto_term = veto
        if not hdr_currency:
            verdict.verdict, verdict.evidence = "vetoed", "header_veto"
            return verdict
    if number_format and "%" in number_format:
        verdict.verdict, verdict.evidence = "vetoed", "percent_format"
        verdict.veto_term = verdict.veto_term or None
        return verdict

    if fmt_currency and numeric_fraction >= opts.numeric_fraction_min:
        verdict.verdict, verdict.evidence = "money", "number_format"
        verdict.currency = hdr_currency or fmt_currency
        verdict.scale, verdict.confidence = scale, 0.95
        return verdict

    if len(present) < opts.min_cells or numeric_fraction < opts.numeric_fraction_min:
        return verdict

    if _money_term(header, opts.extra_header_terms):
        verdict.verdict, verdict.evidence = "money", "header+numeric"
        verdict.currency = hdr_currency or opts.default_currency
        verdict.scale, verdict.confidence = scale, 0.85
        return verdict

    if hdr_currency:
        verdict.verdict, verdict.evidence = "money", "header_currency"
        verdict.currency, verdict.scale, verdict.confidence = hdr_currency, scale, 0.85
        return verdict

    return verdict


def extract_column(
    values: Sequence[str | None],
    verdict: ColumnVerdict,
    *,
    options: ColumnOptions | None = None,
) -> list[tuple[int, Decimal, bool]]:
    """Per-cell ``(index, value, negative)`` for a classified money column."""
    if not verdict.is_money:
        return []
    opts = options or ColumnOptions()
    out = []
    for i, raw in enumerate(values):
        parsed = cell_amount(raw, international=opts.international_numbers)
        if parsed is None:
            continue
        value, negative = parsed
        try:
            out.append((i, scaled(value, verdict.scale), negative))
        except InvalidOperation:
            continue
    return out


__all__ = [
    "ColumnOptions",
    "ColumnVerdict",
    "cell_amount",
    "classify_column",
    "extract_column",
    "fold_header_continuation",
    "format_currency",
    "header_currency",
    "header_scale",
    "is_null_marker",
    "scaled",
]
