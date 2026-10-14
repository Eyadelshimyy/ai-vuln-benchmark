"""Self-evidencing monetary amount recognition (pure core).

Recovers amounts that carry their own evidence — a currency symbol, an ISO
4217 code or a currency word sitting with the number — from a text string, and
normalises them to exact ``Decimal`` values. The column-evidenced path (a bare
number whose money-ness comes from its column) lives in
:mod:`womblex.process.money_columns`; the per-stage driver that applies both
over a shard directory is :mod:`womblex.process.money_stage`.

The number itself may be written in words (``two million dollars``); that
parser is :mod:`womblex.process.money_words`, and the layer beneath the
patterns — number reading, currency resolution, false-positive blocking — is
:mod:`womblex.process.money_numbers`, re-exported here.

Nothing here reads or writes parquet, and nothing rewrites text: offsets index
the string handed in, so callers own the coordinate space.

**Every extraction must have positive evidence that the number represents
money** (``docs/money-extraction.md``). A bare number with no marker is not an extraction
— except under ``implicit_context``, which is off by default because it is
measurably low precision on this corpus.

Patterns are applied in the priority order of ``docs/money-extraction.md``; overlap is
resolved by priority, then span length, then confidence.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal

# Number reading and false-positive blocking live in `money_numbers`; they are
# re-exported here so `from womblex.process.money import parse_number` — the
# established import site — keeps working.
from womblex.process.money_numbers import (
    IntervalIndex,
    ambiguous_number_spans,
    apply_scale,
    blocked_spans,
    parse_number,
    resolve_iso,
    resolve_symbol,
)
from womblex.process.money_vocab import (
    _SCALE_ALT,
    _SUFFIX_SYMBOL_ALT,
    _SYMBOL_ALT,
    _WORD_ALT,
    ACCOUNTING_RE,
    AMBIGUOUS_SCALES,
    CONTEXT_RE,
    CURRENCY_WORDS,
    MODIFIER_RE,
    NUM_AU,
    NUM_INTL,
    SUBUNIT_SYMBOLS,
    SUBUNIT_WORDS,
    SYMBOL_TO_CODE,
    currency_tier,
)
from womblex.process.money_words import find_worded_amounts

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MoneySpan:
    """One extracted amount. ``text`` is the original — never lost."""

    text: str
    start: int
    end: int
    value: Decimal
    currency: str | None
    currency_source: str
    evidence: str
    confidence: float
    negative: bool = False
    multiplier: str | None = None
    modifier: str | None = None
    range_group: int | None = None
    range_role: str | None = None


@dataclass(slots=True)
class MoneyOptions:
    """Detector knobs. Mirrors the narrative half of ``MoneyConfig``."""

    default_currency: str = "AUD"
    international_numbers: bool = False
    implicit_context: bool = False
    min_confidence: float = 0.5
    context_chars: int = 160


# Priority drives overlap resolution (lower wins). Ranges are resolved first
# and claim their whole span, so a range endpoint is never re-claimed.
# Accounting negatives outrank the symbol patterns because they *enclose* one:
# `($100)` must resolve to -100, not to the `$100` sitting inside it.
_PRIORITY = {"p7": 0, "p9": 0, "p1": 1, "p6": 1, "p2": 2, "p4": 4, "p11": 4,
             "p3": 5, "p5": 6, "p10": 9}

_CONFIDENCE = {"p1": 0.99, "p6": 0.99, "p2": 0.99, "p3": 0.90, "p4": 0.90,
               "p5": 0.75, "p9": 0.90, "p10": 0.35, "p11": 0.90}

# Whitespace inside a pattern may span at most one line break, so no match can
# cross the ``\n\n`` element join `reassemble_narrative` writes — a pattern that
# did would bind two unrelated paragraphs together. Same discipline as the PII
# regexes (CLAUDE.md). Range separators are stricter still (`_HGAP`): a range
# fabricates a *relationship* between two numbers, so it stays on one line.
_GAP = r"[^\S\n]*\n?[^\S\n]*"
_HGAP = r"[^\S\n]*"

# A PDF text layer renders a leading minus as the true minus sign as often as
# the ASCII hyphen, and reading `−$5.2 million` as positive inverts the sign
# rather than missing the amount. The en dash is deliberately *not* here: it is
# the range separator, and admitting it would turn `$10–20m` into a negative.
_MINUS = r"[-−]"

# The scale token is matched case-insensitively *locally*. The ISO patterns
# (p2 / p3) are compiled case-sensitively so `[A-Z]{3}` cannot match a
# lowercase word, and that sensitivity used to reach the scale tail too:
# `USD 6.6Mn` matched only `USD 6.6` and stored 6.6 at 0.99 confidence —
# silently wrong by 10**6 — while `6.6Mn USD` missed entirely. The trailing
# `(?![A-Za-z])` still keeps a scale letter out of an ISO code's first
# character (`100 MUR` is Mauritian rupees, not 100 million UR).
_SCALE_TAIL = rf"(?:{_GAP}(?i:(?P<scale>{_SCALE_ALT}))(?![A-Za-z]))?"
_NEG = rf"(?P<neg>{_MINUS}{_HGAP})?"


def _num_alt(international: bool) -> str:
    return f"(?:{NUM_INTL}|{NUM_AU})" if international else f"(?:{NUM_AU})"


def _compile(international: bool) -> dict[str, re.Pattern[str]]:
    """Build the pattern set for a locale mode (cached by :func:`_patterns`)."""
    num = _num_alt(international)
    scale_tail = rf"(?:{_GAP}(?P<scale_a>{_SCALE_ALT})(?![A-Za-z]))?"
    left = rf"(?:(?P<sym_a>{_SYMBOL_ALT}){_HGAP})?(?P<neg_a>{_MINUS}{_HGAP})?(?P<num_a>{num})" \
           rf"{scale_tail}"
    right = rf"(?:(?P<sym_b>{_SYMBOL_ALT}){_HGAP})?(?P<num_b>{num})" \
            rf"(?:{_HGAP}(?P<scale_b>{_SCALE_ALT})(?![A-Za-z]))?"
    return {
        # 7 — ranges. Scanned first so an endpoint is never claimed alone.
        "p7": re.compile(
            rf"(?<![A-Za-z0-9]){left}{_HGAP}(?:[\u2013\u2014\u2012-]|to){_HGAP}{right}",
            re.IGNORECASE),
        "p7_between": re.compile(
            rf"\bbetween{_HGAP}\s?{left}{_HGAP}\s?and{_HGAP}\s?{right}", re.IGNORECASE),
        # 1 / 6 — symbol prefix, optional magnitude.
        "p1": re.compile(
            rf"(?<![A-Za-z0-9]){_NEG}(?P<sym>{_SYMBOL_ALT}){_HGAP}(?P<num>{num})"
            rf"{_SCALE_TAIL}", re.IGNORECASE),
        # 2 — ISO prefix (`AUD 100`, `AUD$21.9 million`). Case-sensitive.
        "p2": re.compile(
            rf"(?<![A-Za-z])(?P<iso>[A-Z]{{3}}){_HGAP}(?P<sym>\$)?{_HGAP}{_NEG}"
            rf"(?P<num>{num}){_SCALE_TAIL}"),
        # 3 — ISO suffix (`100 AUD`).
        "p3": re.compile(
            rf"(?<![A-Za-z0-9]){_NEG}(?P<num>{num})"
            rf"{_SCALE_TAIL}{_HGAP}"
            rf"(?P<iso>[A-Z]{{3}})(?![A-Za-z])"),
        # 4 — currency word (`100 dollars`, `250 Australian dollars`).
        "p4": re.compile(
            rf"(?<![A-Za-z0-9]){_NEG}(?P<num>{num}){_SCALE_TAIL}{_GAP}"
            rf"(?P<word>{_WORD_ALT})\b", re.IGNORECASE),
        # 5 — symbol suffix (`100$`, `50€`).
        "p5": re.compile(
            rf"(?<![A-Za-z0-9]){_NEG}(?P<num>{num}){_HGAP}"
            rf"(?P<sym>{_SUFFIX_SYMBOL_ALT})(?![0-9])"),
        # 9 — accounting negative (`($100)`, `$(100)`, `(6,550.1)` under
        # context). The symbol sits inside or outside the bracket depending on
        # the statement's house style; both mark the same thing.
        "p9": re.compile(
            rf"(?:(?P<sym_out>{_SYMBOL_ALT}){_HGAP})?"
            rf"\({_HGAP}(?P<sym>{_SYMBOL_ALT})?{_HGAP}(?P<num>{num})"
            rf"{_SCALE_TAIL}{_HGAP}\)", re.IGNORECASE),
        # 10 — bare number, implicit financial context. Off by default.
        "p10": re.compile(rf"(?<![A-Za-z0-9.,]){_NEG}(?P<num>{num})"
                          rf"{_SCALE_TAIL}(?![A-Za-z0-9])", re.IGNORECASE),
    }


_PATTERN_CACHE: dict[bool, dict[str, re.Pattern[str]]] = {}


def _patterns(international: bool) -> dict[str, re.Pattern[str]]:
    if international not in _PATTERN_CACHE:
        _PATTERN_CACHE[international] = _compile(international)
    return _PATTERN_CACHE[international]


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


_MALFORMED_GROUP_RE = re.compile(r",\d{1,2}(?!\d)")
_DOTTED_CONTINUATION_RE = re.compile(r"\.\d")


def _ambiguous_continuation(m: re.Match[str], num_group: str, international: bool) -> bool:
    """True when the number runs on in a way this locale can't account for.

    Two cases, both silently wrong if extracted rather than declined:

    - ``€1.000,50`` in Australian mode reads as ``€1.000`` — one euro, wrong by
      10³. ``international_numbers`` is the deliberate opt-in for that format.
    - ``$1,23`` has a malformed thousands group; the pattern consumes ``$1`` and
      drops the rest, reporting one dollar instead of a hundred-odd.
    - ``$3.219.3m`` (a real ANAO typo for ``$3,219.3m``) has a second dotted
      group; the pattern consumes ``$3.219`` and reports three dollars for a
      $3.2 billion project budget. Repairing the typo would be a guess, so the
      amount is declined.
    """
    tail = m.string[m.end(num_group):m.end(num_group) + 4]
    if _MALFORMED_GROUP_RE.match(tail) or _DOTTED_CONTINUATION_RE.match(tail):
        return True
    if international:
        return False
    raw = m.group(num_group)
    return bool(re.fullmatch(r"\d{1,3}\.\d{3}", raw)) and bool(re.match(r",\d", tail))


def _candidate(
    m: re.Match[str], evidence: str, *,
    currency: str | None, currency_source: str,
    num_group: str = "num", scale_group: str = "scale",
    confidence: float | None = None, negative: bool = False,
    international: bool = False, subunit: bool = False,
) -> MoneySpan | None:
    if _ambiguous_continuation(m, num_group, international):
        return None
    value = parse_number(m.group(num_group), international=international)
    if value is None:
        return None
    scale_raw = m.groupdict().get(scale_group)
    value, multiplier = apply_scale(value, scale_raw)
    if subunit:
        value = value / 100
        multiplier = multiplier or "cents"
    neg = negative or bool(m.groupdict().get("neg"))
    if neg:
        value = -value
    conf = _CONFIDENCE[evidence] if confidence is None else confidence
    if currency is not None and currency_tier(currency) == 3:
        conf = max(0.1, conf - 0.10)
    return MoneySpan(
        text=m.group(0).strip(),
        start=m.start(),
        end=m.end(),
        value=value,
        currency=currency,
        currency_source=currency_source,
        evidence=evidence,
        confidence=round(conf, 4),
        negative=neg,
        multiplier=multiplier,
    )


def _scan_symbol_prefix(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    out = []
    for m in pats["p1"].finditer(text):
        evidence = "p6" if m.group("scale") else "p1"
        span = _candidate(
            m, evidence, currency=resolve_symbol(m.group("sym")),
            currency_source="symbol", international=opts.international_numbers,
        )
        if span:
            out.append(span)
    return out


def _tier3_reinforced(text: str, start: int, end: int) -> bool:
    """Is a tier-3 ISO code backed by surrounding financial context?

    Tier 3 is "supported, but lower confidence unless reinforced by surrounding
    context" (docs/money-extraction.md). Enforced as a gate, not just a penalty, because
    several ISO codes are ordinary English words in caps: without it `TOP 10
    projects` is ten Tongan paʻanga and `ALL 25 recipients` is Albanian lek.
    Tier 1 and 2 codes stand on their own.
    """
    window = text[max(0, start - 48):end + 48]
    if any(sym in window for sym in SYMBOL_TO_CODE):
        return True
    return CONTEXT_RE.search(window) is not None


def _scan_iso_prefix(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    out = []
    for m in pats["p2"].finditer(text):
        code = resolve_iso(m.group("iso"))
        if code is None:
            continue
        if currency_tier(code) == 3 and not _tier3_reinforced(text, m.start(), m.end()):
            continue
        evidence = "p6" if m.group("scale") else "p2"
        span = _candidate(
            m, evidence, currency=code, currency_source="iso",
            international=opts.international_numbers,
        )
        if span:
            out.append(span)
    return out


def _scan_iso_suffix(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    out = []
    for m in pats["p3"].finditer(text):
        code = resolve_iso(m.group("iso"))
        if code is None:
            continue
        if currency_tier(code) == 3 and not _tier3_reinforced(text, m.start(), m.end()):
            continue
        span = _candidate(m, "p3", currency=code, currency_source="iso",
                          international=opts.international_numbers)
        if span:
            out.append(span)
    return out


def _scan_word(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    out = []
    for m in pats["p4"].finditer(text):
        word = m.group("word").lower()
        code = CURRENCY_WORDS.get(word, opts.default_currency)
        span = _candidate(
            m, "p4", currency=code, currency_source="word",
            international=opts.international_numbers,
            subunit=word in SUBUNIT_WORDS,
        )
        if span:
            out.append(span)
    return out


def _scan_symbol_suffix(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    out = []
    for m in pats["p5"].finditer(text):
        sym = m.group("sym")
        span = _candidate(m, "p5", currency=resolve_symbol(sym),
                          currency_source="symbol",
                          international=opts.international_numbers,
                          subunit=sym in SUBUNIT_SYMBOLS)
        if span:
            out.append(span)
    return out


def _scan_worded(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    """Pattern 11 — amounts written in words (`two million dollars`).

    The phrase is parsed by :mod:`womblex.process.money_words`; the currency
    word it required is resolved here, so worded and digit amounts share one
    currency model and one sub-unit rule (``fifty cents`` is 0.50).
    """
    out = []
    for amount in find_worded_amounts(text):
        code = CURRENCY_WORDS.get(amount.unit, opts.default_currency)
        value = amount.value / 100 if amount.unit in SUBUNIT_WORDS else amount.value
        conf = _CONFIDENCE["p11"]
        if code is not None and currency_tier(code) == 3:
            conf = max(0.1, conf - 0.10)
        out.append(MoneySpan(
            text=text[amount.start:amount.end],
            start=amount.start, end=amount.end, value=value, currency=code,
            currency_source="word", evidence="p11", confidence=round(conf, 4),
            multiplier=amount.scale or (
                "cents" if amount.unit in SUBUNIT_WORDS else None),
        ))
    return out


def _scan_accounting(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    """Bracketed negatives — gated, never scanned bare.

    Ungated this is the corpus's worst false-positive source (`s167(1)`,
    `(02) 6203 7300`, `(2018)`), so a bracketed number is an amount only when
    a currency marker sits inside or immediately before the brackets, or
    accounting context surrounds it *and* the number is formatted like a
    financial value (a decimal or a thousands group).
    """
    out = []
    for m in pats["p9"].finditer(text):
        sym = m.group("sym") or m.group("sym_out")
        currency = resolve_symbol(sym) if sym else None
        source = "symbol"
        confidence = _CONFIDENCE["p9"]
        if currency is None:
            before = text[max(0, m.start() - 6):m.start()].strip()
            code = resolve_iso(before[-3:]) if len(before) >= 3 else None
            if code:
                currency, source = code, "iso"
            else:
                raw = m.group("num")
                if "," not in raw and "." not in raw:
                    continue
                window = text[max(0, m.start() - 120):m.end() + 120]
                if not ACCOUNTING_RE.search(window):
                    continue
                currency, source, confidence = opts.default_currency, "document_default", 0.60
        span = _candidate(m, "p9", currency=currency, currency_source=source,
                          confidence=confidence, negative=True,
                          international=opts.international_numbers)
        if span:
            out.append(span)
    return out


def _scan_implicit(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    """Pattern 10 — bare numbers near financial trigger vocabulary."""
    windows = [(m.start(), m.end() + 60) for m in CONTEXT_RE.finditer(text)]
    if not windows:
        return []
    out = []
    for m in pats["p10"].finditer(text):
        if not any(s <= m.start() < e for s, e in windows):
            continue
        scale = m.group("scale")
        if scale and scale.lower() in AMBIGUOUS_SCALES:
            continue  # no currency marker to license a bare `m` / `k` / `b`
        span = _candidate(
            m, "p10", currency=opts.default_currency,
            currency_source="document_default",
            international=opts.international_numbers,
        )
        if span:
            out.append(span)
    return out


# ---------------------------------------------------------------------------
# Ranges (pattern 7)
# ---------------------------------------------------------------------------


def _scan_ranges(
    text: str, pats, opts: MoneyOptions, blocked: IntervalIndex,
) -> tuple[list[MoneySpan], list[tuple[int, int]]]:
    """Both endpoints, linked by ``range_group``; the whole span is claimed.

    Australian documents use ranges frequently and one endpoint often carries
    the evidence for both (``$10–20 million``): a missing symbol or magnitude
    on either side is inherited from the other rather than collapsing the
    range to a single value.
    """
    spans: list[MoneySpan] = []
    claimed: list[tuple[int, int]] = []
    group = 0
    for key in ("p7_between", "p7"):
        for m in pats[key].finditer(text):
            if blocked.overlaps(m.start(), m.end()):
                continue
            if any(m.start() < e and s < m.end() for s, e in claimed):
                continue  # claimed ranges are few; a linear check is cheap here
            sym_a, sym_b = m.group("sym_a"), m.group("sym_b")
            if not sym_a and not sym_b:
                continue  # no evidence on either endpoint
            if any(_ambiguous_continuation(m, g, opts.international_numbers)
                   for g in ("num_a", "num_b")):
                continue  # malformed / continental endpoint — decline the pair
            lo = parse_number(m.group("num_a"), international=opts.international_numbers)
            hi = parse_number(m.group("num_b"), international=opts.international_numbers)
            if lo is None or hi is None:
                continue
            scale = m.group("scale_a") or m.group("scale_b")
            lo, mult = apply_scale(lo, scale)
            hi, _ = apply_scale(hi, scale)
            currency = resolve_symbol(sym_a or sym_b)
            neg = bool(m.group("neg_a"))
            if neg:
                lo = -lo
            conf = 0.95 if (sym_a and sym_b) else 0.90
            group += 1
            for value, role, (s, e) in (
                (lo, "lower", (m.start("num_a"), m.end("num_a"))),
                (hi, "upper", (m.start("num_b"), m.end("num_b"))),
            ):
                spans.append(MoneySpan(
                    text=text[s:e], start=s, end=e, value=value,
                    currency=currency, currency_source="symbol", evidence="p7",
                    confidence=conf, negative=neg and role == "lower",
                    multiplier=mult, range_group=group, range_role=role,
                ))
            claimed.append((m.start(), m.end()))
    return spans, claimed


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _resolve_overlaps(candidates: list[MoneySpan]) -> list[MoneySpan]:
    """Keep the highest-quality non-overlapping matches.

    Ranked by pattern priority, then span length, then confidence — a longer
    match at equal priority carries more evidence (`AUD$21.9 million` over
    `$21.9`), so length breaks priority ties rather than the reverse.
    """
    ordered = sorted(
        candidates,
        key=lambda c: (_PRIORITY[c.evidence], -(c.end - c.start), -c.confidence, c.start),
    )
    # `kept` is maintained sorted and disjoint, so only the neighbours either
    # side of the insertion point can overlap — no full rescan per candidate.
    kept: list[MoneySpan] = []
    starts: list[int] = []
    for cand in ordered:
        i = bisect_right(starts, cand.start)
        if i > 0 and kept[i - 1].end > cand.start:
            continue
        if i < len(kept) and kept[i].start < cand.end:
            continue
        kept.insert(i, cand)
        starts.insert(i, cand.start)
    return kept


def _attach_modifier(text: str, span: MoneySpan) -> None:
    """Record `approximately` / `up to` / `~` separately — never in the value."""
    prefix = text[max(0, span.start - 32):span.start]
    m = MODIFIER_RE.search(prefix)
    if m:
        span.modifier = m.group(0).strip().lower()


def _is_restatement(text: str, prev: MoneySpan, cur: MoneySpan) -> bool:
    """Is *cur* a parenthesised restatement of *prev* rather than a new amount?

    Drafting writes one amount twice — ``one million dollars ($1,000,000)`` —
    and both readings of that bracket were wrong: the bracket is an accounting
    negative, so the sentence yielded −1,000,000, and once worded amounts are
    recognised it also yielded the same money twice.

    The two forms must differ — one worded, one in digits. Two bracketed
    *digit* amounts of equal value are the ordinary financial-table shape
    (``5,000 (5,000)`` is this year and last), and collapsing those would
    discard a real negative.
    """
    if abs(cur.value) != abs(prev.value):
        return False
    if (cur.evidence == "p11") == (prev.evidence == "p11"):
        return False
    gap = text[prev.end:cur.start]
    if "\n" in gap:
        return False
    if cur.text.startswith("(") and cur.text.endswith(")"):
        return not gap.strip()                      # `… dollars ($1,000,000)`
    if gap.strip() != "(":
        return False
    return text[cur.end:cur.end + 2].lstrip().startswith(")")  # `$1m (one million dollars)`


def _collapse_restatements(text: str, spans: list[MoneySpan]) -> list[MoneySpan]:
    """Drop the restating half of a `worded (digits)` pair. *spans* is sorted."""
    kept: list[MoneySpan] = []
    for span in spans:
        if kept and _is_restatement(text, kept[-1], span):
            continue
        kept.append(span)
    return kept


def context_for(text: str, span: MoneySpan, width: int) -> str:
    """Surrounding sentence-ish window, capped at *width* characters."""
    if width <= 0:
        return ""
    half = width // 2
    start = max(0, span.start - half)
    end = min(len(text), span.end + half)
    return " ".join(text[start:end].split())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_AMOUNT_SIGNAL_RE = re.compile(r"\d|\b(?:dollars?|cents?)\b", re.IGNORECASE)


def has_amount_signal(text: str) -> bool:
    """Could *text* hold an amount at all? A pre-filter for per-cell scanning.

    Every digit pattern needs a digit, and the worded pattern needs a currency
    word — so a prose cell with neither can be skipped without a scan.
    """
    return _AMOUNT_SIGNAL_RE.search(text) is not None


def find_money(text: str, options: MoneyOptions | None = None) -> list[MoneySpan]:
    """Extract self-evidencing monetary amounts from *text*.

    Offsets index *text* exactly as handed in — no normalisation, no rewriting.
    Returns spans sorted by position, filtered to ``min_confidence``.
    """
    if not text:
        return []
    opts = options or MoneyOptions()
    pats = _patterns(opts.international_numbers)
    blocked = IntervalIndex([
        (s, e) for s, e, _ in
        blocked_spans(text)
        + ambiguous_number_spans(text, international=opts.international_numbers)
    ])

    ranges, claimed = _scan_ranges(text, pats, opts, blocked)
    claimed_index = IntervalIndex(claimed)

    candidates: list[MoneySpan] = []
    for scan in (_scan_symbol_prefix, _scan_iso_prefix, _scan_iso_suffix,
                 _scan_word, _scan_worded, _scan_symbol_suffix, _scan_accounting):
        candidates.extend(scan(text, pats, opts))
    if opts.implicit_context:
        candidates.extend(_scan_implicit(text, pats, opts))

    candidates = [
        c for c in candidates
        if not blocked.overlaps(c.start, c.end)
        and not claimed_index.overlaps(c.start, c.end)
    ]

    resolved = _resolve_overlaps(candidates) + ranges
    resolved.sort(key=lambda c: (c.start, c.end))
    resolved = _collapse_restatements(text, resolved)

    out = []
    for span in resolved:
        if span.confidence < opts.min_confidence:
            continue
        _attach_modifier(text, span)
        out.append(span)
    return out


__all__ = [
    "MoneyOptions",
    "MoneySpan",
    "ambiguous_number_spans",
    "apply_scale",
    "blocked_spans",
    "context_for",
    "find_money",
    "has_amount_signal",
    "parse_number",
    "resolve_iso",
    "resolve_symbol",
]
