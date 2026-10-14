"""Worded monetary amounts — financial values written out, not in digits.

``two million dollars``, ``five hundred thousand dollars``, ``fifty cents``,
``one and a half million dollars``. Narrative prose and legal drafting spell
amounts out where a table would print digits, and the digit-keyed patterns in
:mod:`womblex.process.money` cannot see them at all: there is no number to
anchor on.

Number words carry no evidence of money on their own — ``more than one million
Australians`` is a headcount, and it is the shape this corpus actually contains
(measured: the only worded-number phrase in the benchmark DOCX). So a **currency
word must sit with the phrase**, exactly as pattern 4 requires for digits. That
one gate is what separates this from counting every spelled-out number in the
document.

Values are exact ``Decimal``s; nothing here reads or writes parquet, and
offsets index the string handed in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from womblex.process.money_vocab import _WORD_ALT as _CURRENCY_WORD_ALT
from womblex.process.money_vocab import SCALE_CANONICAL

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

UNITS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}

TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

# Magnitudes that close a group (`two million`, `five hundred thousand`).
# `hundred` multiplies within a group instead, so it is handled separately.
BIG: dict[str, Decimal] = {
    "thousand": Decimal(10) ** 3,
    "million": Decimal(10) ** 6,
    "billion": Decimal(10) ** 9,
    "trillion": Decimal(10) ** 12,
}

# `half a million dollars` is ordinary Australian reporting prose, and dropping
# the fraction reports the wrong order of magnitude rather than nothing.
FRACTIONS: dict[str, Decimal] = {
    "half": Decimal("0.5"), "quarter": Decimal("0.25"), "quarters": Decimal("0.25"),
}

# `and` joins a hundred or a scale to what follows it (`one hundred and
# fifty`, `two thousand and five`) or introduces a fraction (`one and a half
# million`). It never joins two plain numbers: `between ten and twenty dollars`
# is a range, and adding its endpoints reports thirty dollars.
_AND_ANCHORS: frozenset[str] = frozenset({"hundred"}) | frozenset(BIG)

# Grammatical filler inside an amount phrase. `a` carries no value of its own —
# `a dollar` must not become an amount, since `a dollar figure` and `a dollar
# amount` are the common uses — but it does license a bare magnitude, which is
# what separates the amount `a million dollars` from the unit declaration
# `in million dollars`.
FILLERS: frozenset[str] = frozenset({"and", "a", "an", "of"})

# Articles belong to the amount (`a million dollars` is one million) and stay
# in the span text; a leading `of` or `and` is the sentence's, not the amount's.
_ARTICLES: frozenset[str] = frozenset({"a", "an"})
_LEADING_TRIM: frozenset[str] = FILLERS - _ARTICLES

_PHRASE_WORDS = sorted(
    set(UNITS) | set(TENS) | {"hundred"} | set(BIG) | set(FRACTIONS) | FILLERS,
    key=len, reverse=True,
)
_PHRASE_WORD_ALT = "|".join(_PHRASE_WORDS)

# Words may be separated by a hyphen (`twenty-five`), spaces, or a single line
# break — never two, because the reassembled narrative joins elements with
# `\n\n` and a phrase spanning that join would bind unrelated paragraphs
# (docs/money-extraction.md).
#
# The en dash is a separator here, unlike in the digit patterns where it marks
# a range. That is deliberate: `ten–twenty dollars` is then consumed whole and
# declined by the grammar below, rather than leaving `twenty dollars` behind as
# if the range's upper endpoint were the amount.
_SEP = r"(?:[^\S\n]*[-‐‑–][^\S\n]*|[^\S\n]*\n[^\S\n]*|[^\S\n]+)"

# Repetition is bounded: the longest real phrase (`one million two hundred and
# fifty thousand`) is well inside ten tokens, and an unbounded run over a page
# of prose is where a word-alternation regex gets expensive.
_PHRASE = rf"(?:{_PHRASE_WORD_ALT})(?:{_SEP}(?:{_PHRASE_WORD_ALT})){{0,9}}"

WORDED_AMOUNT_RE = re.compile(
    rf"(?<![A-Za-z])(?P<words>{_PHRASE}){_SEP}(?P<unit>{_CURRENCY_WORD_ALT})\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class WordedAmount:
    """One worded amount: offsets into the text handed in, plus its value."""

    start: int
    end: int
    value: Decimal
    unit: str
    scale: str | None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _tokens(phrase: str) -> list[str]:
    return [t for t in re.split(r"[\s‐‑–-]+", phrase.lower()) if t]


def parse_number_words(phrase: str) -> tuple[Decimal, str | None] | None:
    """Parse a spelled-out number to ``(value, canonical_scale)``.

    Returns ``None`` for anything English does not write as a single number:
    a phrase with no number in it (``a``, ``and``), a magnitude sequence no
    amount uses (``million thousand million``), a bare scale word with nothing
    to scale (``million dollars`` is a unit declaration, ``a million dollars``
    is an amount), and two plain numbers run together (``ten and twenty``,
    ``ten–twenty``) — which is a *range*, and adding its endpoints reports
    thirty dollars.

    Declining is the right failure throughout: a partial or recombined parse of
    a worded amount is wrong by a power of ten, and a wrong value is worse than
    a missing one.
    """
    tokens = _tokens(phrase)
    total = Decimal(0)
    current = Decimal(0)
    seen_number = False
    pending_and = False
    article = False
    last_was_unit = False             # a unit word closes its group
    last_big: Decimal | None = None
    largest: str | None = None

    for i, token in enumerate(tokens):
        if token in UNITS:
            # A unit opens a group or follows a tens word (`twenty-five`).
            # Two units running (`five six`) is not one number.
            if last_was_unit:
                return None
            current += UNITS[token]
            last_was_unit = seen_number = True
        elif token in TENS:
            # `ten twenty` / `nineteen fifty` are a range and a year, not
            # numbers: a tens word opens its group or follows a hundred.
            if current % 100:
                return None
            current += TENS[token]
            last_was_unit, seen_number = False, True
        elif token == "hundred":
            if not current and not article:
                return None       # bare `hundred dollars` declares a unit
            current = (current or Decimal(1)) * 100
            last_was_unit, seen_number = False, True
        elif token in FRACTIONS:
            # `one and a half million` adds; `three quarters of a million`
            # multiplies. The conjunction is what distinguishes them.
            frac = FRACTIONS[token]
            current = current + frac if (pending_and or not current) else current * frac
            last_was_unit, seen_number = False, True
        elif token in BIG:
            factor = BIG[token]
            if last_big is not None and factor >= last_big:
                return None       # `thousand million`-style sequence
            if not current and not article:
                return None       # `in million dollars` is a unit declaration
            total += (current or Decimal(1)) * factor
            current = Decimal(0)
            last_was_unit, last_big = False, factor
            largest = largest or SCALE_CANONICAL[token]
            seen_number = True
        elif token == "and":
            if not _and_licensed(tokens, i):
                return None
        elif token not in FILLERS:
            return None

        article = token in _ARTICLES
        if token == "and":
            pending_and = True
        elif token not in _ARTICLES:
            # `one and a half` must keep the conjunction in view across the
            # article, or the fraction multiplies instead of adding.
            pending_and = False

    if not seen_number:
        return None
    return total + current, largest


def _and_licensed(tokens: list[str], i: int) -> bool:
    """Is the ``and`` at *i* joining a number, or two of them?"""
    previous = tokens[i - 1] if i else ""
    following = next((t for t in tokens[i + 1:] if t not in _ARTICLES), "")
    return previous in _AND_ANCHORS or following in FRACTIONS


_LEADING_TRIM_RE = re.compile(
    rf"(?:(?:{'|'.join(sorted(_LEADING_TRIM))})(?:{_SEP}|$))+", re.IGNORECASE)


def _trim_fillers(phrase: str) -> int:
    """Leading characters to drop from a match — ``of`` in ``of ten dollars``.

    The span is what the document wrote for the amount; the preposition that
    attached it to the sentence is not part of it.
    """
    m = _LEADING_TRIM_RE.match(phrase)
    return m.end() if m else 0


def find_worded_amounts(text: str) -> list[WordedAmount]:
    """Every worded amount in *text*, in document order.

    A currency word is required: the phrase alone is a number, not money.
    """
    out: list[WordedAmount] = []
    for m in WORDED_AMOUNT_RE.finditer(text):
        words = m.group("words")
        offset = _trim_fillers(words)
        parsed = parse_number_words(words[offset:])
        if parsed is None:
            continue
        value, scale = parsed
        if value <= 0:
            continue  # `zero dollars` states an absence, not an amount
        out.append(WordedAmount(
            start=m.start("words") + offset,
            end=m.end("unit"),
            value=value,
            unit=m.group("unit").lower(),
            scale=scale,
        ))
    return out


__all__ = [
    "BIG",
    "FRACTIONS",
    "TENS",
    "UNITS",
    "WORDED_AMOUNT_RE",
    "WordedAmount",
    "find_worded_amounts",
    "parse_number_words",
]
