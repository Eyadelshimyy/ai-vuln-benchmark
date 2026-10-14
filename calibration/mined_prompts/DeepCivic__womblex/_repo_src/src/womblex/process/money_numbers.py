"""Number reading and false-positive blocking for the money op (pure core).

The layer beneath the patterns: *is this run of characters a number this
locale can read*, *what currency do these characters name*, and *is this
number allowed to be money at all*. :mod:`womblex.process.money` composes
these into candidate spans; :mod:`womblex.process.money_columns` reuses the
number reading for cells.

Split out when the pattern set grew to cover worded and space-grouped amounts
and ``money.py`` passed the 750-line cap. Nothing here knows about
``MoneySpan`` — which is what makes the split clean rather than arbitrary.
The names stay importable from :mod:`womblex.process.money`.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from decimal import Decimal

from womblex.process.money_vocab import (
    CODE_ALIASES,
    FALSE_POSITIVE_PATTERNS,
    GROUP_SPACES,
    ISO_4217,
    POSTCODE_RE,
    SCALE_CANONICAL,
    SCALES,
    STATE_RE,
    SYMBOL_TO_CODE,
)

# ---------------------------------------------------------------------------
# Number / currency normalisation
# ---------------------------------------------------------------------------


_GROUP_SPACE_RE = re.compile(f"[{GROUP_SPACES}]")


def parse_number(raw: str, *, international: bool = False) -> Decimal | None:
    """Parse an Australian (or, opt-in, international) formatted number."""
    s = _GROUP_SPACE_RE.sub("", raw.strip())  # `10 000` groups its thousands
    if not s:
        return None
    if international and ("." in s and "," in s and s.rindex(",") > s.rindex(".")):
        s = s.replace(".", "").replace(",", ".")
    elif international and "," in s and "." not in s and re.fullmatch(r"\d+,\d+", s):
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return Decimal(s)
    except Exception:
        return None


def apply_scale(value: Decimal, scale: str | None) -> tuple[Decimal, str | None]:
    """Multiply by a magnitude suffix. Returns ``(value, canonical_suffix)``.

    The suffix returned is the canonical name for the magnitude, not the token
    the document happened to use: ``$1.2m`` and ``$1.2 million`` both report
    ``million``, so the persisted ``multiplier`` is one value per magnitude.
    """
    if not scale:
        return value, None
    key = scale.lower()
    factor = SCALES.get(key)
    if factor is None:
        return value, None
    return value * factor, SCALE_CANONICAL[key]


def resolve_symbol(sym: str) -> str:
    return SYMBOL_TO_CODE.get(sym, SYMBOL_TO_CODE.get(sym.upper(), "AUD"))


def resolve_iso(code: str) -> str | None:
    """Return the ISO code if *code* is a real currency code, else ``None``."""
    upper = code.upper()
    if upper in CODE_ALIASES:
        return CODE_ALIASES[upper]
    return upper if upper in ISO_4217 else None


# ---------------------------------------------------------------------------
# False-positive blocking
# ---------------------------------------------------------------------------


def blocked_spans(text: str) -> list[tuple[int, int, str]]:
    """Spans covering Australian false-positive classes (dates, ABNs, …).

    A candidate overlapping one of these is discarded. Measurement and
    percentage matches are skipped where a currency marker immediately
    precedes the number, so ``$100 m`` stays a magnitude expression while
    ``100m road`` does not.
    """
    out: list[tuple[int, int, str]] = []
    for name, pattern in FALSE_POSITIVE_PATTERNS.items():
        for m in pattern.finditer(text):
            if name in ("measurement", "temperature") and _preceded_by_currency(text, m.start()):
                continue
            if name == "incident_ref" and (
                _iso_prefixed(m.group(0)) or _preceded_by_currency(text, m.start())
            ):
                continue  # `USD100` / `$US655.5m` are amounts, not reference numbers
            out.append((m.start(), m.end(), name))
    for m in POSTCODE_RE.finditer(text):
        window = text[max(0, m.start() - 40):m.end() + 40]
        if STATE_RE.search(window):
            out.append((m.start(), m.end(), "postcode"))
    return out


_NUMBER_RUN_RE = re.compile(r"\d[\d,.]*")
_CONTINENTAL_RUN_RE = re.compile(r"\d{1,3}(?:\.\d{3})+,\d+")
_MALFORMED_RUN_RE = re.compile(r"\d+(?:,\d{3})*,\d{1,2}")


def ambiguous_number_spans(
    text: str, *, international: bool = False,
) -> list[tuple[int, int, str]]:
    """Numeric runs this locale cannot read, blocked whole.

    :func:`_ambiguous_continuation` declines the candidate that *starts* at
    such a run, but declining is not enough on its own: the run's decimal tail
    is itself a complete match for a suffix pattern, so in Australian mode
    ``1.234,56 EUR`` came back as ``56 EUR`` — the value wrong by 10³, which is
    the failure this guard exists to prevent. Blocking the whole run keeps
    every pattern off it, so the amount is missed rather than misread.

    Prefix-marker forms were already safe (``€1.000,50`` yields nothing,
    because the tail has no leading marker to match), which is why only the
    ISO-suffix, currency-word and symbol-suffix patterns leaked.
    """
    if international:
        return []  # the continental reading is the correct one in this mode
    out: list[tuple[int, int, str]] = []
    for m in _NUMBER_RUN_RE.finditer(text):
        raw = m.group(0).rstrip(".,")
        if _CONTINENTAL_RUN_RE.fullmatch(raw) or _MALFORMED_RUN_RE.fullmatch(raw):
            out.append((m.start(), m.start() + len(raw), "ambiguous_number"))
    return out


def _iso_prefixed(token: str) -> bool:
    """True for `USD100`-shaped tokens whose letters are a real currency code."""
    m = re.match(r"([A-Z]{2,4})", token)
    return m is not None and resolve_iso(m.group(1)) is not None


_TRAILING_NUMBER_RE = re.compile(r"[\d,.]+$")


def _preceded_by_currency(text: str, pos: int) -> bool:
    """True when a currency symbol or ISO code sits just before *pos*.

    The digits of the number itself are stepped over first. A measurement match
    can begin *inside* an amount — ``$US655.5m`` has no word boundary before
    ``655``, so the metre pattern matches at ``5m`` and the marker to find is
    behind ``655.``. Without that step-back the whole amount is blocked as a
    length. Only an unbroken run of digits is crossed, so ``$5 million or 100m
    of cable`` still blocks its metres.
    """
    stripped = text[max(0, pos - 24):pos].rstrip()
    core = _TRAILING_NUMBER_RE.sub("", stripped).rstrip()
    if not core:
        return False
    if any(core.endswith(sym) for sym in SYMBOL_TO_CODE):
        return True
    tail = core[-3:]
    return bool(re.fullmatch(r"[A-Z]{3}", tail)) and resolve_iso(tail) is not None


class IntervalIndex:
    """Sorted interval set answering "does anything overlap [start, end)?".

    A linear scan per candidate is quadratic in document length — measured at
    3s on a 300 KB narrative, which is an ordinary FOI bundle. Sorting by start
    and carrying a running maximum end answers each query with one bisect: an
    overlap exists iff some interval starting before ``end`` reaches past
    ``start``, and the prefix maximum is exactly that reach.
    """

    __slots__ = ("_max_end", "_starts")

    def __init__(self, spans: list[tuple[int, int]]) -> None:
        items = sorted(spans)
        self._starts = [s for s, _ in items]
        self._max_end: list[int] = []
        reach = -1
        for _, end in items:
            reach = max(reach, end)
            self._max_end.append(reach)

    def overlaps(self, start: int, end: int) -> bool:
        i = bisect_left(self._starts, end)
        return i > 0 and self._max_end[i - 1] > start

__all__ = [
    "IntervalIndex",
    "ambiguous_number_spans",
    "apply_scale",
    "blocked_spans",
    "parse_number",
    "resolve_iso",
    "resolve_symbol",
]
