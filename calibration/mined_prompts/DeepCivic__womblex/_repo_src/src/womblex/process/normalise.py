"""OCR / native-text normalisation transforms (downstream cleaning op).

Extraction is **verbatim by policy** — whatever the producing extractor
emits lands on the element ``text`` unchanged (see CLAUDE.md and
``docs/decisions.md`` "Verbatim policy"). Systematic cleanup belongs to
*this* downstream op, never to a normalisation pass at the extraction
boundary.

This module is the pure-function core: each transform takes a string and
returns ``(new_text, n_changes)``. :func:`normalise_text` composes the
enabled transforms in a fixed order for one element's text. The per-stage
driver (:mod:`womblex.process.normalise_stage`) applies it across a shard
directory and writes a ``*.normalised_text.parquet`` sibling — a drop-in
text layer over the narrative elements, mirroring the ``clean_text``
sidecar the PII stage writes.

v1 scope (intra-element):

- :func:`normalise_unicode` — fold unicode whitespace (NBSP, en/em spaces,
  U+2028/9 separators) to ASCII space/newline and strip zero-width marks,
  BOM and stray control chars. Smart quotes / dashes are preserved.
- :func:`collapse_whitespace` — collapse runs of spaces/tabs to one and
  strip per-line trailing whitespace, without touching newlines.
- :func:`despace_page_marker` — heal sub-glyph-kerning ``3|P age`` footers
  (the documented native-text artefact) back to ``3|Page``.
- :func:`apply_substitutions` — literal replacements for known
  letterhead / font-map typos. Empty by default: corpus-driven, never
  hardcoded into core.

Deferred (cross-element, needs a reassembly join-hint): re-joining
redaction-induced paragraph breaks — the PDF text layer returns text either side of
a mid-paragraph redaction bar as separate blocks. See
``docs/decisions.md`` "Deferred / backlog".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Horizontal-whitespace runs (never newlines — line structure is meaningful
# for page mapping and table rejoin). Collapsed to a single space; a lone tab
# counts because it changes, a lone space does not.
_INLINE_WS_RE = re.compile(r"[^\S\n]+")
# Trailing non-newline whitespace at the end of each line.
_TRAILING_WS_RE = re.compile(r"[^\S\n]+(?=\n|$)")

# Sub-glyph-kerning footer artefact: "P age" / "Pa ge" / "P a g e" → "Page".
# Anchored on the literal word so it can't despace arbitrary prose; matches
# the documented "3|P age" native-text footer case only.
_PAGE_MARKER_RE = re.compile(r"\bP\s*a\s*g\s*e\b", re.IGNORECASE)

# Unicode whitespace hygiene (see `normalise_unicode`). Visible unicode spaces
# fold to an ASCII space; line/paragraph separators fold to a newline;
# zero-width marks and the BOM are removed. Punctuation (smart quotes, em/en
# dashes, bullets) is deliberately NOT touched — it is valid, tokeniser-safe
# typography. The producing code points are kept explicit so the transform is
# auditable and the codebook can enumerate them.
_UC_SPACE = frozenset({
    0x00A0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
    0x2007, 0x2008, 0x2009, 0x200A, 0x202F, 0x205F, 0x3000,
})
_UC_NEWLINE = frozenset({0x2028, 0x2029})
_UC_REMOVE = frozenset({0x200B, 0x200C, 0x200D, 0xFEFF})  # zero-width / BOM


@dataclass
class NormaliseTransforms:
    """Which transforms run, in fixed compose order. Mirrors config toggles."""

    unicode_hygiene: bool = True
    collapse_whitespace: bool = True
    despace_page_marker: bool = True
    substitutions: dict[str, str] = field(default_factory=dict)


def collapse_whitespace(text: str) -> tuple[str, int]:
    """Collapse inline whitespace runs and strip per-line trailing whitespace.

    Every horizontal-whitespace run (including a lone tab) becomes a single
    space; trailing whitespace is then stripped per line. Newlines are
    preserved — paragraph/line structure carries page-mapping and table
    semantics downstream. Only runs that actually change are counted, so
    already-clean text returns ``n_changes == 0``. Returns ``(text, n_changes)``.
    """
    count = 0

    def _collapse(m: re.Match[str]) -> str:
        nonlocal count
        if m.group(0) != " ":
            count += 1
        return " "

    def _strip(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return ""

    text = _INLINE_WS_RE.sub(_collapse, text)
    text = _TRAILING_WS_RE.sub(_strip, text)
    return text, count


def despace_page_marker(text: str) -> tuple[str, int]:
    """Heal the spaced-glyph ``P age`` footer artefact back to ``Page``.

    Preserves the case of the leading ``P``. Returns ``(text, n_changes)``.
    """
    def _repl(m: re.Match[str]) -> str:
        return "Page" if m.group(0)[0] == "P" else "page"

    return _PAGE_MARKER_RE.subn(_repl, text)


def normalise_unicode(text: str) -> tuple[str, int]:
    """Fold unicode whitespace to ASCII; strip stray control chars.

    Visible unicode spaces (NBSP, en/em spaces, ideographic space, …) become a
    single ASCII space; line/paragraph separators (U+2028/U+2029) become a
    newline; zero-width marks and the BOM are removed; other control chars
    (except newline/tab) are dropped. Punctuation such as smart quotes and
    em/en dashes is preserved — it is valid, tokeniser-safe typography.
    Returns ``(text, n_changes)`` counting each char folded or removed.
    """
    out: list[str] = []
    n = 0
    for ch in text:
        cp = ord(ch)
        if cp in _UC_SPACE:
            out.append(" ")
            n += 1
        elif cp in _UC_NEWLINE:
            out.append("\n")
            n += 1
        elif cp in _UC_REMOVE:
            n += 1
        elif ch in "\n\t" or unicodedata.category(ch)[0] != "C":
            out.append(ch)
        else:
            n += 1  # stray control char
    return "".join(out), n


def apply_substitutions(text: str, substitutions: dict[str, str]) -> tuple[str, int]:
    """Apply literal ``{find: replace}`` substitutions (longest find first).

    Longest-first avoids a shorter key shadowing a longer overlapping one.
    Returns ``(text, n_changes)`` where ``n_changes`` counts replacements.
    """
    n = 0
    for find in sorted(substitutions, key=len, reverse=True):
        if not find:
            continue
        count = text.count(find)
        if count:
            text = text.replace(find, substitutions[find])
            n += count
    return text, n


def normalise_text(text: str, kind: str, transforms: NormaliseTransforms) -> tuple[str, int]:
    """Apply the enabled transforms to one element's text in fixed order.

    ``kind`` is the element kind (``ElementKind``); ``despace_page_marker``
    only fires on page-furniture kinds (``footer`` / ``header``) where the
    spaced-glyph artefact occurs, so it can't damage body prose containing
    the word "page".
    """
    if not text:
        return text, 0

    total = 0
    if transforms.unicode_hygiene:
        text, n = normalise_unicode(text)
        total += n
    if transforms.substitutions:
        text, n = apply_substitutions(text, transforms.substitutions)
        total += n
    if transforms.despace_page_marker and kind in ("footer", "header"):
        text, n = despace_page_marker(text)
        total += n
    if transforms.collapse_whitespace:
        text, n = collapse_whitespace(text)
        total += n
    return text, total


__all__ = [
    "NormaliseTransforms",
    "apply_substitutions",
    "collapse_whitespace",
    "despace_page_marker",
    "normalise_text",
    "normalise_unicode",
]
