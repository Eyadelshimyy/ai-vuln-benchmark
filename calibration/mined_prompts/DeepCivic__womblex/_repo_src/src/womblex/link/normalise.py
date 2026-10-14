"""Minimal normalisation for entity-link matching.

Casefold + strip punctuation + expand common street abbreviations + drop
state / PO-box tokens. This picks up the surface variation Kanon-2 leaves
in (``PTYLTD``, newline-split names, mixed case, OCR spacing) so that
normalised equality on addresses and fuzzy similarity on names both
improve.

This is deliberately *minimal*. It is NOT address validation — there is
no G-NAF lookup here; canonical address resolution is a separate,
corpus-dependent concern (see docs/decisions.md "Entity linking").
"""

from __future__ import annotations

import re

# Street-type abbreviations seen in AU government address columns.
_STREET_ABBREV = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "cres": "crescent", "cct": "circuit", "pl": "place", "dr": "drive",
    "ln": "lane", "hwy": "highway", "tce": "terrace", "pde": "parade",
    "ct": "court", "cl": "close", "blvd": "boulevard", "sq": "square",
}
# Standalone tokens dropped from addresses (state codes vary by OCR/source).
_STATE_TOKENS = frozenset({"act", "nsw", "vic", "qld", "sa", "wa", "tas", "nt", "australia"})

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_PO_BOX_RE = re.compile(r"\b(?:g?po)\s*box\s*\d+\b")


def normalise_name(text: str | None) -> str:
    """Casefold, strip punctuation, collapse whitespace. Empty-safe."""
    if not text:
        return ""
    t = text.replace("\n", " ").casefold()
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def normalise_address(text: str | None) -> str:
    """Normalise an address: drop PO boxes + state tokens, expand street types.

    Lets ``"11 Cessnock st, Fyshwick ACT 2609"`` and a register's
    ``"11 Cessnock St" + "FYSHWICK" + "2609"`` collapse to the same string,
    which is the OCR-robust match key for the link stage.
    """
    if not text:
        return ""
    t = text.replace("\n", " ").casefold()
    t = _PO_BOX_RE.sub(" ", t)
    t = _PUNCT_RE.sub(" ", t)
    tokens = [
        _STREET_ABBREV.get(tok, tok)
        for tok in t.split()
        if tok not in _STATE_TOKENS
    ]
    return _WS_RE.sub(" ", " ".join(tokens)).strip()
