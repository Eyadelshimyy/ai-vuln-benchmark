"""Recover the ligature glyphs a PDF's /ToUnicode map forgets.

Word embeds one subset font per style and sometimes leaves the ligature glyph out of
the map, so pypdf falls back to the raw byte and 'instituciones' arrives as
'insVtuciones'. The same font still writes a genuine 'V' elsewhere, so once the text
exists the two are the same character and nothing can tell them apart: the repair has
to happen on the font. The evidence is the advance width in /Widths — a glyph as wide
as 't' plus 'i', and as wide as no other pair, is the 'ti' ligature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.generic import DictionaryObject, StreamObject

# The ligatures a Latin text font actually carries; anything else stays unknown.
_LIGATURES: tuple[str, ...] = (
    "ffi",
    "ffl",
    "ff",
    "fi",
    "fl",
    "ft",
    "fj",
    "ti",
    "tt",
    "st",
    "ct",
)

_HEX = r"<([0-9a-fA-F]+)>"
_BFCHAR_RE = re.compile(r"beginbfchar(.*?)endbfchar", re.DOTALL)
_BFRANGE_RE = re.compile(r"beginbfrange(.*?)endbfrange", re.DOTALL)
_PAIR_RE = re.compile(rf"{_HEX}\s*{_HEX}")
_TRIPLE_RE = re.compile(rf"{_HEX}\s*{_HEX}\s*{_HEX}")
_ARRAY_RE = re.compile(rf"{_HEX}\s*{_HEX}\s*\[(.*?)\]", re.DOTALL)
_SUBSET_TAG_RE = re.compile(r"^[A-Z]{6}\+")


@dataclass
class _Font:
    """A simple font, its per-character widths, and the codes it never explains."""

    base_name: str
    to_unicode: StreamObject
    char_widths: dict[str, int]
    holes: dict[int, int]


def repair_missing_ligatures(reader: PdfReader) -> list[str]:
    """Fill the gaps in every /ToUnicode map, in place; warn about what stays unknown."""
    fonts = _simple_fonts(reader)
    by_family: dict[tuple[str, int], str] = {}
    filled: list[dict[int, str]] = []

    for font in fonts:
        found: dict[int, str] = {}
        for code, width in font.holes.items():
            ligature = _ligature_for_width(width, font.char_widths)
            if ligature is not None:
                found[code] = ligature
                by_family[(font.base_name, width)] = ligature
        filled.append(found)

    warnings: list[str] = []
    for font, found in zip(fonts, filled, strict=True):
        # A tiny subset may lack the letters that prove the width; a sibling subset of
        # the same font, at the same width, already proved it.
        for code, width in font.holes.items():
            if code in found:
                continue
            sibling = by_family.get((font.base_name, width))
            if sibling is not None:
                found[code] = sibling
            else:
                warnings.append(_hole_warning(font.base_name, code))
        if found:
            _add_mappings(font.to_unicode, found)

    return warnings


def _hole_warning(base_name: str, code: int) -> str:
    shown = chr(code) if 0x20 < code < 0x7F else f"byte {code:#04x}"
    return (
        f"{base_name}: one glyph carries no Unicode and no width to identify it, "
        f"so '{shown}' appears inside words where a ligature belongs"
    )


def _simple_fonts(reader: PdfReader) -> list[_Font]:
    """Every font that maps single bytes and declares its widths, listed once."""
    fonts: list[_Font] = []
    seen: set[int] = set()

    for page in reader.pages:
        resources = _resolve(page.get("/Resources"))
        if not isinstance(resources, DictionaryObject):
            continue
        page_fonts = _resolve(resources.get("/Font"))
        if not isinstance(page_fonts, DictionaryObject):
            continue
        for reference in page_fonts.values():
            font = reference.get_object()
            if not isinstance(font, DictionaryObject):
                continue
            parsed = _read_font(font)
            if parsed is None or id(parsed.to_unicode) in seen:
                continue
            seen.add(id(parsed.to_unicode))
            fonts.append(parsed)

    return fonts


def _resolve(value: object) -> object:
    getter = getattr(value, "get_object", None)
    return getter() if callable(getter) else value


def _read_font(font: DictionaryObject) -> _Font | None:
    to_unicode = font.get("/ToUnicode")
    widths = _resolve(font.get("/Widths"))
    first_char = _resolve(font.get("/FirstChar"))
    if to_unicode is None or widths is None or first_char is None:
        return None

    stream = to_unicode.get_object()
    if not isinstance(stream, StreamObject):
        return None
    if not isinstance(widths, list) or not isinstance(first_char, int | float):
        return None

    try:
        mapping = _parse_to_unicode(stream.get_data().decode("latin-1"))
        first = int(first_char)
        by_code = {first + offset: int(width) for offset, width in enumerate(widths)}
    except (ValueError, TypeError, UnicodeDecodeError):
        return None

    char_widths: dict[str, int] = {}
    for code, text in mapping.items():
        width = by_code.get(code)
        if len(text) == 1 and width is not None:
            char_widths.setdefault(text, width)

    holes = {
        code: width
        for code, width in by_code.items()
        if code not in mapping and width > 0
    }
    if not holes:
        return None

    return _Font(
        base_name=_base_name(font.get("/BaseFont")),
        to_unicode=stream,
        char_widths=char_widths,
        holes=holes,
    )


def _base_name(base_font: object) -> str:
    """'/AAAAAK+Calibri-Bold' → 'Calibri-Bold': the subset tag differs per style run."""
    name = str(base_font or "font").lstrip("/")
    return _SUBSET_TAG_RE.sub("", name)


def _ligature_for_width(width: int, char_widths: dict[str, int]) -> str | None:
    """The one ligature whose parts are as wide as this glyph, or None when unclear."""
    matches = [
        ligature
        for ligature in _LIGATURES
        if all(char in char_widths for char in ligature)
        and _fits(width, sum(char_widths[char] for char in ligature))
    ]
    return matches[0] if len(matches) == 1 else None


def _fits(width: int, parts: int) -> bool:
    """A ligature is drawn tighter than its parts, never wider beyond rounding."""
    slack = parts - width
    return -2 <= slack <= max(8, round(parts * 0.02))


def _parse_to_unicode(data: str) -> dict[int, str]:
    mapping: dict[int, str] = {}

    for body in _BFRANGE_RE.findall(data):
        for first, last, targets in _ARRAY_RE.findall(body):
            codes = range(int(first, 16), int(last, 16) + 1)
            for code, target in zip(codes, re.findall(_HEX, targets), strict=False):
                mapping[code] = _decode(target)
        for first, last, target in _TRIPLE_RE.findall(_ARRAY_RE.sub(" ", body)):
            _map_range(mapping, int(first, 16), int(last, 16), target)

    for body in _BFCHAR_RE.findall(data):
        for code, target in _PAIR_RE.findall(body):
            mapping[int(code, 16)] = _decode(target)

    return mapping


def _map_range(mapping: dict[int, str], first: int, last: int, target: str) -> None:
    if len(target) > 4:
        # A multi-character target cannot be incremented; it names the whole range.
        text = _decode(target)
        for code in range(first, last + 1):
            mapping[code] = text
        return
    base = int(target, 16)
    for offset, code in enumerate(range(first, last + 1)):
        mapping[code] = chr(base + offset)


def _decode(target: str) -> str:
    padded = target if len(target) % 4 == 0 else target.zfill(4)
    return bytes.fromhex(padded).decode("utf-16-be", errors="replace")


def _add_mappings(stream: StreamObject, mapping: dict[int, str]) -> None:
    entries = "\n".join(
        f"<{code:02x}> <{text.encode('utf-16-be').hex()}>"
        for code, text in sorted(mapping.items())
    )
    block = f"\n{len(mapping)} beginbfchar\n{entries}\nendbfchar\n"
    data = stream.get_data().decode("latin-1")
    patched = data.replace("endcmap", block + "endcmap", 1) if "endcmap" in data else data + block
    stream.set_data(patched.encode("latin-1"))
