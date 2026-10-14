"""Form field extraction.

Four sources, in priority order:
1. PDF AcroForm widgets (`_extract_form_fields`) — interactive form
   fields native to the PDF, highest confidence.
2. Spatial text-pair extraction (`_extract_form_pairs_from_text`) —
   walks `page.get_text("dict")` looking for label/value pairs by span
   alignment. Catches text-only forms in native PDFs (NQA notifications,
   provider/service field blocks).
3. Region-based extraction (`_extract_form_pairs_from_regions`) — scans
   per-region OCR output for label/value pairs, preserving each
   region's bbox as the pair's position. Preferred OCR path when the
   engine returns per-region detections (PaddleOCR / RapidOCR).
4. Line-based extraction (`_extract_form_pairs_from_lines`) — scans
   assembled plain text. Bbox is zero (no positional info). Used as a
   fallback when no per-region detections are available (LLM-OCR
   engines that resolve reading order natively and only emit markdown).

Label heuristics: 1–6 words, must start with uppercase, no sentence
punctuation, no list-marker shapes (`A)`, `(i)`), no URL-prefix labels
(`http`, `https`).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from womblex.ingest.extract import (
    FormField,
    Position,
    _normalise_bbox,
    _normalise_rect,
)
from womblex.ingest.interfaces.protocols import OCRRegionResult

if TYPE_CHECKING:
    from womblex.ingest.pdf.types import Page

# Label heuristics: 1–6 words, must start with uppercase, no sentence
# punctuation inside, no list-marker shapes ("A)", "(i)").
_FORM_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9 /'\-()&]{0,49}:?$")
_LIST_MARKER_RE = re.compile(r"^([A-Za-z]\)|\(.+\))$")
_URL_LABEL_LOWER = {"http", "https", "ftp", "file", "ssh", "mailto"}
_FORM_PAIR_GAP_RE = re.compile(r"^([A-Z]\S?(?:.{0,48}?\S)?)\s{2,}(\S.+?)\s*$")

# Labels that recur in Australian regulatory letter prose and aren't real
# form fields — see docs/decisions.md "Element-kind classification".
_LABEL_DENYLIST: frozenset[str] = frozenset({
    "Penalty",      # Regulation citation: "Penalty: $10 000, in the case of an individual"
    "OFFICIAL",     # Document classification banner: "OFFICIAL: Sensitive - Legislative Secrecy"
    "Note",         # Aside: "Note: ..."
    "Caution",      # Email warning banner
})


def _looks_like_form_label(text: str) -> bool:
    text = text.strip().rstrip(":")
    if not text or len(text) > 50:
        return False
    if not text[0].isupper():
        return False
    if text in _LABEL_DENYLIST:
        return False
    if text.lower() in _URL_LABEL_LOWER:
        return False
    words = text.split()
    if not (1 <= len(words) <= 6):
        return False
    if any(c in text for c in ".?!"):
        return False
    if _LIST_MARKER_RE.match(text):
        return False
    return bool(_FORM_LABEL_RE.match(text + ":"))


def _extract_form_fields(page: Page) -> list[FormField]:
    """Extract interactive AcroForm widgets from a page."""
    fields: list[FormField] = []
    pw, ph = page.rect.width, page.rect.height

    for widget in page.widgets():
        pos = _normalise_rect(widget.rect, pw, ph)
        fields.append(FormField(
            field_name=widget.field_name, value=widget.field_value, position=pos, confidence=0.9,
        ))

    return fields


def _extract_form_pairs_from_text(page: Page) -> list[FormField]:
    """Extract label/value pairs from text-only forms (no AcroForm widgets).

    Catches NQA-style notification forms where labels and values are
    separate spans on the same line (`Provider Name    Wonderschool ...`)
    or single spans split by colon (`Provider Name: Wonderschool ...`).
    """
    fields: list[FormField] = []
    pw, ph = page.rect.width, page.rect.height

    for block in page.text_dict():
        if block.kind != "text":
            continue
        for line in block.lines:
            spans = [s for s in line.spans if s.text.strip()]
            if not spans:
                continue

            # Multi-span line — first span = label candidate, rest = value
            if len(spans) >= 2:
                first = spans[0]
                label = first.text.strip().rstrip(":")
                value_parts = [s.text.strip() for s in spans[1:]]
                value = " ".join(p for p in value_parts if p).strip()
                if not value or not _looks_like_form_label(label):
                    continue
                # Require horizontal gap between label end and value start
                gap = spans[1].bbox.x0 - first.bbox.x1
                if gap < 5:
                    continue
                pos = _normalise_bbox(first.bbox.as_tuple(), pw, ph)
                fields.append(
                    FormField(field_name=label, value=value, position=pos, confidence=0.65)
                )
                continue

            # Single span — only catch explicit "Label: value" pattern
            text = spans[0].text.strip()
            if ":" not in text or len(text) > 200:
                continue
            label, _, value = text.partition(":")
            label = label.strip()
            value = value.strip()
            if not value or not _looks_like_form_label(label):
                continue
            pos = _normalise_bbox(spans[0].bbox.as_tuple(), pw, ph)
            fields.append(
                FormField(field_name=label, value=value, position=pos, confidence=0.7)
            )

    return fields


def _extract_forms(page: Page) -> list[FormField]:
    """Extract form fields, preferring AcroForm widgets, falling back to text."""
    fields = _extract_form_fields(page)
    if fields:
        return fields
    return _extract_form_pairs_from_text(page)


def _extract_form_pairs_from_lines(text: str) -> list[FormField]:
    """Scan plain text for label/value pairs without positional info.

    Fallback used only when no per-region OCR detections are available
    (LLM-OCR engines that return markdown / reading-order-resolved text).
    Bbox is zero — callers wanting per-pair locality should prefer
    `_extract_form_pairs_from_regions`.
    Detects same-line pairs separated by ≥2-space gap or by a colon.
    """
    fields: list[FormField] = []
    zero_pos = Position(x=0.0, y=0.0, width=0.0, height=0.0)

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line or len(line) < 4:
            continue

        m = _FORM_PAIR_GAP_RE.match(line)
        if m:
            label = m.group(1).strip().rstrip(":")
            value = m.group(2).strip()
            if value and _looks_like_form_label(label):
                fields.append(
                    FormField(field_name=label, value=value, position=zero_pos, confidence=0.6)
                )
                continue

        if ":" in line:
            label, _, value = line.partition(":")
            label = label.strip()
            value = value.strip()
            if (
                label
                and value
                and not value.startswith("//")
                and _looks_like_form_label(label)
            ):
                fields.append(
                    FormField(field_name=label, value=value, position=zero_pos, confidence=0.65)
                )

    return fields


def _extract_form_pairs_from_regions(
    regions: list[OCRRegionResult],
    pix_width: float,
    pix_height: float,
) -> list[FormField]:
    """Scan per-region OCR results for label/value pairs.

    Preferred OCR path: each region's polygon bbox becomes the pair's
    position (normalised by the OCR-input image dimensions). Closes the
    OCR-form bbox-loss issue (see docs/decisions.md "Element-kind classification").

    Regions are treated independently — multi-region pairs (label and
    value on adjacent detections) are not joined here, since PaddleOCR
    typically yields one detection per visible text line which already
    contains both halves of a `Label: value` or `Label    value` pair.
    """
    fields: list[FormField] = []
    if pix_width <= 0 or pix_height <= 0:
        return fields

    for region in regions:
        line = region.text.strip()
        if not line or len(line) < 4:
            continue

        # Polygon → axis-aligned extent → normalised Position.
        xs = [p[0] for p in region.bbox]
        ys = [p[1] for p in region.bbox]
        pos = _normalise_bbox(
            (min(xs), min(ys), max(xs), max(ys)), pix_width, pix_height,
        )

        m = _FORM_PAIR_GAP_RE.match(line)
        if m:
            label = m.group(1).strip().rstrip(":")
            value = m.group(2).strip()
            if value and _looks_like_form_label(label):
                fields.append(
                    FormField(field_name=label, value=value, position=pos, confidence=0.6)
                )
                continue

        if ":" in line:
            label, _, value = line.partition(":")
            label = label.strip()
            value = value.strip()
            if (
                label
                and value
                and not value.startswith("//")
                and _looks_like_form_label(label)
            ):
                fields.append(
                    FormField(field_name=label, value=value, position=pos, confidence=0.65)
                )

    return fields
