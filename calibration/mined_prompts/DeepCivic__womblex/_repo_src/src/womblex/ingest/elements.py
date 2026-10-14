"""Element-stream model for extraction output.

An ExtractionResult is an ordered sequence of Elements. Each element is
one structural atom of the source: a paragraph, table, form, image,
page break, or — for spreadsheets — a cell or sheet boundary.

Tables nest their cells and forms nest their fields in memory. The
parquet writer denormalises these attachments into sidecar files
(``table_cells.parquet`` / ``form_fields.parquet``) keyed by
``(source_hash, parent_elem_order)``.

Text is verbatim from the producing extractor. Extraction applies no
post-processing; downstream stages (PII, redaction) operate on
``ExtractionResult.pages`` and do not modify elements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------

ElementKind = Literal[
    "paragraph",
    "heading",
    "list_item",
    "caption",
    "header",
    "footer",
    "footnote",
    "signature",
    "figure",
    "table",
    "form",
    "image",
    "page_break",
    "sheet_meta",
    "sheet_cell",
]

TEXT_KINDS: frozenset[str] = frozenset({
    "paragraph", "heading", "list_item", "caption",
    "header", "footer", "footnote", "signature",
})


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BBox:
    """Normalised 0–1 page coordinates, top-left origin."""

    x: float
    y: float
    width: float
    height: float


# ---------------------------------------------------------------------------
# Nested attachments
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Cell:
    """One cell of a table element. Verbatim; value_type is a hint, not a coercion.

    ``bbox`` is normalised like ``Element.bbox``; null where the producer has no
    cell geometry (DOCX, OCR-reconstructed and spreadsheet-print tables).
    """

    row: int
    col: int
    value: str
    rowspan: int = 1
    colspan: int = 1
    value_type: str = "text"
    bbox: BBox | None = None


@dataclass(slots=True)
class FieldEntry:
    """One field of a form element."""

    name: str
    value: str
    field_type: str = "text"


# ---------------------------------------------------------------------------
# Element
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Element:
    """One structural atom of a source document.

    Populated per ``kind``:

    - text kinds (paragraph, heading, list_item, caption, header, footer, signature):
      ``text`` carries the verbatim string.
    - ``table``: ``cells`` holds the dense list of Cell entries;
      ``header_rows`` lists row indices that act as headers.
    - ``form``: ``fields`` holds the dense list of FieldEntry entries.
    - ``image`` / ``figure``: ``alt_text`` is whatever the extractor produced.
    - ``sheet_cell``: ``sheet`` / ``row`` / ``col`` / ``value`` /
      ``value_type``, plus optional ``formula`` / ``number_format`` /
      ``merge_range``. ``merge_range`` is the merge's openpyxl address
      (e.g. ``"A1:C1"``), set only on the merge's top-left cell.
    - ``sheet_meta``: ``sheet`` plus ``meta`` keys for sheet-level
      properties (column widths, hidden flags, ordinal).
    - ``page_break``: ``page`` only; no payload.

    ``order`` is monotonic across the whole source. Consumers reassemble
    a source by ``ORDER BY order``.
    """

    order: int
    kind: ElementKind
    extractor: str
    confidence: float = 0.0

    # Document layout
    page: int | None = None
    bbox: BBox | None = None

    # Text-bearing kinds
    text: str | None = None

    # Tables
    cells: list[Cell] | None = None
    header_rows: list[int] | None = None

    # Forms
    fields: list[FieldEntry] | None = None

    # Images / figures
    alt_text: str | None = None

    # Spreadsheet location
    sheet: str | None = None
    row: int | None = None
    col: int | None = None

    # Spreadsheet cell payload
    value: str | None = None
    value_type: str | None = None
    formula: str | None = None
    number_format: str | None = None
    merge_range: str | None = None

    # Parser-specific overflow; typed fields above cover the common cases.
    meta: dict[str, str] = field(default_factory=dict)
