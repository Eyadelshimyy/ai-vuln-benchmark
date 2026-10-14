"""Legacy view types + ExtractionResult.

Split out of ``extract.py`` to keep the file under the 750-line cap.
The dataclasses below are read-only projections over the canonical
``Element`` stream defined in ``ingest/elements.py``. Downstream PII /
redact / chunk stages that haven't migrated still see the legacy
shape via ``ExtractionResult`` properties.

``Position`` aliases ``BBox`` so existing constructors keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from womblex.ingest.elements import TEXT_KINDS, BBox, Cell, Element

if TYPE_CHECKING:
    from womblex.ingest.layout_step import LayoutOutcome
    from womblex.redact.stage import RedactionReport


# ---------------------------------------------------------------------------
# Legacy view types
# ---------------------------------------------------------------------------

Position = BBox


@dataclass
class TableData:
    """Legacy view of a kind='table' element. Single-header-row shape;
    merges (rowspan / colspan) are dropped on this projection.
    """

    headers: list[str]
    rows: list[list[str]]
    position: Position
    confidence: float
    context: dict[str, str] = field(default_factory=dict)
    cell_boxes: dict[tuple[int, int], Position] = field(default_factory=dict)
    """Normalised cell boxes by (row, col) in the headers-then-rows grid; absent where unknown."""


@dataclass
class FormField:
    """Legacy view of one (label, value) pair from a kind='form' element."""

    field_name: str
    value: str
    position: Position
    confidence: float


@dataclass
class ImageData:
    """Legacy view of a kind='image' element."""

    alt_text: str
    position: Position
    confidence: float


@dataclass
class TextBlock:
    """Legacy view of a text-bearing element (paragraph / heading / etc.)."""

    text: str
    position: Position
    block_type: str
    confidence: float


# ---------------------------------------------------------------------------
# Forward projection (view -> element)
# ---------------------------------------------------------------------------


def table_to_element(
    t: TableData, page: int | None, extractor: str, order: int,
) -> Element:
    """Build a kind='table' element by re-cellifying a legacy TableData.

    Headers become row 0; data rows shift to rows 1..n. Header row index
    captured in ``header_rows`` so the legacy projection round-trips.

    Sits beside the ``TableData`` view it reads and the reverse projections
    below, so both directions of the view↔element mapping are in one file.
    """
    cells: list[Cell] = []
    boxes = t.cell_boxes
    for col_idx in range(len(t.headers)):
        cells.append(Cell(row=0, col=col_idx, value=t.headers[col_idx], bbox=boxes.get((0, col_idx))))
    for row_idx, row in enumerate(t.rows, start=1):
        for col_idx in range(len(row)):
            cells.append(Cell(
                row=row_idx, col=col_idx, value=row[col_idx], bbox=boxes.get((row_idx, col_idx)),
            ))
    return Element(
        order=order, kind="table", extractor=extractor,
        page=page, bbox=t.position,
        cells=cells, header_rows=[0] if t.headers else [],
        confidence=t.confidence,
        meta={**({"context_" + k: v for k, v in t.context.items()} if t.context else {})},
    )


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass
class PageResult:
    """Extracted text for a single page."""

    page_number: int
    text: str
    method: str


@dataclass
class ExtractionMetadata:
    """Document-level extraction metadata."""

    extraction_strategy: str
    confidence: float
    processing_time: float
    page_count: int
    text_coverage: float
    preprocessing_steps: list[str] = field(default_factory=list)
    content_mix: dict[str, float] = field(default_factory=dict)


_ZERO_POS = Position(x=0.0, y=0.0, width=0.0, height=0.0)


@dataclass
class ExtractionResult:
    """Result of text extraction from a document.

    ``elements`` is the canonical structural stream — what the parquet
    writer serialises. ``pages`` carries per-page concatenated text and
    is mutable: downstream PII / redaction stages mutate ``page.text``
    in place. The on-disk parquet retains extraction-time verbatim text
    because the writer reads ``elements``, not ``pages``.

    ``text_blocks`` / ``tables`` / ``forms`` / ``images`` are read-only
    derived views over ``elements`` for callers that haven't migrated.
    Mutating these views has no effect on the source elements.
    """

    pages: list[PageResult] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)
    method: str = ""
    error: str | None = None
    document_metadata: dict[str, str] = field(default_factory=dict)
    metadata: ExtractionMetadata | None = None
    warnings: list[str] = field(default_factory=list)
    document_id: str | None = None
    redaction_report: RedactionReport | None = None
    layout: LayoutOutcome | None = None

    @property
    def full_text(self) -> str:
        """Concatenate page texts (reflects in-memory PII / redaction mutations)."""
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text_blocks(self) -> list[TextBlock]:
        """Read-only view: TextBlock per text-bearing element."""
        return [
            TextBlock(
                text=e.text or "",
                position=e.bbox or _ZERO_POS,
                block_type=e.kind,
                confidence=e.confidence,
            )
            for e in self.elements
            if e.kind in TEXT_KINDS
        ]

    @property
    def tables(self) -> list[TableData]:
        """Read-only view: TableData per kind='table' element, plus one
        synthetic TableData per spreadsheet sheet so the chunker and
        other table-oriented downstream consumers see a unified table view.
        """
        out = [_element_to_table_data(e) for e in self.elements if e.kind == "table"]
        out.extend(_sheets_to_table_data(self.elements))
        return out

    @property
    def forms(self) -> list[FormField]:
        """Read-only view: flat FormField list across all kind='form' elements."""
        return [
            FormField(
                field_name=f.name,
                value=f.value,
                position=e.bbox or _ZERO_POS,
                confidence=e.confidence,
            )
            for e in self.elements
            if e.kind == "form" and e.fields
            for f in e.fields
        ]

    @property
    def images(self) -> list[ImageData]:
        """Read-only view: ImageData per kind='image' element."""
        return [
            ImageData(
                alt_text=e.alt_text or "",
                position=e.bbox or _ZERO_POS,
                confidence=e.confidence,
            )
            for e in self.elements
            if e.kind == "image"
        ]


# ---------------------------------------------------------------------------
# Element → legacy view helpers
# ---------------------------------------------------------------------------


def _sheets_to_table_data(elements: list[Element]) -> list[TableData]:
    """Aggregate sheet_cell elements into one TableData per sheet.

    Row 0 becomes ``headers``; remaining rows go to ``rows``. Preserves
    the legacy "spreadsheet = list of tables" view that downstream
    chunking relies on.
    """
    sheets: dict[str, dict[int, dict[int, str]]] = {}
    sheet_order: list[str] = []
    for e in elements:
        if e.kind != "sheet_cell" or e.sheet is None or e.row is None or e.col is None:
            continue
        if e.sheet not in sheets:
            sheets[e.sheet] = {}
            sheet_order.append(e.sheet)
        sheets[e.sheet].setdefault(e.row, {})[e.col] = e.value or ""

    out: list[TableData] = []
    for sheet in sheet_order:
        rows = sheets[sheet]
        if not rows:
            continue
        max_col = max(max(cols) for cols in rows.values())
        sorted_indices = sorted(rows)
        materialised: list[list[str]] = []
        for idx in sorted_indices:
            row = [""] * (max_col + 1)
            for c, v in rows[idx].items():
                row[c] = v
            materialised.append(row)
        headers = materialised[0] if materialised else []
        data_rows = materialised[1:] if len(materialised) > 1 else []
        out.append(TableData(
            headers=headers,
            rows=data_rows,
            position=_ZERO_POS,
            confidence=0.95,
            context={"sheet": sheet},
        ))
    return out


def _element_to_table_data(elem: Element) -> TableData:
    """Flatten a kind='table' element into the legacy single-header TableData.

    The first declared header row becomes ``headers``; remaining rows
    become ``rows``. Merges (rowspan / colspan) are not represented in
    the legacy shape and are dropped on this projection.
    """
    cells = elem.cells or []
    if not cells:
        return TableData(
            headers=[], rows=[],
            position=elem.bbox or _ZERO_POS,
            confidence=elem.confidence,
        )
    max_col = max(c.col for c in cells)
    headers_set = set(elem.header_rows or [])
    by_row: dict[int, list[str]] = {}
    for c in cells:
        row = by_row.setdefault(c.row, [""] * (max_col + 1))
        if c.col >= len(row):
            row.extend([""] * (c.col + 1 - len(row)))
        row[c.col] = c.value
    headers: list[str] = []
    rows: list[list[str]] = []
    for idx in sorted(by_row):
        row = by_row[idx]
        if len(row) < max_col + 1:
            row.extend([""] * (max_col + 1 - len(row)))
        if idx in headers_set and not headers:
            headers = row
        else:
            rows.append(row)
    return TableData(
        headers=headers,
        rows=rows,
        position=elem.bbox or _ZERO_POS,
        confidence=elem.confidence,
    )
