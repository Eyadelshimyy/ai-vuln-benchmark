"""Spreadsheet-printed-to-PDF extractor.

Government FOI releases frequently distribute manifests, schedules, and
indexes as native PDFs that were printed from a spreadsheet (CSV/Excel).
The native text layer is intact but the row/column structure lives only
in **consistent x-positions of text spans across many y-bands** — there
are no ruled cell borders. The page's `find_tables(strategy="text")` is
too conservative for this layout: it sees the columns but won't commit
to row groupings, so it returns headers-as-cells and 0–1-row tables.

This module dedicates a per-doc primitive that:

1. Infers column boundaries by 1-D clustering x-positions of all spans.
2. Bins spans into y-bands, then assigns each span to its column by
   nearest-left x-position. Multi-span cells join with single space.
3. Detects the first full-coverage row as headers; deduplicates frozen
   header rows that repeat at the top of subsequent pages.
4. Captures the metadata block above the first data row (e.g.
   ``213A reference / 213A-2025-008``, ``Element # / 2(a)(i)–2(a)(iv)``)
   as label-value pairs. Whether this lives on the table, on the
   document, or both is config-driven.
5. Accumulates rows across pages into a single ``TableData`` rather than
   one-table-per-page.

Triggered by a cheap qualifier from ``PageProfile`` (text layer +
table signal + char density) plus optional filename hints, then
structurally vetted (column count + row coverage) before this primitive
runs. See ``page_profile.qualify_for_spreadsheet_print``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from womblex.ingest.extract import TableData, _normalise_bbox
from womblex.ingest.table_grid import (
    COLUMN_X_TOLERANCE_PT,
    ROW_BAND_PT,
    Column,
    Span,
    bands_to_rows,
    bin_y_bands,
    columns_from_data,
    drop_blank_rows,
)

if TYPE_CHECKING:
    from womblex.ingest.pdf.types import Document

# Minimum columns the page must expose for spreadsheet-print routing.
MIN_COLUMNS = 3
# Header zone vertical tolerance — spans within this y-distance of the
# anchor header band are treated as multi-line header continuations
# (e.g. "Subsection \n code", "Out of Scope \n Exemption").
HEADER_MERGE_PX = 12.0
# Header band must span at least this fraction of page width.
HEADER_X_SPREAD_RATIO = 0.4
# Header band must contain at least this many spans.
HEADER_MIN_SPANS = 4


@dataclass(slots=True)
class _PageData:
    spans: list[Span]
    width: float
    height: float


def extract_spreadsheet_print(
    doc: Document,
    *,
    metadata_location: str = "both",
) -> tuple[list[TableData], dict[str, str]]:
    """Extract a spreadsheet-print PDF as one multi-page table + metadata.

    Returns ``(tables, document_metadata)``. The list contains a single
    ``TableData`` whose ``rows`` accumulate across all pages of the doc.
    Whether the metadata block populates the table's ``context`` and/or
    ``document_metadata`` depends on ``metadata_location``:
    ``"table"``, ``"document"``, or ``"both"`` (default).

    Returns ``([], {})`` if the doc doesn't structurally vet (no header band
    with enough columns).
    """
    if metadata_location not in ("both", "table", "document"):
        raise ValueError(f"metadata_location must be both|table|document, got {metadata_location!r}")

    pages = _collect_pages(doc)
    if not pages:
        return [], {}

    # Locate header band on page 0 — wide-spread band with most spans is the
    # canonical anchor. Multi-line header lines (e.g. "Subsection \n code",
    # "Out of Scope \n Exemption") merge in within HEADER_MERGE_PX vertically.
    page0 = pages[0]
    header_anchor = _find_header_band(page0.spans, page0.width)
    if header_anchor is None:
        return [], {}
    header_y, _ = header_anchor

    bands_p0 = bin_y_bands(page0.spans)
    header_spans = _merge_multi_line_header(bands_p0, header_y)
    if len(header_spans) < MIN_COLUMNS:
        return [], {}

    # Header zone vertical range — bands inside this range are the (possibly
    # multi-line) header itself and must be excluded from body extraction on
    # every page (frozen-header dedup). Bottom is the latest header span
    # y_top + small buffer.
    header_zone_top = min(s.y_top for s in header_spans) - HEADER_MERGE_PX
    header_zone_bottom = max(s.y_top for s in header_spans) + ROW_BAND_PT

    # Metadata block: y-bands strictly above the header zone.
    metadata_bands = [(by, bs) for by, bs in bands_p0 if by < header_zone_top]
    metadata = _parse_metadata_block_from_bands(metadata_bands)

    # Column boundaries derived from DATA spans (below header zone), not
    # header spans. Many spreadsheets centre header text within a wider
    # cell; using header positions misaligns left-anchored data.
    data_spans: list[Span] = []
    for page in pages:
        for s in page.spans:
            if s.y_top > header_zone_bottom:
                data_spans.append(s)
    expected_k = _distinct_header_columns(header_spans)
    columns = columns_from_data(data_spans, page0.width, expected_k=expected_k)
    if len(columns) < MIN_COLUMNS:
        return [], {}
    headers = _header_text_from_spans(header_spans, columns)

    all_rows: list[list[str]] = []
    bbox_min_x = columns[0].x_left
    bbox_max_x = columns[-1].x_right
    bbox_min_y = header_y
    bbox_max_y = 0.0

    for page_idx, page in enumerate(pages):
        bands = bin_y_bands(page.spans)
        # Drop bands inside the header zone — frozen headers repeat per page.
        body_bands = [
            (by, bs) for by, bs in bands
            if by > header_zone_bottom
        ]
        column_rows = bands_to_rows(body_bands, columns)
        all_rows.extend(drop_blank_rows(column_rows))
        if body_bands:
            bbox_max_y = max(bbox_max_y, body_bands[-1][0])

    if not headers:
        return [], {}

    # Build position from union bbox. Falls back to page-0 dimensions.
    pw, ph = pages[0].width, pages[0].height
    if bbox_min_x == float("inf"):
        bbox_min_x = bbox_min_y = 0.0
        bbox_max_x, bbox_max_y = pw, ph
    pos = _normalise_bbox(
        (bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y), pw, ph
    )

    table_context = metadata if metadata_location in ("table", "both") else {}
    doc_metadata = metadata if metadata_location in ("document", "both") else {}

    table = TableData(
        headers=headers,
        rows=all_rows,
        position=pos,
        confidence=0.7,
        context=table_context,
    )
    return [table], doc_metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_pages(doc: Document) -> list[_PageData]:
    """Walk the doc and gather text spans per page.

    Rotation: ``page.text_dict()`` returns bboxes in unrotated mediabox
    coordinates. Spreadsheet-print PDFs are commonly rotated 90° (landscape
    spreadsheet rendered to a portrait mediabox + rotation flag) — we apply
    ``page.rotation_matrix`` so downstream column/row inference works in
    displayed coordinates.
    """
    pages: list[_PageData] = []
    for page in doc:
        rotation = page.rotation
        rotation_matrix = page.rotation_matrix if rotation else None

        spans: list[Span] = []
        for block in page.text_dict():
            if block.kind != "text":
                continue
            for line in block.lines:
                for span in line.spans:
                    text = span.text.strip()
                    if not text:
                        continue
                    # `Rect.transform` re-normalises so x0<x1 and y0<y1.
                    box = span.bbox if rotation_matrix is None else span.bbox.transform(rotation_matrix)
                    spans.append(Span(
                        y_top=box.y0, y_bottom=box.y1,
                        x_left=box.x0, x_right=box.x1, text=text,
                    ))
        pages.append(_PageData(spans=spans, width=page.rect.width, height=page.rect.height))
    return pages


def _find_header_band(
    spans: list[Span], page_width: float,
) -> tuple[float, list[Span]] | None:
    """Find the y-band most likely to be the table's anchor header row.

    Selects the topmost band that has ≥ HEADER_MIN_SPANS spans and an
    x-spread ≥ HEADER_X_SPREAD_RATIO × page width. "Topmost" matters
    because the metadata block above the table can have wide-spread bands
    too (label-then-value pairs); the table header is the *first* such
    band that also clears HEADER_MIN_SPANS.
    """
    if not spans:
        return None
    bands = bin_y_bands(spans)
    for by, bs in bands:
        if len(bs) < HEADER_MIN_SPANS:
            continue
        x_spread = max(s.x_right for s in bs) - min(s.x_left for s in bs)
        if x_spread < page_width * HEADER_X_SPREAD_RATIO:
            continue
        return by, bs
    return None


def _merge_multi_line_header(
    bands: list[tuple[float, list[Span]]], anchor_y: float,
) -> list[Span]:
    """Collect anchor band's spans plus nearby bands within HEADER_MERGE_PX.

    Spreadsheet headers like "Subsection \n code" and "Out of Scope \n
    Exemption" render as adjacent y-bands. We pull all spans from bands
    whose y is within ``HEADER_MERGE_PX`` of the anchor into a single
    synthetic header.
    """
    merged: list[Span] = []
    for by, bs in bands:
        if abs(by - anchor_y) <= HEADER_MERGE_PX:
            merged.extend(bs)
    return merged


def _distinct_header_columns(header_spans: list[Span]) -> int:
    """Count distinct x_left clusters in the header band.

    Used as the expected column count when picking data clusters: avoids
    spurious mid-column data clusters (variable-width Author values etc.)
    inflating the column count.
    """
    if not header_spans:
        return 0
    xs = sorted(s.x_left for s in header_spans)
    clusters = 1
    for i in range(1, len(xs)):
        if xs[i] - xs[i - 1] > COLUMN_X_TOLERANCE_PT:
            clusters += 1
    return clusters


def _header_text_from_spans(
    header_spans: list[Span], columns: list[Column],
) -> list[str]:
    """Compose per-column header text from the merged header spans.

    Headers are typically centred within wider cells, so a header span's
    x_left often does NOT match the data column's x_left (which is
    left-aligned). We assign each header span to the column whose
    midpoint is nearest the header span's midpoint, then space-join
    multi-line header pieces in y_top order.
    """
    if not columns:
        return []
    col_midpoints = [(c.x_left + c.x_right) / 2 for c in columns]
    cells: list[list[tuple[float, str]]] = [[] for _ in columns]
    for s in header_spans:
        s_mid = (s.x_left + s.x_right) / 2
        nearest = min(
            range(len(columns)),
            key=lambda i: abs(s_mid - col_midpoints[i]),
        )
        cells[nearest].append((s.y_top, s.text))
    return [
        " ".join(t for _, t in sorted(parts, key=lambda yt: yt[0]))
        for parts in cells
    ]


# Metadata-block parsing -----------------------------------------------------

# Match "Label: value" on a single span.
_KV_COLON_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 #/'\-()&]{1,40})\s*:\s*(.+\S)\s*$")
# Match a label line that doesn't end with "value-shaped" tail. Used in the
# label-then-value pairing fallback.
_LABEL_LINE_RE = re.compile(r"^[A-Z][A-Za-z0-9 #/'\-()&]{0,40}$")


def _parse_metadata_block_from_bands(
    bands_above: list[tuple[float, list[Span]]],
) -> dict[str, str]:
    """Parse the metadata block above the first data row into label-value pairs.

    Three patterns handled in priority order:
    1. Same band has 2+ spans → first span is label, rest joined as value.
    2. Single-span ``Label: value`` → split on colon.
    3. Consecutive bands where band N is a label-shape and band N+1 is its
       value → pair them.
    """
    pairs: dict[str, str] = {}
    pending_label: str | None = None

    for _band_y, spans in bands_above:
        texts = [s.text for s in spans if s.text.strip()]
        if not texts:
            pending_label = None
            continue

        # Pattern 1: 2+ spans in the same band — first is label, rest is value.
        if len(texts) >= 2:
            label = texts[0].strip().rstrip(":")
            value = " ".join(texts[1:]).strip()
            if label and value:
                pairs[label] = value
                pending_label = None
                continue

        # Pattern 2: single span with explicit colon split.
        cell = texts[0].strip()
        m = _KV_COLON_RE.match(cell)
        if m:
            pairs[m.group(1).strip().rstrip(":")] = m.group(2).strip()
            pending_label = None
            continue

        # Pattern 3: consecutive-band label-then-value.
        if pending_label is None and _LABEL_LINE_RE.match(cell):
            pending_label = cell
        elif pending_label is not None:
            pairs[pending_label] = cell
            pending_label = None

    return pairs
