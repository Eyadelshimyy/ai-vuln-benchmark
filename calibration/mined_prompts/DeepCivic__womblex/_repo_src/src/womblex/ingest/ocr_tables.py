"""OCR table feeder: reconstruct cells inside a layout-detected table rect.

The second feeder of the shared ``table_grid`` algorithm (the first is
``spreadsheet_print``). An OCR quad reduces exactly to a ``Span``; the
lifted binning/clustering then infers the grid. Round 1 targets flat,
clean contemporary tables — ``reconstruct_table`` returns ``None`` (never
a partial) whenever the grid fails its precision gates, so refusal on a
hard shape is a correct outcome. See
docs/decisions.md “Table-cell reconstruction on OCR pages”.

All coordinates here are image pixels at the caller's render dpi; the
point-space ``table_grid`` tolerances are scaled by ``dpi / 72``.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from womblex.ingest.extract import TableData, _normalise_bbox
from womblex.ingest.interfaces.protocols import OCRRegionResult
from womblex.ingest.table_grid import (
    COLUMN_X_TOLERANCE_PT,
    DATA_CLUSTER_GAP_PT,
    LAST_COLUMN_PAD_PT,
    ROW_BAND_PT,
    Span,
    bands_to_rows,
    bin_y_bands,
    column_for_x,
    columns_from_data,
    drop_blank_rows,
)

logger = logging.getLogger(__name__)

# Precision gates. Below any gate the reconstructor refuses (returns
# None); a wrongly-binned grid is worse than today's silence. Calibrated
# against the rendered-clean fixtures (must pass) and the false-table set
# (must refuse) — see the table benchmark in the womblex-benchmark
# repository, and the table section of docs/evaluation.md.
MIN_COLUMNS = 3
# Three, not two: ``columns_from_data`` independently drops any x-cluster
# holding fewer than 3 spans, so every column of a 2-body-row table is
# filtered out and the shape can never reconstruct. A lower value here
# would be unreachable rather than permissive.
MIN_BODY_ROWS = 3
# Fraction of in-rect spans that must land in an inferred column.
# Asymmetric by construction: ``column_for_x`` assigns anything at or
# right of the first column, so this catches spans overflowing the *left*
# edge only — the right edge and the hard-shape leak are caught by
# ``MIN_ROW_FILL_RATIO`` below, which is the load-bearing precision gate.
MIN_ASSIGNED_RATIO = 0.9
# Mean cell occupancy across the reconstructed body, as a fraction of the
# column count — the round-1 precision guardrail. A flat clean table fills
# essentially every cell (measured >= 0.98 on all six rendered-clean
# fixtures); the shapes round 1 must refuse — stacked-header hierarchical
# tables (dense_text_548: 0.45) and non-tables the detector or a whole-page
# rect fed in (FUNSD forms 0.38-0.41, diverse_layout_49 0.49) — all fall
# well below. 0.75 sits in the empty gap between the two populations.
#
# This is the signal the asymmetric ``MIN_ASSIGNED_RATIO`` and the ``MIN_*``
# counts could not provide: those shapes over-segment columns and over-merge
# rows (via ``bands_to_rows``'s continuation rule) into a grid that is
# structurally large but mostly empty. Density catches exactly that.
MIN_ROW_FILL_RATIO = 0.75


def span_from_region(region: OCRRegionResult) -> Span:
    """Reduce a four-point OCR quad to its axis-aligned bounds as a ``Span``."""
    xs = [p[0] for p in region.bbox]
    ys = [p[1] for p in region.bbox]
    return Span(
        y_top=min(ys), y_bottom=max(ys),
        x_left=min(xs), x_right=max(xs), text=region.text,
    )


def regions_in_rect(
    regions: Sequence[OCRRegionResult],
    rect: tuple[float, float, float, float],
) -> list[OCRRegionResult]:
    """Select OCR regions whose bbox centroid falls inside a pixel-space rect.

    Both sides must be in the same image-pixel space — the OCR render and
    the layout render of the same page at the same dpi. Callers verify
    that (see ``_layout_blocks_and_tables``). Centroid containment, not
    overlap, so a detection straddling the rect edge belongs to whichever
    side holds its middle.
    """
    x0, y0, x1, y1 = rect
    inside: list[OCRRegionResult] = []
    for r in regions:
        if not r.text.strip():
            continue
        xs = [p[0] for p in r.bbox]
        ys = [p[1] for p in r.bbox]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            inside.append(r)
    return inside


def reconstruct_table(
    regions: Sequence[OCRRegionResult],
    table_rect: tuple[float, float, float, float],
    dpi: int,
    conf: float,
    *,
    pix_dims: tuple[int, int],
) -> TableData | None:
    """Reconstruct a table's cells from the OCR regions inside its rect.

    ``regions`` are the page's per-detection OCR results and ``table_rect``
    the layout model's table bbox, both in image-pixel coords at ``dpi``
    (the caller's coordinate-space guard has already verified they are
    comparable). ``conf`` is the layout detector's confidence for the
    table region (0-1); ``pix_dims`` the page render's pixel dimensions,
    used to normalise the element position.

    The first reconstructed row is taken as the header row — round 1 does
    not attempt to flatten stacked/spanning headers; such shapes are
    expected to refuse. Returns ``None`` below the precision gates.
    """
    scale = dpi / 72.0
    inside = regions_in_rect(regions, table_rect)
    spans = [span_from_region(r) for r in inside]
    if not spans:
        logger.debug("table reconstruction refused: no OCR regions inside the table rect")
        return None

    bands = bin_y_bands(spans, band_tolerance=ROW_BAND_PT * scale)
    if len(bands) < 1 + MIN_BODY_ROWS:
        logger.debug("table reconstruction refused: only %d y-bands", len(bands))
        return None
    header_bands, body_bands = bands[:1], bands[1:]

    # Columns come from the body spans, not the header band — headers are
    # commonly centred within wider cells and would skew the clusters.
    body_spans = [s for _by, bs in body_bands for s in bs]
    columns = columns_from_data(
        body_spans, table_rect[2],
        expected_k=0,
        cluster_gap=DATA_CLUSTER_GAP_PT * scale,
        last_column_pad=LAST_COLUMN_PAD_PT * scale,
    )
    if len(columns) < MIN_COLUMNS:
        logger.debug("table reconstruction refused: only %d columns inferred", len(columns))
        return None

    x_tol = COLUMN_X_TOLERANCE_PT * scale
    assigned = sum(
        1 for s in spans if column_for_x(s.x_left, columns, x_tolerance=x_tol) is not None
    )
    if assigned / len(spans) < MIN_ASSIGNED_RATIO:
        logger.debug(
            "table reconstruction refused: column fit %d/%d below %s",
            assigned, len(spans), MIN_ASSIGNED_RATIO,
        )
        return None

    # Header and body bin separately. ``bands_to_rows``'s continuation rule
    # folds a band with no leading-column value into the row above — right
    # for a wrapped body cell, silently wrong for a first body row whose
    # leading cell is blank (indented or grouped rows), which would be
    # absorbed into the header and lost.
    headers = bands_to_rows(header_bands, columns, x_tolerance=x_tol)[0]
    if not any(h for h in headers):
        logger.debug("table reconstruction refused: no header text recovered")
        return None
    body = drop_blank_rows(bands_to_rows(body_bands, columns, x_tolerance=x_tol))
    if len(body) < MIN_BODY_ROWS:
        logger.debug("table reconstruction refused: only %d body rows after binning", len(body))
        return None

    # Row-fill density — the precision guardrail. A sparse grid means the
    # binning over-segmented columns or over-merged rows: a hierarchical or
    # form shape, not a flat table. Refuse it (see MIN_ROW_FILL_RATIO).
    filled = sum(1 for row in body for cell in row if cell)
    fill_ratio = filled / (len(body) * len(columns))
    if fill_ratio < MIN_ROW_FILL_RATIO:
        logger.debug(
            "table reconstruction refused: row fill %.2f (%d/%d cells) below %s",
            fill_ratio, filled, len(body) * len(columns), MIN_ROW_FILL_RATIO,
        )
        return None

    # Lineage, not defaults: confidence from the constituent regions
    # (capped by the detector's), producer marker so reconstructed tables
    # are distinguishable from native-finder ones in the parquet.
    mean_conf = sum(r.confidence for r in inside) / len(inside)
    confidence = min(mean_conf, conf) if conf > 0 else mean_conf

    pw, ph = float(pix_dims[0]), float(pix_dims[1])
    position = _normalise_bbox(table_rect, pw, ph)
    return TableData(
        headers=headers,
        rows=body,
        position=position,
        confidence=confidence,
        context={"producer": "table_grid"},
    )
