"""Shared table-grid geometry: one binning/clustering algorithm, two feeders.

Lifted from ``spreadsheet_print`` (see
docs/decisions.md “Table-cell reconstruction on OCR pages”) so the OCR
table feeder (``ingest/ocr_tables.py``) and the spreadsheet-print
extractor consume the
same row/column inference rather than growing a fourth table algorithm.
Pure geometry over ``Span`` — no PDF library, no OCR types.

Coordinate space is the caller's: spreadsheet-print feeds PDF points
(``page.get_text("dict")`` coords), the OCR feeder image pixels at its
render dpi. The ``*_PT`` defaults are PDF points — pixel-space callers
must scale them by ``dpi / 72`` or the tolerances are ~2.8× too tight at
200 dpi.
"""
from __future__ import annotations

from dataclasses import dataclass

# Y-jitter tolerance when grouping spans into the same row band (points).
ROW_BAND_PT = 3.0
# Tolerance on x_left when assigning a span to a column (points).
COLUMN_X_TOLERANCE_PT = 6.0
# Gap between data x-clusters that signals a column boundary (points).
DATA_CLUSTER_GAP_PT = 12.0
# Pad added to the last column's rightmost data x (points).
LAST_COLUMN_PAD_PT = 8.0


@dataclass(slots=True)
class Span:
    """One positioned text fragment — a PDF text span or an OCR quad's bounds."""
    y_top: float
    y_bottom: float
    x_left: float
    x_right: float
    text: str

    @property
    def cx(self) -> float:
        return (self.x_left + self.x_right) / 2

    @property
    def cy(self) -> float:
        return (self.y_top + self.y_bottom) / 2

    @property
    def height(self) -> float:
        return self.y_bottom - self.y_top


@dataclass(slots=True)
class Column:
    """An inferred column (x-position cluster)."""
    x_left: float
    x_right: float


def bin_y_bands(
    spans: list[Span], *, band_tolerance: float = ROW_BAND_PT,
) -> list[tuple[float, list[Span]]]:
    """Group spans into y-bands. Bands within ``band_tolerance`` merge.

    Returns ``[(y_top_of_band, spans_in_band), ...]`` sorted top-to-bottom.
    """
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: (s.y_top, s.x_left))
    bands: list[list[Span]] = [[sorted_spans[0]]]
    for s in sorted_spans[1:]:
        if abs(s.y_top - bands[-1][0].y_top) <= band_tolerance:
            bands[-1].append(s)
        else:
            bands.append([s])
    return [(b[0].y_top, sorted(b, key=lambda s: s.x_left)) for b in bands]


def columns_from_data(
    data_spans: list[Span],
    page_width: float,
    *,
    expected_k: int,
    cluster_gap: float = DATA_CLUSTER_GAP_PT,
    last_column_pad: float = LAST_COLUMN_PAD_PT,
) -> list[Column]:
    """Derive column boundaries from data span x_lefts (not headers).

    Many spreadsheets centre header text within a wider cell. Using header
    positions misaligns left-anchored data. Clustering data x_lefts gives
    the true column starts. We then keep the top ``expected_k`` clusters
    by population (``expected_k=0`` keeps all) — this filters intra-column
    variance (right-aligned numbers, variable-width Author names) that
    creates spurious mid-column clusters.

    The last column's x_right tracks its data x_right + a small padding,
    not page_width — important when headers are centred and we use column
    midpoints to assign header text. Letting the last column extend to
    page_width pushes its midpoint far right and steals header pieces
    that should belong to it.
    """
    if not data_spans:
        return []
    # Index x_lefts back to the originating span so we can recover x_right
    # for the kept clusters when computing the last column's right edge.
    xs_with_right = sorted(((s.x_left, s.x_right) for s in data_spans), key=lambda p: p[0])
    clusters: list[list[tuple[float, float]]] = [[xs_with_right[0]]]
    for xl, xr in xs_with_right[1:]:
        if xl - clusters[-1][-1][0] > cluster_gap:
            clusters.append([(xl, xr)])
        else:
            clusters[-1].append((xl, xr))

    # Sort clusters by population descending; keep top expected_k.
    clusters.sort(key=lambda c: -len(c))
    if expected_k > 0:
        clusters = clusters[:expected_k]
    # Keep only clusters that have at least a meaningful share of the
    # densest cluster (1/4) — drops outliers when expected_k overshoots.
    if clusters:
        max_pop = len(clusters[0])
        clusters = [c for c in clusters if len(c) >= max(3, max_pop // 4)]
    # Restore left-to-right order.
    clusters.sort(key=lambda c: min(p[0] for p in c))

    columns: list[Column] = []
    for i, cluster in enumerate(clusters):
        x_left = min(p[0] for p in cluster)
        if i + 1 < len(clusters):
            x_right = min(p[0] for p in clusters[i + 1]) - 0.1
        else:
            # Last column: cap at the rightmost data x in this cluster
            # plus a small pad — keeps its midpoint near actual content.
            x_right = min(max(p[1] for p in cluster) + last_column_pad, page_width)
        columns.append(Column(x_left=x_left, x_right=x_right))
    return columns


def column_for_x(
    x: float, columns: list[Column], *, x_tolerance: float = COLUMN_X_TOLERANCE_PT,
) -> int | None:
    """Assign x to the column with the largest x_left ≤ x.

    Tolerance only applies at the very-left edge (a data span may begin
    a few pixels left of the header anchor). Adjacent columns therefore
    do not overlap — assignment to column N requires x ≥ N's x_left and
    x < N+1's x_left.
    """
    if not columns:
        return None
    if x < columns[0].x_left - x_tolerance:
        return None
    found = 0
    for i, col in enumerate(columns):
        if x >= col.x_left:
            found = i
        else:
            break
    return found


def bands_to_rows(
    bands: list[tuple[float, list[Span]]],
    columns: list[Column],
    *,
    x_tolerance: float = COLUMN_X_TOLERANCE_PT,
) -> list[list[str]]:
    """Turn body y-bands into rows aligned to the column boundaries.

    Multi-band cells (a long value that wraps to a second visual line at
    the *same* x but a slightly larger y) merge with the row above when
    that row's leftmost column was populated and the current band only
    populates non-leftmost columns. This treats the leftmost column as
    the row anchor — common in spreadsheet-print where the first column
    holds the unique row identifier.
    """
    rows: list[list[str]] = []
    for _by, bs in bands:
        row = ["" for _ in columns]
        for sp in bs:
            idx = column_for_x(sp.x_left, columns, x_tolerance=x_tolerance)
            if idx is None:
                continue
            if row[idx]:
                row[idx] += " " + sp.text
            else:
                row[idx] = sp.text

        # Continuation rule: if this band has no value in column 0 but the
        # previous row exists, merge into it.
        if rows and not row[0] and any(c for c in row[1:]):
            for i, cell in enumerate(row):
                if cell:
                    if rows[-1][i]:
                        rows[-1][i] += " " + cell
                    else:
                        rows[-1][i] = cell
        else:
            rows.append(row)
    return rows


def drop_blank_rows(rows: list[list[str]]) -> list[list[str]]:
    """Filter rows that have no non-empty cells."""
    return [r for r in rows if any(c for c in r)]


def rows_from_spans(
    spans: list[Span], *, line_tolerance: float = 0.5,
) -> tuple[list[list[Span]], float]:
    """Cluster spans into visual rows by y-centroid.

    The row threshold adapts to the content — average span height ×
    ``line_tolerance`` — unlike ``bin_y_bands``'s fixed tolerance, which
    suits OCR quads whose y-jitter tracks glyph size. Returns rows
    top-to-bottom, each sorted left-to-right by x-centroid, plus the
    average span height (0.0 when ``spans`` is empty).
    """
    if not spans:
        return [], 0.0
    avg_h = sum(s.height for s in spans) / len(spans)
    threshold = avg_h * line_tolerance
    ordered = sorted(spans, key=lambda s: (s.cy, s.cx))
    rows: list[list[Span]] = [[ordered[0]]]
    for s in ordered[1:]:
        if abs(s.cy - rows[-1][0].cy) <= threshold:
            rows[-1].append(s)
        else:
            rows.append([s])
    for row in rows:
        row.sort(key=lambda s: s.cx)
    return rows, avg_h


def cluster_x_centroids(xs: list[float], gap_threshold: float) -> list[float]:
    """1-D agglomerative clustering: items within ``gap_threshold`` group together."""
    if not xs:
        return []
    xs_sorted = sorted(xs)
    clusters: list[list[float]] = [[xs_sorted[0]]]
    for x in xs_sorted[1:]:
        if x - clusters[-1][-1] > gap_threshold:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    return [sum(c) / len(c) for c in clusters]
