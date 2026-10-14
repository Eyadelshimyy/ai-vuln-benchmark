"""The pdfium table finder: pdfplumber's `TableFinder` fed from pdfium.

Pages are drawn at known coordinates so the expected grid is the one drawn.
"""

from __future__ import annotations

import pytest
from reportlab.pdfgen.canvas import Canvas

from womblex.ingest.pdf import _tables, open_document
from womblex.ingest.pdf.types import Rect

_HEIGHT = 842


def _grid(path, *, one_path: bool) -> None:
    """A 3-row, 2-column ruled grid with a label in each cell. Drawn either as
    separate lines or as one path, which a bounding box would collapse."""
    canvas = Canvas(str(path), pagesize=(595, _HEIGHT))
    rows = [_HEIGHT - y for y in (200, 220, 240, 260)]
    cols = (72, 172, 272)
    if one_path:
        grid = canvas.beginPath()
        for y in rows:
            grid.moveTo(cols[0], y)
            grid.lineTo(cols[-1], y)
        for x in cols:
            grid.moveTo(x, rows[0])
            grid.lineTo(x, rows[-1])
        canvas.drawPath(grid, stroke=1, fill=0)
    else:
        for y in rows:
            canvas.line(cols[0], y, cols[-1], y)
        for x in cols:
            canvas.line(x, rows[0], x, rows[-1])
    canvas.setFont("Helvetica", 9)
    for row in range(3):
        for col in range(2):
            canvas.drawString(80 + col * 100, rows[row] - 14, f"r{row}c{col}")
    canvas.save()


def _aligned(path) -> None:
    """Unruled columns: four rows of three left-aligned fields."""
    canvas = Canvas(str(path), pagesize=(595, _HEIGHT))
    canvas.setFont("Helvetica", 10)
    for row in range(4):
        for col in range(3):
            canvas.drawString(72 + col * 150, _HEIGHT - (200 + row * 18), f"item{row}{col}")
    canvas.save()


def _prose(path) -> None:
    canvas = Canvas(str(path), pagesize=(595, _HEIGHT))
    canvas.setFont("Helvetica", 11)
    for i in range(6):
        canvas.drawString(72, _HEIGHT - (100 + i * 14), "The quick brown wombat reads the compliance notice.")
    canvas.save()


def _find(path, strategy):
    with open_document(path) as doc:
        return doc[0].find_tables(strategy=strategy)


@pytest.mark.parametrize("one_path", [False, True], ids=["lines", "single-path"])
def test_ruled_grid(tmp_path, one_path) -> None:
    path = tmp_path / "grid.pdf"
    _grid(path, one_path=one_path)
    (table,) = _find(path, "lines")
    assert (table.row_count, table.col_count) == (3, 2)
    assert table.rows == (("r0c0", "r0c1"), ("r1c0", "r1c1"), ("r2c0", "r2c1"))
    assert [round(v) for v in table.bbox.as_tuple()] == [72, 200, 272, 260]


def test_text_strategy_finds_unruled_columns(tmp_path) -> None:
    path = tmp_path / "aligned.pdf"
    _aligned(path)
    assert _find(path, "lines") == []
    (table,) = _find(path, "text")
    assert table.col_count == 3 and table.row_count >= 3
    assert table.rows[0] == ("item00", "item01", "item02")


def test_prose_is_not_a_ruled_table(tmp_path) -> None:
    path = tmp_path / "prose.pdf"
    _prose(path)
    assert _find(path, "lines") == []


class TestEdges:
    def test_axis_steps_become_edges_and_diagonals_do_not(self) -> None:
        edges = _tables.edges_from_polylines([[(0.0, 5.0), (10.0, 5.0), (10.0, 20.0), (0.0, 0.0)]])
        assert [(e["orientation"], e["x0"], e["x1"], e["top"], e["bottom"]) for e in edges] == [
            ("h", 0.0, 10.0, 5.0, 5.0),
            ("v", 10.0, 10.0, 5.0, 20.0),
        ]

    def test_float_noise_is_still_axis_aligned(self) -> None:
        (edge,) = _tables.edges_from_polylines([[(0.0, 5.0), (10.0, 5.0004)]])
        assert edge["orientation"] == "h" and edge["height"] == 0.0

    def test_edges_are_only_built_when_the_strategy_reads_them(self) -> None:
        def boom():
            raise AssertionError("edges read by the text strategy")

        assert _tables.find_tables("text", Rect(0, 0, 100, 100), list, boom) == []


def test_close_after_a_curve_draws_no_phantom_rule(tmp_path) -> None:
    """A curve from A down to C, a rule to D, then close: the only straight
    runs are C to D and D back to A, never the curve's chord A to C."""
    path = tmp_path / "curve.pdf"
    canvas = Canvas(str(path), pagesize=(595, _HEIGHT))
    shape = canvas.beginPath()
    shape.moveTo(72, _HEIGHT - 100)
    shape.curveTo(172, _HEIGHT - 120, 172, _HEIGHT - 180, 72, _HEIGHT - 200)
    shape.lineTo(272, _HEIGHT - 200)
    shape.close()
    canvas.drawPath(shape, stroke=1, fill=0)
    canvas.save()
    with open_document(path) as doc:
        page = doc[0]
        runs = list(_tables.read_polylines(page._path_objects(), page._origin))
    edges = _tables.edges_from_polylines(runs)
    assert [(e["orientation"], round(e["x0"]), round(e["x1"]), round(e["top"])) for e in edges] == [
        ("h", 72, 272, 200),
    ]


def test_rotated_page_finds_tables_in_the_displayed_frame() -> None:
    """The FOI index is a landscape table on a rotated portrait page: its text
    runs vertically in the unrotated frame, which once gave 39 columns for 11."""
    from tests._synthetic import FOI_INDEX_PDF

    with open_document(FOI_INDEX_PDF) as doc:
        (table,) = doc[0].find_tables(strategy="text")
    assert (table.row_count, table.col_count) == (78, 11)
    assert table.rows[0][:3] == ("FOI referen", "ce", "FOI-2025-042")


def _header_merged_grid(builder) -> None:
    """Two rows by three columns; the header's first two cells are one merged cell."""
    ys, xs = (200, 220, 240), (72, 172, 272, 372)
    for y in ys:
        builder.line(xs[0], y, xs[-1], y)
    for x in xs:
        builder.line(x, 220 if x == 172 else 200, x, 240)
    builder.text(80, 214, "head").text(280, 214, "r0c2")
    for col in range(3):
        builder.text(80 + col * 100, 234, f"r1c{col}")


def _rounded(table) -> list[list[tuple[int, ...] | None]]:
    return [[None if b is None else tuple(round(v) for v in b.as_tuple()) for b in row] for row in table.cell_boxes]


def test_merged_cell_box_spans_and_merged_away_cell_has_none(tmp_path) -> None:
    from tests._pdf_builders import PdfBuilder

    builder = PdfBuilder(tmp_path / "merged.pdf").page()
    _header_merged_grid(builder)
    with builder.open() as doc:
        (table,) = doc[0].find_tables(strategy="lines")
    assert table.rows[0] == ("head", None, "r0c2")
    assert _rounded(table) == [
        [(72, 200, 272, 220), None, (272, 200, 372, 220)],
        [(72, 220, 172, 240), (172, 220, 272, 240), (272, 220, 372, 240)],
    ]


def test_rotated_page_cell_boxes_are_in_the_displayed_frame(tmp_path) -> None:
    """On a /Rotate 90 page the grid turns with the text: cell boxes share the
    displayed frame of the table bbox, not the unrotated drawing frame."""
    from tests._pdf_builders import PdfBuilder

    builder = PdfBuilder(tmp_path / "rotated.pdf").page(rotation=90)
    _header_merged_grid(builder)
    with builder.open() as doc:
        (table,) = doc[0].find_tables(strategy="lines")
    assert table.rows == (("r1c0", "head"), ("r1c1", None), ("r1c2", "r0c2"))
    assert _rounded(table) == [
        [(602, 72, 622, 172), (622, 72, 642, 272)],
        [(602, 172, 622, 272), None],
        [(602, 272, 622, 372), (622, 272, 642, 372)],
    ]
    t = table.bbox
    for row in table.cell_boxes:
        for b in row:
            assert b is None or (t.x0 <= b.x0 and t.y0 <= b.y0 and b.x1 <= t.x1 and b.y1 <= t.y1)
