"""The PDF seam's vocabulary: geometry, tuple shape, and no backend import."""

from __future__ import annotations

import subprocess
import sys

from womblex.ingest.pdf.types import (
    Block,
    Drawing,
    FoundTable,
    Rect,
    Word,
    render_box,
)


class TestRect:
    def test_width_and_height(self) -> None:
        r = Rect(10.0, 20.0, 40.0, 60.0)
        assert (r.width, r.height) == (30.0, 40.0)
        assert r.as_tuple() == (10.0, 20.0, 40.0, 60.0)

    def test_of_normalises_inverted_corners(self) -> None:
        assert Rect.of((40.0, 60.0, 10.0, 20.0)) == Rect(10.0, 20.0, 40.0, 60.0)

    def test_transform_applies_the_matrix(self) -> None:
        # Translate by (5, 7): the identity matrix with e/f set.
        assert Rect(0.0, 0.0, 2.0, 4.0).transform((1, 0, 0, 1, 5, 7)) == Rect(5.0, 7.0, 7.0, 11.0)

    def test_transform_renormalises_after_a_rotation(self) -> None:
        # A 90-degree rotation maps (x, y) to (-y, x), inverting one axis. The
        # result must come back normalised, not with x0 > x1.
        out = Rect(0.0, 0.0, 10.0, 4.0).transform((0, 1, -1, 0, 0, 0))
        assert out == Rect(-4.0, 0.0, 0.0, 10.0)
        assert out.x0 <= out.x1 and out.y0 <= out.y1

    def test_is_frozen(self) -> None:
        r = Rect(0.0, 0.0, 1.0, 1.0)
        try:
            r.x0 = 5.0  # type: ignore[misc]
        except Exception as e:
            assert "frozen" in repr(e).lower()
        else:
            raise AssertionError("Rect should be immutable")


class TestWord:
    def test_is_shaped_like_mupdfs_eight_tuple(self) -> None:
        w = Word(1.0, 2.0, 3.0, 4.0, "hello", 0, 1, 2)
        assert len(w) == 8
        # grid_projection slices positionally; both must keep working.
        assert w[:4] == (1.0, 2.0, 3.0, 4.0)
        assert w[4] == "hello"
        assert (w.block_no, w.line_no, w.word_no) == (0, 1, 2)


class TestDrawing:
    def test_filled_covers_fill_and_fill_stroke_only(self) -> None:
        rect = Rect(0.0, 0.0, 1.0, 1.0)
        assert Drawing(kind="fill", rect=rect).filled
        assert Drawing(kind="fill_stroke", rect=rect).filled
        assert not Drawing(kind="stroke", rect=rect).filled


class TestRenderBox:
    def test_an_edge_within_a_thousandth_of_a_pixel_rounds_to_it(self) -> None:
        """MuPDF's values: 595.2pt at 150dpi is 1240px, not 1241."""
        assert render_box(Rect(0.0, 0.0, 595.2, 100.0004), 150)[2:] == (1240, 209)
        assert render_box(Rect(0.0, 0.0, 612.0001, 300.0), 72)[2:] == (612, 300)

    def test_a_clip_rounds_outward_within_the_same_tolerance(self) -> None:
        clip = Rect(10.0004, 9.9996, 20.0004, 30.5)
        assert render_box(Rect(0.0, 0.0, 100.0, 100.0), 72, clip) == (10, 10, 20, 31)


class TestDefaults:
    def test_a_block_is_text_and_carries_neither_payload(self) -> None:
        b = Block(bbox=Rect(0.0, 0.0, 1.0, 1.0), number=0)
        assert (b.kind, b.text, b.lines) == ("text", "", ())

    def test_a_found_table_keeps_the_finders_own_counts(self) -> None:
        t = FoundTable(bbox=Rect(0.0, 0.0, 1.0, 1.0), row_count=3, col_count=2)
        assert (t.row_count, t.col_count, t.rows) == (3, 2, ())


def test_the_seam_imports_no_pdf_backend() -> None:
    """The point of the package: vocabulary without a library behind it."""
    code = (
        "import sys, womblex.ingest.pdf, womblex.ingest.pdf.types as t; "
        "sys.exit(int('pypdfium2' in sys.modules or t.Rect is None))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
