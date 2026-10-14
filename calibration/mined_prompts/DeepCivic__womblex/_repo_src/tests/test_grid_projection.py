"""Tests for womblex.ingest.grid_projection — column detection and spatial rendering.

These are synthetic correctness tests for the algorithm itself.  Real-fixture
accuracy validation (CER / CER-s / WER deltas) lives in the accuracy benchmark
suite and is run separately.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from tests._pdf_builders import PdfBuilder
from womblex.ingest.grid_projection import (
    ColumnRegion,
    extract_page_text,
    project_to_columns,
    render_spatial_text,
)
from womblex.ingest.pdf.types import Document

# ---------------------------------------------------------------------------
# project_to_columns — algorithm-level tests using fake word tuples
# ---------------------------------------------------------------------------


def _word(x0: float, y0: float, x1: float, y1: float, text: str) -> tuple:
    """Build a word tuple."""
    return (x0, y0, x1, y1, text, 0, 0, 0)


def _grid_words(
    column_xs: list[float],
    column_width: float,
    rows: int,
    row_height: float = 14.0,
) -> list[tuple]:
    """Generate word tuples laid out in N columns × M rows.

    Each row of each column is filled with multiple words back-to-back so
    the projection histogram sees the column as continuously occupied
    (no spurious internal gutters).
    """
    words: list[tuple] = []
    word_h = 12.0
    word_pixel_width = 30.0
    word_gap = 4.0  # below min_gutter_px so adjacent words don't split a column
    word_step = word_pixel_width + word_gap

    for col_idx, col_x in enumerate(column_xs):
        words_per_row = max(1, int(column_width // word_step))
        for r in range(rows):
            y0 = 50.0 + r * row_height
            for w_idx in range(words_per_row):
                x0 = col_x + w_idx * word_step
                x1 = x0 + word_pixel_width
                words.append(_word(x0, y0, x1, y0 + word_h, f"c{col_idx}r{r}w{w_idx}"))
    return words


class TestProjectToColumns:
    def test_empty_input_returns_empty_list(self) -> None:
        assert project_to_columns([], 612.0) == []

    def test_zero_page_width_returns_empty_list(self) -> None:
        assert project_to_columns([_word(0, 0, 50, 12, "x")], 0.0) == []

    def test_sparse_page_returns_single_column(self) -> None:
        # Below min_words threshold even though there's a clear gap
        words = [
            _word(50, 100, 100, 112, "left"),
            _word(400, 100, 450, 112, "right"),
        ]
        cols = project_to_columns(words, 612.0)
        assert len(cols) == 1
        assert len(cols[0].words) == 2

    def test_single_column_dense_text(self) -> None:
        # Words placed back-to-back across the page (≤4 px gap < 12 px min_gutter)
        words = []
        for r in range(12):
            for x_start in (50, 130, 210, 290, 370, 450, 530):
                words.append(_word(x_start, 50 + r * 14, x_start + 75, 62 + r * 14, "w"))
        cols = project_to_columns(words, 612.0)
        assert len(cols) == 1

    def test_two_column_layout_detected(self) -> None:
        words = _grid_words(
            column_xs=[50.0, 350.0],
            column_width=240.0,
            rows=8,
        )
        cols = project_to_columns(words, 612.0)
        assert len(cols) == 2
        # Left column starts within left half; right column starts within right half
        assert cols[0].x0 < 306
        assert cols[1].x0 >= 306

    def test_three_column_layout_detected(self) -> None:
        words = _grid_words(
            column_xs=[50.0, 240.0, 430.0],
            column_width=140.0,
            rows=6,
        )
        cols = project_to_columns(words, 612.0)
        assert len(cols) == 3

    def test_words_distributed_correctly_across_columns(self) -> None:
        words = _grid_words(
            column_xs=[50.0, 350.0],
            column_width=240.0,
            rows=5,
        )
        cols = project_to_columns(words, 612.0)
        assert len(cols) == 2
        # Each column should contain exactly half the total words
        assert len(cols[0].words) == len(words) // 2
        assert len(cols[1].words) == len(words) // 2
        # All words in left column have midpoint < midpage; right column ≥ midpage
        assert all((w[0] + w[2]) / 2 < 306 for w in cols[0].words)
        assert all((w[0] + w[2]) / 2 >= 306 for w in cols[1].words)

    def test_column_words_sorted_top_to_bottom(self) -> None:
        words = _grid_words(
            column_xs=[50.0, 350.0],
            column_width=240.0,
            rows=5,
        )
        cols = project_to_columns(words, 612.0)
        for col in cols:
            ys = [w[1] for w in col.words]
            assert ys == sorted(ys)


# ---------------------------------------------------------------------------
# render_spatial_text
# ---------------------------------------------------------------------------


class TestRenderSpatialText:
    def test_empty_columns(self) -> None:
        assert render_spatial_text([]) == ""

    def test_single_column_renders_in_order(self) -> None:
        words = [
            _word(50, 100, 90, 112, "alpha"),
            _word(50, 120, 90, 132, "beta"),
            _word(50, 140, 90, 152, "gamma"),
        ]
        col = ColumnRegion(x0=50, x1=120, words=words)
        rendered = render_spatial_text([col])
        # Lines should appear in document order
        idx_a = rendered.index("alpha")
        idx_b = rendered.index("beta")
        idx_c = rendered.index("gamma")
        assert idx_a < idx_b < idx_c

    def test_columns_joined_left_to_right(self) -> None:
        left = ColumnRegion(x0=50, x1=300, words=[_word(50, 100, 90, 112, "LEFT")])
        right = ColumnRegion(x0=350, x1=600, words=[_word(350, 100, 410, 112, "RIGHT")])
        rendered = render_spatial_text([left, right])
        assert "LEFT" in rendered
        assert "RIGHT" in rendered
        assert rendered.index("LEFT") < rendered.index("RIGHT")
        # Columns separated by blank line
        assert "\n\n" in rendered


# ---------------------------------------------------------------------------
# extract_page_text — integration with programmatic PDFs
# ---------------------------------------------------------------------------


@pytest.fixture
def letter_doc(tmp_path):
    """Build a US Letter one-page document from ``(x, y, text)`` lines."""
    opened: list[Document] = []

    def build(lines: Sequence[tuple[float, float, str]] = ()) -> Document:
        builder = PdfBuilder(tmp_path / "letter.pdf").page(612, 792)
        for x, y, text in lines:
            builder.text(x, y, text)
        opened.append(builder.open())
        return opened[-1]

    yield build
    for doc in opened:
        doc.close()


def _columns(left: str, right: str, rows: int, step: int) -> list[tuple[float, float, str]]:
    return [
        line
        for i in range(rows)
        for line in ((72, 100 + i * step, left.format(i)), (340, 100 + i * step, right.format(i)))
    ]


class TestExtractPageText:
    def test_empty_page_returns_empty_string(self, letter_doc) -> None:
        assert extract_page_text(letter_doc()[0]) == ""

    def test_single_column_page_falls_back_to_get_text(self, letter_doc) -> None:
        doc = letter_doc([(72, 100 + i * 16, f"single column line number {i}") for i in range(20)])
        output = extract_page_text(doc[0])
        for i in range(20):
            assert f"line number {i}" in output
        lines = [f"line number {i}" for i in range(20)]
        positions = [output.index(line) for line in lines]
        assert positions == sorted(positions)

    def test_two_column_page_uses_grid_projection(self, letter_doc) -> None:
        doc = letter_doc(_columns("LEFT{:02d}", "RIGHT{:02d}", 10, 18))
        output = extract_page_text(doc[0])

        for i in range(10):
            assert f"LEFT{i:02d}" in output
            assert f"RIGHT{i:02d}" in output

        last_left = max(output.index(f"LEFT{i:02d}") for i in range(10))
        first_right = min(output.index(f"RIGHT{i:02d}") for i in range(10))
        assert last_left < first_right, (
            "Multi-column reading order incorrect — left column should be "
            "fully consumed before right column starts"
        )

    def test_two_column_page_preserves_within_column_order(self, letter_doc) -> None:
        doc = letter_doc(_columns("L{}", "R{}", 8, 18))
        output = extract_page_text(doc[0])
        left_positions = [output.index(f"L{i}") for i in range(8)]
        right_positions = [output.index(f"R{i}") for i in range(8)]
        assert left_positions == sorted(left_positions)
        assert right_positions == sorted(right_positions)


# ---------------------------------------------------------------------------
# Regression: native extractor wiring still works end-to-end
# ---------------------------------------------------------------------------


class TestNativeExtractorIntegration:
    """Confirm grid_projection wiring through the orchestrator's native path."""

    def _extract_native(self, doc):
        from womblex.ingest.detect import DocumentProfile, DocumentType
        from womblex.ingest.orchestrator import extract_pdf_with_plan

        profile = DocumentProfile(
            doc_type=DocumentType.NATIVE_NARRATIVE,
            page_count=len(doc),
            has_text_layer=True,
            text_coverage=1.0,
            has_images=False,
            has_tables=False,
            has_handwriting_signals=False,
            ocr_confidence=None,
            glyph_regularity=None,
            stroke_consistency=None,
            confidence=0.9,
        )
        return extract_pdf_with_plan(doc, profile)

    def test_native_extractor_handles_two_column_page(self, letter_doc) -> None:
        doc = letter_doc(_columns("COL_A_line_{}", "COL_B_line_{}", 8, 18))
        result = self._extract_native(doc)

        assert len(result.pages) == 1
        text = result.pages[0].text
        for i in range(8):
            assert f"COL_A_line_{i}" in text
            assert f"COL_B_line_{i}" in text
        last_a = max(text.index(f"COL_A_line_{i}") for i in range(8))
        first_b = min(text.index(f"COL_B_line_{i}") for i in range(8))
        assert last_a < first_b

    def test_native_extractor_unchanged_on_single_column(self, letter_doc) -> None:
        doc = letter_doc([(72, 100 + i * 16, f"single body line {i}") for i in range(10)])
        result = self._extract_native(doc)
        text = result.pages[0].text
        for i in range(10):
            assert f"single body line {i}" in text
