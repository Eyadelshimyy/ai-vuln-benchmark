"""The pdfium text engine: segmentation on synthetic characters, and the page
methods against pages the builder draws."""

from __future__ import annotations

import pytest

from tests._pdf_builders import PdfBuilder
from womblex.ingest.pdf import _text, open_document
from womblex.ingest.pdf.types import Rect

LINE = 14.0


def _chars(text: str, x: float, y: float, *, size: float = 10, font: str = "Helv", bold: bool = False):
    out = []
    for i, ch in enumerate(text):
        left = x + i * size * 0.5
        out.append(_text.Char(ch, Rect(left, y, left + size * 0.5, y + size), size, font, bold))
    return out


class TestSegment:
    def test_consecutive_lines_form_one_block(self) -> None:
        chars = _chars("first line", 72, 100) + _chars("second line", 72, 100 + LINE)
        assert [len(b) for b in _text.segment(chars)] == [2]

    def test_a_vertical_gap_starts_a_block(self) -> None:
        chars = _chars("first", 72, 100) + _chars("second", 72, 160)
        assert [len(b) for b in _text.segment(chars)] == [1, 1]

    def test_a_pitch_over_one_and_a_half_sizes_starts_a_block(self) -> None:
        # A 6pt gap is inside the proximity band; the 1.6-size pitch breaks it.
        chars = _chars("first", 72, 100) + _chars("second", 72, 116)
        assert [len(b) for b in _text.segment(chars)] == [1, 1]

    def test_a_weight_change_alone_does_not_start_a_block(self) -> None:
        chars = _chars("Label", 72, 100, bold=True) + _chars("body text", 72, 100 + LINE)
        assert [len(b) for b in _text.segment(chars)] == [2]

    def test_a_row_across_a_gutter_stays_in_one_block(self) -> None:
        chars = _chars("left", 72, 100) + _chars("right", 340, 100)
        assert [[line.text for line in b] for b in _text.segment(chars)] == [["left", "right"]]

    def test_a_line_starting_right_of_the_previous_one_starts_a_block(self) -> None:
        # MuPDF: a start 1pt right of the previous line's start opens a block.
        chars = _chars("first line", 72, 100) + _chars("second", 76, 100 + LINE)
        assert [len(b) for b in _text.segment(chars)] == [1, 1]

    def test_a_line_starting_left_of_the_previous_one_stays(self) -> None:
        chars = _chars("first line", 72, 100) + _chars("second", 60, 100 + LINE)
        assert [len(b) for b in _text.segment(chars)] == [2]

    def test_oblique_lines_never_continue_by_pitch(self) -> None:
        def slanted(y: float) -> list[_text.Char]:
            return [
                _text.Char(c, Rect(72 + i * 4, y, 76 + i * 4, y + 10), 10, "Helv", False, (0.94, -0.34), (72 + i * 4, y + 10))
                for i, c in enumerate("slanted")
            ]

        assert [len(b) for b in _text.segment(slanted(100) + slanted(100 + LINE))] == [1, 1]

    def test_a_fragment_out_of_stream_order_rejoins_its_line(self) -> None:
        chars = _chars("alpha", 72, 100) + _chars("later", 72, 300) + _chars("beta", 112, 100)
        lines = [line.text for block in _text.segment(chars) for line in block]
        assert lines == ["alpha beta", "later"]

    def test_a_column_gutter_is_not_bridged(self) -> None:
        chars = _chars("left", 72, 100) + _chars("right", 340, 100)
        assert [line.text for block in _text.segment(chars) for line in block] == ["left", "right"]

    def test_rotated_text_forms_lines_along_its_direction(self) -> None:
        # Bottom to top, as a side tab reads; each character sits above the last.
        up = [
            _text.Char(ch, Rect(400, 500 - (i + 1) * 5, 410, 500 - i * 5), 10, "Helv", False, (0.0, -1.0))
            for i, ch in enumerate("Part 2")
        ]
        assert [line.text for block in _text.segment(up) for line in block] == ["Part 2"]

    def test_text_objects_are_read_only_for_vertical_text(self) -> None:
        def unread() -> list[tuple[Rect, str]]:
            raise AssertionError("text objects read for a page without vertical text")

        _text._restore_content_order(_chars("across", 72, 100), unread)

    def test_upward_text_continues_only_on_the_side_mupdf_favours(self) -> None:
        def column(x: float, text: str) -> list[_text.Char]:
            return [
                _text.Char(
                    ch, Rect(x, 500 - (i + 1) * 5, x + 10, 500 - i * 5), 10, "Helv", False,
                    (0.0, -1.0), (x + 8, 500 - i * 5),
                )
                for i, ch in enumerate(text)
            ]

        # Across the baseline of upward text is +x: the next column to the right splits,
        # the one 12pt to the left (the "way back") continues.
        assert [len(b) for b in _text.segment(column(100, "first") + column(112, "next"))] == [1, 1]
        assert [len(b) for b in _text.segment(column(100, "first") + column(88, "next"))] == [2]

    def test_spans_split_on_font_and_bold(self) -> None:
        chars = _chars("plain ", 72, 100) + _chars("heavy", 102, 100, font="Helv-Bold", bold=True)
        (block,) = _text.text_dict(chars)
        assert [(s.text, s.bold) for s in block.lines[0].spans] == [("plain ", False), ("heavy", True)]


class TestHyphens:
    def test_a_line_end_hyphen_and_its_break_are_kept(self) -> None:
        chars = _chars("a hyphen-", 72, 100) + _chars("ated word", 72, 100 + LINE)
        assert _text.plain_text(chars) == "a hyphen-\nated word\n"


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory: pytest.TempPathFactory):
    builder = PdfBuilder(tmp_path_factory.mktemp("text") / "text.pdf").page(400, 600)
    builder.text(72, 100, "Heading", size=16, bold=True)
    builder.text(72, 130, "A body line of ordinary text").text(72, 144, "that carries on below it.")
    builder.text(72, 400, "Far below")
    return builder.save()


class TestPage:
    """Expected values were measured on PyMuPDF 1.27.2.2 before it was removed."""

    def test_plain_text_matches_mupdf(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            assert doc[0].plain_text() == (
                "Heading\nA body line of ordinary text\nthat carries on below it.\nFar below\n"
            )

    def test_words_match_mupdf_text_and_order(self, pdf_path) -> None:
        # (text, block, line, word, x0); y1 sits within 3pt of MuPDF's.
        want = [
            ("Heading", 0, 0, 0, 72.0), ("A", 1, 0, 0, 72.0), ("body", 1, 0, 1, 82.4),
            ("line", 1, 0, 2, 109.3), ("of", 1, 0, 3, 129.5), ("ordinary", 1, 0, 4, 141.7),
            ("text", 1, 0, 5, 184.5), ("that", 1, 1, 0, 72.0), ("carries", 1, 1, 1, 93.4),
            ("on", 1, 1, 2, 129.5), ("below", 1, 1, 3, 144.8), ("it.", 1, 1, 4, 176.5),
            ("Far", 2, 0, 0, 72.0), ("below", 2, 0, 1, 91.6),
        ]
        y1 = [104.9] + [133.3] * 6 + [147.3] * 5 + [403.3] * 2
        with open_document(pdf_path) as doc:
            got = doc[0].words()
        assert [(w.text, w.block_no, w.line_no, w.word_no) for w in got] == [t[:4] for t in want]
        for w, (*_, x0), top in zip(got, want, y1, strict=True):
            assert w.x0 == pytest.approx(x0, abs=1) and w.y1 == pytest.approx(top, abs=3)

    def test_blocks_group_like_mupdf(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            assert [b.text for b in doc[0].text_blocks()] == [
                "Heading\n", "A body line of ordinary text\nthat carries on below it.\n", "Far below\n",
            ]

    def test_dict_reports_size_font_and_bold(self, pdf_path) -> None:
        with open_document(pdf_path) as doc:
            spans = [s for b in doc[0].text_dict() for line in b.lines for s in line.spans]
        heading = spans[0]
        assert (heading.text, heading.size, heading.bold) == ("Heading", 16.0, True)
        assert heading.font == "Helvetica-Bold"
        assert not spans[1].bold and spans[1].size == 11.0

    def test_a_line_end_hyphen_matches_mupdf(self, tmp_path) -> None:
        builder = PdfBuilder(tmp_path / "h.pdf").page()
        builder.text(72, 100, "the Auditor-").text(72, 113, "General reported")
        with open_document(builder.save()) as doc:
            assert doc[0].plain_text() == "the Auditor-\nGeneral reported\n"

    def test_text_off_the_page_is_clipped_like_mupdf(self, tmp_path) -> None:
        builder = PdfBuilder(tmp_path / "off.pdf").page(400, 400)
        builder.text(50, -20, "Above the page").text(50, 100, "On the page").text(350, 200, "Straddles the edge")
        with open_document(builder.save()) as doc:
            got = doc[0].plain_text()
            assert got.startswith("On the page\nStraddles") and "Above" not in got and "edge" not in got

    def test_size_scaled_by_the_text_matrix(self, tmp_path) -> None:
        # Producers commonly set `1 Tf` and scale through `Tm`; pdfium's own size
        # is then 1, and every size-relative threshold collapses with it.
        # Words are placed apart with no space glyph between them, as kerned
        # output often is.
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.pdfgen.canvas import Canvas

        path = tmp_path / "tm.pdf"
        canvas = Canvas(str(path), pagesize=(400, 400))
        text = canvas.beginText()
        text.setTextTransform(12, 0, 0, 12, 50, 300)
        text.setFont("Helvetica", 1)
        for word in ("Kerned", "scaled", "words"):
            text.textOut(word)
            text.moveCursor(stringWidth(word, "Helvetica", 1) + 0.4, 0)
        canvas.drawText(text)
        canvas.save()
        with open_document(path) as doc:
            assert doc[0].plain_text() == "Kerned scaled words\n"
            (span, *_) = [s for b in doc[0].text_dict() for line in b.lines for s in line.spans]
            assert span.size == pytest.approx(12.0)

    def test_empty_page(self, tmp_path) -> None:
        with PdfBuilder(tmp_path / "e.pdf").page().open() as doc:
            assert doc[0].plain_text() == "" and doc[0].words() == [] and doc[0].text_dict() == []
