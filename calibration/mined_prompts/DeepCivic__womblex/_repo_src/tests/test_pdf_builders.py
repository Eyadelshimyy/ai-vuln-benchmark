"""The reportlab test builders: what is drawn reads back through the seam."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tests._pdf_builders import PdfBuilder
from womblex.ingest.pdf.types import Rect


def test_pages_keep_their_size_and_a_trailing_blank_page(tmp_path: Path) -> None:
    b = PdfBuilder(tmp_path / "p.pdf").page(400, 600).text(50, 50, "first").page()
    with b.open() as doc:
        assert len(doc) == 2
        assert doc[0].rect == Rect(0, 0, 400, 600)
        assert doc[1].rect == Rect(0, 0, 595, 842)


def test_an_unpaged_builder_saves_one_blank_a4_page(tmp_path: Path) -> None:
    with PdfBuilder(tmp_path / "blank.pdf").open() as doc:
        assert len(doc) == 1
        assert doc[0].plain_text(dehyphenate=False).strip() == ""


def test_text_baseline_is_measured_from_the_top(tmp_path: Path) -> None:
    with PdfBuilder(tmp_path / "t.pdf").page().text(72, 100, "alpha", size=11).open() as doc:
        [word] = doc[0].words()
    assert word.text == "alpha"
    assert word.x0 == pytest.approx(72, abs=0.5)
    # The baseline sits inside the word box, below its top.
    assert word.y0 < 100 < word.y1


def test_rect_reads_back_as_one_filled_drawing(tmp_path: Path) -> None:
    b = PdfBuilder(tmp_path / "r.pdf").page().rect(Rect(100, 200, 300, 220), (0.8, 0.8, 0.8))
    with b.open() as doc:
        [drawing] = doc[0].drawings()
    assert drawing.filled
    assert drawing.rect == Rect(100, 200, 300, 220)
    assert drawing.fill == pytest.approx((0.8, 0.8, 0.8), abs=0.01)


def test_image_reads_back_as_one_draw_inside_its_rect(tmp_path: Path) -> None:
    img = Image.new("RGB", (40, 20), (255, 0, 0))
    b = PdfBuilder(tmp_path / "i.pdf").page(200, 200).image(Rect(0, 0, 200, 200), img)
    with b.open() as doc:
        [placed] = doc[0].images()
    # 2:1 kept and centred in the square: full width, half height.
    assert placed.rect == Rect(0, 50, 200, 150)


def test_output_is_byte_reproducible(tmp_path: Path) -> None:
    def build(name: str) -> bytes:
        return PdfBuilder(tmp_path / name).page().text(72, 72, "same").save().read_bytes()

    assert build("a.pdf") == build("b.pdf")


def test_drawing_before_a_page_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="page"):
        PdfBuilder(tmp_path / "x.pdf").text(0, 0, "nowhere")
