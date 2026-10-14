"""The pdfium backend behind the seam: PDFium's bottom-left space in, top-left out.

Expected values are the coordinates the builder drew at.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from tests._pdf_builders import PdfBuilder
from womblex.ingest.pdf import open_document
from womblex.ingest.pdf.types import Rect, render_box


def _open(path):
    return open_document(path)


def _box(rect: Rect) -> tuple[float, ...]:
    return tuple(round(v, 2) for v in rect.as_tuple())


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory: pytest.TempPathFactory):
    """Page 0: a black rect, a line, one image drawn twice, a text field.
    Page 1: the same rect on a page rotated 90 degrees."""
    path = tmp_path_factory.mktemp("pdf") / "pdfium.pdf"
    red = Image.new("RGB", (20, 10), (255, 0, 0))
    builder = PdfBuilder(path).page(400, 600)
    builder.rect(Rect(100, 200, 200, 240)).line(50, 400, 350, 400, width=2)
    builder.image(Rect(50, 450, 150, 500), red).image(Rect(200, 450, 300, 500), red)
    builder.text_field("provider_name", "Papilio Barton", Rect(72, 120, 272, 140))
    builder.page(400, 600, rotation=90).rect(Rect(100, 200, 200, 240))
    return builder.save()


class TestDocument:
    def test_pages_and_select(self, pdf_path) -> None:
        with _open(pdf_path) as doc:
            assert doc.page_count == len(doc) == 2
            assert doc.name.endswith("pdfium.pdf")
            assert [page.number for page in doc] == [0, 1]
            doc.select([1])
            assert doc.page_count == 1 and doc[0].rotation == 90


class TestGeometry:
    def test_rotated_page_rect_and_matrix(self, pdf_path) -> None:
        with _open(pdf_path) as doc:
            upright, rotated = doc[0], doc[1]
            assert _box(upright.rect) == (0, 0, 400, 600)
            assert upright.rotation_matrix == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            assert _box(rotated.rect) == (0, 0, 600, 400)
            assert rotated.rotation_matrix == (0.0, 1.0, -1.0, 0.0, 600, 0.0)

    def test_drawings_are_top_left_without_stroke_width(self, pdf_path) -> None:
        with _open(pdf_path) as doc:
            drawings = {(d.kind, _box(d.rect)): d.fill for d in doc[0].drawings()}
            rotated = [_box(d.rect) for d in doc[1].drawings()]
        assert drawings[("fill_stroke", (100, 200, 200, 240))] == (0.0, 0.0, 0.0)
        assert drawings[("stroke", (50, 400, 350, 400))] is None
        assert rotated == [(100, 200, 200, 240)], "objects stay in unrotated space"

    def test_one_page_image_per_draw(self, pdf_path) -> None:
        with _open(pdf_path) as doc:
            rects = sorted(_box(image.rect) for image in doc[0].images())
        # Kept to aspect (2:1) inside a 100x50 box: fills it exactly.
        assert rects == [(50, 450, 150, 500), (200, 450, 300, 500)]

    def test_nested_forms_compose_to_page_space(self, tmp_path) -> None:
        path = tmp_path / "forms.pdf"
        canvas = Canvas(str(path), pagesize=(400, 600), invariant=1)
        canvas.beginForm("inner")
        canvas.rect(10, 10, 50, 20, stroke=0, fill=1)
        canvas.endForm()
        canvas.beginForm("outer")
        canvas.translate(100, 300)
        canvas.doForm("inner")
        canvas.endForm()
        canvas.translate(5, 7)
        canvas.doForm("outer")
        canvas.save()
        with _open(path) as doc:
            [drawing] = doc[0].drawings()
        # User space (115, 317)-(165, 337), flipped against the 600pt page.
        assert _box(drawing.rect) == (115, 263, 165, 283)

    def test_widgets(self, pdf_path) -> None:
        with _open(pdf_path) as doc:
            fields = doc[0].widgets()
            assert doc[1].widgets() == []
        assert [(f.field_name, f.field_value, _box(f.rect)) for f in fields] == [
            ("provider_name", "Papilio Barton", (72, 120, 272, 140)),
        ]


def _annotated_pdf(path, flags: int = 4) -> None:
    """One page whose only black box is a Square annotation's appearance stream,
    the shape of a fake redaction that never touches the page content."""
    appearance = b"0 g 0 0 152 20 re f"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 600] /Annots [4 0 R] >>",
        b"<< /Type /Annot /Subtype /Square /Rect [48 500 200 520] /F %d /AP << /N 5 0 R >> >>" % flags,
        b"<< /Type /XObject /Subtype /Form /BBox [0 0 152 20] /Length %d >>\nstream\n%s\nendstream"
        % (len(appearance), appearance),
    ]
    body = b"%PDF-1.7\n"
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += b"%d 0 obj\n%s\nendobj\n" % (number, obj)
    xref = len(body)
    body += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    body += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    body += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(body)


class TestAnnotations:
    def test_an_annotation_appearance_is_a_drawing(self, tmp_path) -> None:
        path = tmp_path / "annotated.pdf"
        _annotated_pdf(path)
        with _open(path) as doc:
            [drawing] = doc[0].drawings()
        assert drawing.filled and drawing.fill == (0.0, 0.0, 0.0)
        assert _box(drawing.rect) == (48, 80, 200, 100)

    def test_a_noview_annotation_is_not_drawn(self, tmp_path) -> None:
        path = tmp_path / "noview.pdf"
        _annotated_pdf(path, flags=32)  # NoView: MuPDF leaves it out, pdfium's flatten does not
        with _open(path) as doc:
            assert doc[0].drawings() == []

    def test_reading_drawings_leaves_the_page_unflattened(self, pdf_path) -> None:
        with _open(pdf_path) as doc:
            page = doc[0]
            first = page.drawings()
            assert page.drawings() == first
            assert "Papilio Barton" not in page.plain_text()
            assert len(page.widgets()) == 1


class TestRender:
    def test_shape_follows_the_render_box(self, pdf_path) -> None:
        clip = Rect(10.3, 20.7, 110.1, 70.9)
        with _open(pdf_path) as doc:
            page = doc[0]
            full = page.render(dpi=96)
            clipped = page.render(dpi=150, clip=clip)
            black = page.render(dpi=72, clip=Rect(120, 210, 180, 230))
            x0, y0, x1, y1 = render_box(page.rect, 150, clip)
        assert full.shape == (800, 534, 3) and full.dtype == np.uint8
        assert clipped.shape == (y1 - y0, x1 - x0, 3) == (105, 209, 3)
        assert black.max() < 10, "the clip lands inside the black rect"

    def test_red_is_red(self, pdf_path) -> None:
        """BGR from pdfium, RGB out."""
        with _open(pdf_path) as doc:
            pixel = doc[0].render(dpi=72, clip=Rect(90, 470, 110, 480))[5, 10]
        assert pixel[0] > 200 and pixel[1] < 50 and pixel[2] < 50

    def test_clip_off_the_page_is_empty(self, pdf_path) -> None:
        with _open(pdf_path) as doc:
            assert doc[1].render(dpi=72, clip=Rect(50, 450, 150, 500)).shape == (0, 100, 3)
