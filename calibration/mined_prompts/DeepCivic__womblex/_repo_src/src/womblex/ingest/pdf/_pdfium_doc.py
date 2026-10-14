"""The pypdfium2 backend behind the `ingest.pdf` seam: document and page.

PDFium's coordinates are PDF user space, origin bottom-left, y growing upward;
`types.py`'s are MuPDF's, origin at the crop box's top-left, y downward. Every
rect leaves this module through `PdfiumPage._rect`, which does that flip, so no
caller sees pdfium's convention. Like MuPDF, objects are reported in unrotated
page space while `rect` is the rotated page.

Text is `_text.py` and the table finder is `_tables.py`.

PDFium is not thread-safe (see `docs/decisions.md`); nothing here locks, since
extraction runs one document at a time per process.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np

# pypdfium2 ships neither stubs nor a py.typed marker.
import pypdfium2 as pdfium  # type: ignore[import-untyped]
import pypdfium2.raw as pdfium_c  # type: ignore[import-untyped]

from womblex.ingest.pdf import _image, _tables, _text
from womblex.ingest.pdf.types import (
    Block,
    Drawing,
    DrawingKind,
    FoundTable,
    Matrix,
    PageImage,
    Rect,
    TableStrategy,
    Widget,
    Word,
    render_box,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from womblex.ingest.pdf.types import Document

_IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _compose(inner: Matrix, outer: Matrix) -> Matrix:
    """The transform applying *inner* then *outer*."""
    a, b, c, d, e, f = inner
    A, B, C, D, E, F = outer
    return (a * A + b * C, a * B + b * D, c * A + d * C, c * B + d * D, e * A + f * C + E, e * B + f * D + F)


def _drawing_kind(obj: pdfium.PdfObject) -> DrawingKind | None:
    fill_mode, stroke = ctypes.c_int(0), ctypes.c_int(0)
    if not pdfium_c.FPDFPath_GetDrawMode(obj, ctypes.byref(fill_mode), ctypes.byref(stroke)):
        return None
    if fill_mode.value and stroke.value:
        return "fill_stroke"
    if fill_mode.value:
        return "fill"
    return "stroke" if stroke.value else None


def _fill(obj: pdfium.PdfObject) -> tuple[float, float, float] | None:
    rgba = [ctypes.c_uint(0) for _ in range(4)]
    if not pdfium_c.FPDFPageObj_GetFillColor(obj, *(ctypes.byref(c) for c in rgba)):
        return None
    return (rgba[0].value / 255, rgba[1].value / 255, rgba[2].value / 255)


def _path_points(obj: pdfium.PdfObject) -> list[tuple[float, float]]:
    """The path's own points, before its matrix — without the stroke width
    `get_bounds` adds, which MuPDF's drawing rect leaves out."""
    points = []
    x, y = ctypes.c_float(0), ctypes.c_float(0)
    for i in range(pdfium_c.FPDFPath_CountSegments(obj)):
        segment = pdfium_c.FPDFPath_GetPathSegment(obj, i)
        if segment and pdfium_c.FPDFPathSegment_GetPoint(segment, ctypes.byref(x), ctypes.byref(y)):
            points.append((x.value, y.value))
    return points


def _wide_string(fn: object, *args: object) -> str:
    """Read one of pdfium's UTF-16LE out-parameters: size first, then fill."""
    size = fn(*args, None, 0)  # type: ignore[operator]
    if size <= 2:
        return ""
    buffer = ctypes.create_string_buffer(size)
    fn(*args, ctypes.cast(buffer, ctypes.POINTER(pdfium_c.FPDF_WCHAR)), size)  # type: ignore[operator]
    return buffer.raw[: size - 2].decode("utf-16-le")


class PdfiumPage:
    """One pypdfium2 page, presented as `types.Page`."""

    def __init__(self, doc: PdfiumDocument, page: pdfium.PdfPage, number: int, source: int) -> None:
        self._doc = doc
        self._page = page
        self._number = number
        self._source = source
        left, _bottom, _right, top = page.get_cropbox()
        self._origin = (left, top)
        self._char_cache: list[_text.Char] | None = None

    def _rect(self, box: tuple[float, float, float, float], matrix: Matrix = _IDENTITY) -> Rect:
        """A user-space box, through *matrix*, into top-left page space."""
        x0, y0, x1, y1 = Rect.of(box).transform(matrix).as_tuple()
        left, top = self._origin
        return Rect.of((x0 - left, top - y1, x1 - left, top - y0))

    def _flip(self, box: tuple[float, float, float, float]) -> Rect:
        """`_rect` without a matrix, for the per-character hot path."""
        left, top = self._origin
        return Rect.of((box[0] - left, top - box[3], box[2] - left, top - box[1]))

    def _objects(self, page: pdfium.PdfPage | None = None) -> Iterator[tuple[pdfium.PdfObject, Matrix]]:
        """Every object, form XObjects descended, with its container's transform.

        An object inside a form reports bounds in the form's space, so each
        level carries the composed matrices of the forms above it.
        """
        to_page: list[Matrix] = [_IDENTITY]
        for obj in (page or self._page).get_objects():
            del to_page[obj.level + 1:]
            if obj.type == pdfium_c.FPDF_PAGEOBJ_FORM:
                to_page.append(_compose(tuple(obj.get_matrix().get()), to_page[obj.level]))
            yield obj, to_page[obj.level]

    def _path_objects(self) -> Iterator[tuple[pdfium.PdfObject, Matrix]]:
        """Every visible path with its own space's transform to user space."""
        for obj, matrix in self._objects():
            if obj.type == pdfium_c.FPDF_PAGEOBJ_PATH and _drawing_kind(obj) is not None:
                yield obj, _compose(tuple(obj.get_matrix().get()), matrix)

    @property
    def number(self) -> int:
        return self._number

    @property
    def rect(self) -> Rect:
        width, height = self._page.get_size()
        return Rect(0.0, 0.0, float(width), float(height))

    @property
    def rotation(self) -> int:
        return int(self._page.get_rotation())

    @property
    def rotation_matrix(self) -> Matrix:
        """MuPDF's: unrotated top-left space onto the rotated page."""
        left, bottom, right, top = self._page.get_cropbox()
        w, h = right - left, top - bottom
        return {
            90: (0.0, 1.0, -1.0, 0.0, h, 0.0),
            180: (-1.0, 0.0, 0.0, -1.0, w, h),
            270: (0.0, -1.0, 1.0, 0.0, 0.0, w),
        }.get(self.rotation, _IDENTITY)

    @property
    def doc_name(self) -> str:
        return self._doc.name

    def _text_boxes(self, textpage: pdfium.PdfTextPage) -> list[tuple[Rect, str]]:
        """Each text object's box in top-left page space and its text, in content order."""
        return [
            (self._rect(obj.get_bounds(), matrix), _wide_string(pdfium_c.FPDFTextObj_GetText, obj.raw, textpage.raw))
            for obj, matrix in self._objects()
            if obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT
        ]

    def _chars(self) -> list[_text.Char]:
        if self._char_cache is None:
            self._char_cache = _text.read_chars(
                self._page, self._flip, self._rect(self._page.get_cropbox()), self._text_boxes,
            )
        return self._char_cache

    def plain_text(self, *, dehyphenate: bool = True) -> str:
        return _text.plain_text(self._chars())

    def text_dict(self) -> list[Block]:
        return _text.text_dict(self._chars())

    def words(self, *, dehyphenate: bool = True) -> list[Word]:
        return _text.words(self._chars())

    def text_blocks(self, *, dehyphenate: bool = True) -> list[Block]:
        return _text.text_blocks(self._chars())

    def find_tables(self, *, strategy: TableStrategy = "lines") -> list[FoundTable]:
        return _tables.find_tables(
            strategy, self.rect, self._chars,
            lambda: _tables.read_polylines(self._path_objects(), self._origin),
            rotation=self.rotation_matrix if self.rotation else None,
        )

    def render(self, *, dpi: int, clip: Rect | None = None) -> np.ndarray:
        x0, y0, x1, y1 = render_box(self.rect, dpi, clip)
        width, height = x1 - x0, y1 - y0
        if width <= 0 or height <= 0:
            return np.zeros((max(height, 0), max(width, 0), 3), dtype=np.uint8)
        _, _, full_w, full_h = render_box(self.rect, dpi)
        bitmap = pdfium.PdfBitmap.new_native(width, height, pdfium_c.FPDFBitmap_BGR)
        try:
            bitmap.fill_rect((255, 255, 255, 255), 0, 0, width, height)
            # Offset the full-page canvas so only the clip lands in the bitmap.
            args = (bitmap, self._page, -x0, -y0, full_w, full_h, 0, pdfium_c.FPDF_ANNOT)
            pdfium_c.FPDF_RenderPageBitmap(*args)
            if self._doc.forms is not None:
                pdfium_c.FPDF_FFLDraw(self._doc.forms, *args)
            return np.ascontiguousarray(bitmap.to_numpy()[:, :, ::-1])
        finally:
            bitmap.close()

    def images(self) -> list[PageImage]:
        """One per drawn image object. PDFium does not expose the resource's
        object number, so ``xref`` stays 0."""
        return [
            PageImage(rect=self._rect(obj.get_bounds(), matrix))
            for obj, matrix in self._objects()
            if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE
        ]

    def drawings(self) -> list[Drawing]:
        """Page paths, and an annotation's appearance paths as MuPDF reports them.

        pdfium's page objects leave annotations out, so a page that has any is
        read again from a flattened scratch copy (the main document is never
        flattened: that would put widget text into the text layer).
        """
        flat = self._doc.flattened(self._source) if pdfium_c.FPDFPage_GetAnnotCount(self._page) else None
        out: list[Drawing] = []
        for obj, matrix in self._objects(flat):
            if obj.type != pdfium_c.FPDF_PAGEOBJ_PATH or (kind := _drawing_kind(obj)) is None:
                continue
            points = _path_points(obj)
            if points:
                xs, ys = [p[0] for p in points], [p[1] for p in points]
                own = tuple(obj.get_matrix().get())
                rect = self._rect((min(xs), min(ys), max(xs), max(ys)), _compose(own, matrix))
            else:
                rect = self._rect(obj.get_bounds(), matrix)
            fill = _fill(obj) if kind != "stroke" else None
            out.append(Drawing(kind=kind, rect=rect, fill=fill))
        return out

    def widgets(self) -> list[Widget]:
        forms = self._doc.forms
        if forms is None:
            return []
        out: list[Widget] = []
        for i in range(pdfium_c.FPDFPage_GetAnnotCount(self._page)):
            annot = pdfium_c.FPDFPage_GetAnnot(self._page, i)
            try:
                if pdfium_c.FPDFAnnot_GetSubtype(annot) != pdfium_c.FPDF_ANNOT_WIDGET:
                    continue
                box = pdfium_c.FS_RECTF()
                pdfium_c.FPDFAnnot_GetRect(annot, box)
                out.append(Widget(
                    field_name=_wide_string(pdfium_c.FPDFAnnot_GetFormFieldName, forms, annot),
                    field_value=_wide_string(pdfium_c.FPDFAnnot_GetFormFieldValue, forms, annot),
                    rect=self._rect((box.left, box.bottom, box.right, box.top)),
                ))
            finally:
                pdfium_c.FPDFPage_CloseAnnot(annot)
        return out


class PdfiumDocument:
    """One pypdfium2 document, presented as `types.Document`.

    `select` keeps an index list rather than editing the document, which is
    all MuPDF's in-memory ``select`` gave its callers.
    """

    def __init__(self, path: Path) -> None:
        self._name = str(path)
        self._pdf = pdfium.PdfDocument(str(path))
        # The form environment has to exist before any page loads.
        if self._pdf.get_formtype() != pdfium_c.FORMTYPE_NONE:
            self._pdf.init_forms()
        self._indices = list(range(len(self._pdf)))
        self._scratch: pdfium.PdfDocument | None = None

    @property
    def forms(self) -> pdfium.PdfFormEnv | None:
        return self._pdf.formenv

    @property
    def page_count(self) -> int:
        return len(self._indices)

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self.page_count

    def __getitem__(self, index: int) -> PdfiumPage:
        source = self._indices[index]
        return PdfiumPage(self, self._pdf[source], index % self.page_count, source)

    def flattened(self, source: int) -> pdfium.PdfPage:
        """Page *source* of a scratch copy with its annotations flattened into content."""
        if self._scratch is None:
            self._scratch = pdfium.PdfDocument(self._name)
        page = self._scratch[source]
        # pdfium's display flatten skips Hidden but keeps NoView, which MuPDF does not draw.
        for i in reversed(range(pdfium_c.FPDFPage_GetAnnotCount(page))):
            annot = pdfium_c.FPDFPage_GetAnnot(page, i)
            flags = pdfium_c.FPDFAnnot_GetFlags(annot)
            pdfium_c.FPDFPage_CloseAnnot(annot)
            if flags & pdfium_c.FPDF_ANNOT_FLAG_NOVIEW:
                pdfium_c.FPDFPage_RemoveAnnot(page, i)
        pdfium_c.FPDFPage_Flatten(page, pdfium_c.FLAT_NORMALDISPLAY)
        return self._scratch[source]

    def __iter__(self) -> Iterator[PdfiumPage]:
        return (self[i] for i in range(self.page_count))

    def select(self, pages: list[int]) -> None:
        self._indices = [self._indices[i] for i in pages]

    def close(self) -> None:
        # pypdfium2 closes the pages and form environment before the document.
        if self._scratch is not None:
            self._scratch.close()
        self._pdf.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_document(path: Path) -> Document:
    """A PDF through pdfium; anything else through Pillow, as an image.

    Sniffed rather than trusted to the suffix, as MuPDF did: the PDF header may
    sit anywhere in the first kilobyte. A format Pillow opens but this backend
    does not support is a `ValueError`; a file that is no image at all raises
    Pillow's `UnidentifiedImageError`, as MuPDF raised on one it could not open.
    """
    with Path(path).open("rb") as handle:
        head = handle.read(1024)
    if b"%PDF" in head:
        return PdfiumDocument(path)
    return _image.ImageDocument(path)
