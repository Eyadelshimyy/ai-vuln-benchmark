"""Raster images opened through Pillow, one page per frame.

The formats are MuPDF's that Pillow decodes — PNG, JPEG, TIFF, BMP, GIF,
JPEG 2000, PNM, PSD — plus WebP and AVIF, which MuPDF could not open. Every
frame of an animated GIF, PNG or WebP, or of a JPEG Pillow reads as MPO, is a
page, where MuPDF gave only the first; a PSD's layers are not frames, so it
is its composite, one page.

MuPDF opened a raster image as a document of image-only pages; this keeps that
shape so images stay on the orchestrator's per-page OCR path. Such a page has
no text, tables, drawings or widgets, and — as under MuPDF — no `images()`:
the page *is* the image, so callers render it.

The page rect follows MuPDF's rule, measured in Phase 0 of
the permissive-dependencies work: ``pixels * 72 / dpi`` on both axes from the
horizontal resolution, rounded to a whole dpi; 96 when the file declares none;
72 when the declared value is outside 72..4800. A JPEG 2000 is always 72dpi,
declared or not: MuPDF ignores its resolution box, and a PSD's is unread by
both. Orientation tags are applied before measuring, as MuPDF does.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
from PIL import Image, ImageOps

from womblex.ingest.pdf.types import (
    Block,
    Drawing,
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

_FORMATS = {"PNG", "JPEG", "MPO", "TIFF", "BMP", "GIF", "JPEG2000", "PPM", "PSD", "WEBP", "AVIF"}
_DEFAULT_DPI, _SANE_DPI, _INSANE_DPI = 96, 72, 4800
_UNDECLARED_DPI = {"JPEG2000": 72}
_CM_PER_INCH = 2.54


def _declared_dpi(image: Image.Image) -> float | None:
    """The horizontal resolution the file itself declares, in dpi.

    Read from the format's own fields, not ``info["dpi"]``: Pillow fills that
    with 72 for a JPEG whose EXIF lacks a resolution and with 1 for an untagged
    TIFF, where MuPDF sees no resolution at all.
    """
    if image.format == "JPEG2000":
        # MuPDF never reads the resolution box, so Pillow's value would split the page rect from it.
        return None
    if image.format in ("JPEG", "MPO"):
        unit, density = image.info.get("jfif_unit"), image.info.get("jfif_density")
        if unit in (1, 2) and density:
            return float(density[0]) * (_CM_PER_INCH if unit == 2 else 1)
        exif = image.getexif()
        resolution, unit = exif.get(0x011A), exif.get(0x0128, 2)
    elif image.format == "TIFF":
        tags = getattr(image, "tag_v2", {})
        resolution, unit = tags.get(282), tags.get(296, 2)
    else:
        dpi = image.info.get("dpi")
        return float(dpi[0]) if dpi and dpi[0] else None
    if not resolution:
        return None
    return float(resolution) * (_CM_PER_INCH if unit == 3 else 1)


def page_dpi(image: Image.Image) -> int:
    declared = _declared_dpi(image)
    if not declared:
        return _UNDECLARED_DPI.get(image.format or "", _DEFAULT_DPI)
    dpi = round(declared)
    return dpi if _SANE_DPI <= dpi <= _INSANE_DPI else _SANE_DPI


class ImagePage:
    """One frame, presented as `types.Page`."""

    def __init__(self, doc: ImageDocument, frame: Image.Image, dpi: int, number: int) -> None:
        self._doc = doc
        self._frame = frame
        self._number = number
        width, height = frame.size
        self._rect = Rect(0.0, 0.0, width * 72 / dpi, height * 72 / dpi)

    @property
    def number(self) -> int:
        return self._number

    @property
    def rect(self) -> Rect:
        return self._rect

    @property
    def rotation(self) -> int:
        return 0

    @property
    def rotation_matrix(self) -> Matrix:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    @property
    def doc_name(self) -> str:
        return self._doc.name

    def plain_text(self, *, dehyphenate: bool = True) -> str:
        return ""

    def text_dict(self) -> list[Block]:
        return []

    def words(self, *, dehyphenate: bool = True) -> list[Word]:
        return []

    def text_blocks(self, *, dehyphenate: bool = True) -> list[Block]:
        return []

    def find_tables(self, *, strategy: TableStrategy = "lines") -> list[FoundTable]:
        return []

    def render(self, *, dpi: int, clip: Rect | None = None) -> np.ndarray:
        """Resample to *dpi* bilinearly — the Pillow filter nearest MuPDF's
        output on the vendored PNG fixtures."""
        x0, y0, x1, y1 = render_box(self._rect, dpi, clip)
        width, height = x1 - x0, y1 - y0
        if width <= 0 or height <= 0:
            return np.zeros((max(height, 0), max(width, 0), 3), dtype=np.uint8)
        px_w, px_h = self._frame.size
        sx = px_w / (self._rect.width * dpi / 72)
        sy = px_h / (self._rect.height * dpi / 72)
        box = (x0 * sx, y0 * sy, min(px_w, x1 * sx), min(px_h, y1 * sy))
        out = self._frame.resize((width, height), Image.Resampling.BILINEAR, box=box)
        return np.asarray(out, dtype=np.uint8)

    def images(self) -> list[PageImage]:
        return []

    def drawings(self) -> list[Drawing]:
        return []

    def widgets(self) -> list[Widget]:
        return []


def _rgb(frame: Image.Image) -> Image.Image:
    """Upright 8-bit RGB, any alpha composited onto white as MuPDF renders it.

    Pillow's own conversion clips 16-bit greyscale at 255 where MuPDF scales
    it, so those samples are shifted down to 8 bits first.
    """
    frame = ImageOps.exif_transpose(frame)
    if frame.mode.startswith("I"):
        wide = np.asarray(frame).astype(np.uint32).clip(0, 0xFFFF)
        frame = Image.fromarray((wide >> 8).astype(np.uint8))
    if frame.mode in ("RGBA", "LA", "PA") or "transparency" in frame.info:
        rgba = frame.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return frame.convert("RGB")


class ImageDocument:
    """A raster image, presented as `types.Document`."""

    def __init__(self, path: Path) -> None:
        self._name = str(path)
        self._image = Image.open(path)
        if self._image.format not in _FORMATS:
            fmt = self._image.format
            self._image.close()
            raise ValueError(f"unsupported image format {fmt!r} for {path}")
        # Pillow counts a PSD's layers as frames; the composite is the page.
        frames = 1 if self._image.format == "PSD" else getattr(self._image, "n_frames", 1)
        self._indices = list(range(frames))

    @property
    def page_count(self) -> int:
        return len(self._indices)

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self.page_count

    def __getitem__(self, index: int) -> ImagePage:
        if self._image.format != "PSD":  # Pillow numbers PSD layers from 1; it opens on the composite
            self._image.seek(self._indices[index])
        # The resolution is read before the frame is converted, which drops it.
        dpi = page_dpi(self._image)
        return ImagePage(self, _rgb(self._image), dpi, index % self.page_count)

    def __iter__(self) -> Iterator[ImagePage]:
        return (self[i] for i in range(self.page_count))

    def select(self, pages: list[int]) -> None:
        self._indices = [self._indices[i] for i in pages]

    def close(self) -> None:
        self._image.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

