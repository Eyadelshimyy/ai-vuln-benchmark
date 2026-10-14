"""Raster images through the pdfium backend's Pillow path: MuPDF's page rects.

Each expected rect is what MuPDF reported for the same file (Phase 0 of the
permissive-dependencies work, re-measured for P5): ``pixels * 72 / dpi``
from the horizontal resolution, 96 when undeclared, 72 when out of range.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from womblex.ingest.pdf import open_document

#: 200x100 px throughout, so a 96dpi page is 150x75pt.
_SIZE = (200, 100)


def _image(mode: str = "RGB", colour: object = (0, 0, 255)) -> Image.Image:
    return Image.new(mode, _SIZE, colour)  # type: ignore[arg-type]


def _page_size(path) -> tuple[float, float]:
    with open_document(path) as doc:
        rect = doc[0].rect
    return round(rect.width, 2), round(rect.height, 2)


@pytest.mark.parametrize(("name", "save_kwargs", "expected"), [
    ("untagged.png", {}, (150, 75)),
    ("tagged.png", {"dpi": (150, 150)}, (96, 48)),
    ("anisotropic.png", {"dpi": (150, 300)}, (96, 48)),  # horizontal resolution on both axes
    ("low.png", {"dpi": (50, 50)}, (200, 100)),  # below 72: 72
    ("high.png", {"dpi": (6000, 6000)}, (200, 100)),  # above 4800: 72
    ("jfif.jpg", {"dpi": (300, 300)}, (48, 24)),
    ("untagged.jpg", {}, (150, 75)),
    ("tagged.tif", {"dpi": (300, 300)}, (48, 24)),
    ("untagged.tif", {}, (150, 75)),  # Pillow reports (1, 1); MuPDF sees none
    ("centimetres.tif", {"tiffinfo": {296: 3, 282: 100.0, 283: 100.0}}, (56.69, 28.35)),
])
def test_page_rect_follows_the_declared_resolution(tmp_path, name, save_kwargs, expected) -> None:
    path = tmp_path / name
    _image().save(path, **save_kwargs)
    assert _page_size(path) == expected


def test_exif_without_a_resolution_is_undeclared(tmp_path) -> None:
    """Pillow fills ``info["dpi"]`` with 72 here; MuPDF assumes 96."""
    exif = Image.Exif()
    exif[0x010F] = "scanner"
    path = tmp_path / "exif.jpg"
    _image().save(path, exif=exif)
    assert _page_size(path) == (150, 75)


def test_exif_orientation_is_applied(tmp_path) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    path = tmp_path / "turned.jpg"
    _image().save(path, exif=exif)
    assert _page_size(path) == (75, 150)


def test_multi_frame_tiff_is_one_page_per_frame(tmp_path) -> None:
    path = tmp_path / "frames.tif"
    _image().save(path, save_all=True, append_images=[_image().rotate(90, expand=True)], dpi=(200, 200))
    with open_document(path) as doc:
        sizes = [(page.number, round(page.rect.width), round(page.rect.height)) for page in doc]
        doc.select([1])
        assert doc.page_count == 1 and doc[0].rect.height == 72
    assert sizes == [(0, 72, 36), (1, 36, 72)]


@pytest.mark.parametrize(("name", "fmt"), [
    ("animated.png", None), ("animated.gif", None), ("animated.webp", None), ("camera.jpg", "MPO"),
])
def test_every_frame_is_a_page(tmp_path, name, fmt) -> None:
    """MuPDF gave only the first frame of these."""
    path = tmp_path / name
    _image().save(path, format=fmt, save_all=True, append_images=[_image(colour=(255, 0, 0))])
    with open_document(path) as doc:
        assert doc.page_count == 2 and doc[1].rect.width == 150
        r, g, b = doc[1].render(dpi=96)[40, 100].astype(int)  # lossy formats drift a little
        assert r > 240 and g < 15 and b < 15


@pytest.mark.parametrize(("name", "save_kwargs", "expected"), [
    ("plain.bmp", {}, (150, 75)),
    ("tagged.bmp", {"dpi": (150, 150)}, (96, 48)),
    ("plain.gif", {}, (150, 75)),
    ("plain.ppm", {}, (150, 75)),
    ("plain.jp2", {}, (200, 100)),  # MuPDF assumes 72dpi for an undeclared JPEG 2000
    ("plain.webp", {}, (150, 75)),
    ("plain.avif", {}, (150, 75)),
])
def test_mupdfs_other_formats_and_webp_open(tmp_path, name, save_kwargs, expected) -> None:
    path = tmp_path / name
    _image().save(path, **save_kwargs)
    assert _page_size(path) == expected


@pytest.mark.parametrize("name", ["grey16.png", "grey16.tif"])
def test_16_bit_grey_is_scaled_not_clipped(tmp_path, name) -> None:
    path = tmp_path / name
    Image.fromarray(np.full(_SIZE[::-1], 0x8000, dtype=np.uint16)).save(path)
    with open_document(path) as doc:
        assert (doc[0].render(dpi=96) == 128).all()


def test_an_image_page_has_nothing_but_pixels(tmp_path) -> None:
    path = tmp_path / "plain.png"
    _image().save(path)
    with open_document(path) as doc:
        page = doc[0]
        assert page.plain_text() == "" and page.words() == [] and page.text_dict() == []
        assert page.images() == [] and page.drawings() == [] and page.widgets() == []
        assert page.find_tables() == [] and page.rotation == 0


def test_render_resamples_to_the_render_box(tmp_path) -> None:
    path = tmp_path / "blue.png"
    _image().save(path)
    with open_document(path) as doc:
        full = doc[0].render(dpi=300)
        clipped = doc[0].render(dpi=150, clip=doc[0].rect.of((10.3, 5.2, 50.7, 40.1)))
    assert full.shape == (313, 625, 3) and full.dtype == np.uint8
    assert clipped.shape == (74, 85, 3)
    assert (full[150, 300] == (0, 0, 255)).all()


def test_alpha_is_composited_onto_white(tmp_path) -> None:
    path = tmp_path / "clear.png"
    _image("RGBA", (255, 0, 0, 0)).save(path)
    with open_document(path) as doc:
        assert (doc[0].render(dpi=96) == 255).all()


def test_an_unsupported_image_format_is_refused(tmp_path) -> None:
    path = tmp_path / "icon.ico"
    _image().save(path)
    with pytest.raises(ValueError, match="unsupported image format 'ICO'"):
        open_document(path)


def test_a_file_that_is_no_image_raises(tmp_path) -> None:
    path = tmp_path / "notes.png"
    path.write_text("not an image")
    with pytest.raises(UnidentifiedImageError):
        open_document(path)


def _flat_psd(path, width: int = 40, height: int = 30, resolution_ppi: int | None = None) -> None:
    """A layerless 8-bit RGB PSD: what a flattened Photoshop export is."""
    import struct

    resources = b""
    if resolution_ppi is not None:
        data = struct.pack(">IHHIHH", resolution_ppi << 16, 1, 1, resolution_ppi << 16, 1, 1)
        resources = b"8BIM" + struct.pack(">H", 0x03ED) + b"\x00\x00" + struct.pack(">I", len(data)) + data
    header = b"8BPS" + struct.pack(">H", 1) + bytes(6) + struct.pack(">HIIHH", 3, height, width, 8, 3)
    body = struct.pack(">I", 0) + struct.pack(">I", len(resources)) + resources + struct.pack(">I", 0)
    path.write_bytes(header + body + struct.pack(">H", 0) + bytes([200]) * (width * height * 3))


def _jp2_with_resolution(path, pixels_per_metre: int) -> None:
    """A JPEG 2000 whose header carries a capture-resolution box."""
    import io
    import struct

    buffer = io.BytesIO()
    Image.new("RGB", (40, 30)).save(buffer, "JPEG2000")
    data = bytearray(buffer.getvalue())
    resc = struct.pack(">I4sHHHHBB", 18, b"resc", pixels_per_metre, 1, pixels_per_metre, 1, 0, 0)
    box = struct.pack(">I4s", 8 + len(resc), b"res ") + resc
    start = data.find(b"jp2h") - 4
    length = struct.unpack(">I", data[start:start + 4])[0]
    data[start:start + 4] = struct.pack(">I", length + len(box))
    data[start + length:start + length] = box
    path.write_bytes(bytes(data))


def test_a_flat_psd_opens_as_its_composite(tmp_path) -> None:
    path = tmp_path / "flat.psd"
    _flat_psd(path)
    with open_document(path) as doc:
        assert len(doc) == 1
        assert doc[0].rect.as_tuple() == (0.0, 0.0, 30.0, 22.5)  # 96 dpi: undeclared
        assert doc[0].render(dpi=96).shape == (30, 40, 3)


def test_a_psd_resolution_is_unread_as_under_mupdf(tmp_path) -> None:
    path = tmp_path / "declared.psd"
    _flat_psd(path, resolution_ppi=300)
    assert _page_size(path) == (30.0, 22.5)


def test_a_jpeg_2000_is_72_dpi_whatever_it_declares(tmp_path) -> None:
    path = tmp_path / "declared.jp2"
    _jp2_with_resolution(path, 11811)  # 300 dpi
    assert _page_size(path) == (40.0, 30.0)
