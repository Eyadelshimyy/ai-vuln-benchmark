"""Minimal image I/O on the standard library only.

Decodes PNG (non-interlaced, 8-bit) and BMP (24/32-bit uncompressed).
JPEG / WEBP are delegated to Pillow when it happens to be installed;
without Pillow they raise UnsupportedFormat — Slice still runs.

Also provides PNG encoding so overlays can be rendered without deps.
"""

from __future__ import annotations

import struct
import zlib
from typing import Tuple

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
BMP_MAGIC = b"BM"
JPEG_MAGIC = b"\xff\xd8"


class UnsupportedFormat(ValueError):
    pass


class Bitmap:
    __slots__ = ("width", "height", "data")

    def __init__(self, width: int, height: int, data: bytearray):
        self.width = width
        self.height = height
        self.data = data  # RGBA, row-major

    @classmethod
    def new(cls, width: int, height: int, rgba: Tuple[int, int, int, int]) -> "Bitmap":
        px = bytes(rgba)
        return cls(width, height, bytearray(px * (width * height)))

    def get(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """Pixel at (x, y). Out-of-frame reads raise: a negative
        index would wrap to the image's tail rows and pass those
        pixels off as real data — loud failure is the honest read."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(
                f"pixel ({x}, {y}) outside {self.width}x{self.height}")
        i = (y * self.width + x) * 4
        d = self.data
        return d[i], d[i + 1], d[i + 2], d[i + 3]

    def set(self, x: int, y: int, rgba: Tuple[int, int, int, int]) -> None:
        if len(rgba) != 4:
            raise ValueError("rgba must be a 4-tuple")
        if 0 <= x < self.width and 0 <= y < self.height:
            i = (y * self.width + x) * 4
            self.data[i:i + 4] = bytes(rgba)

    def downscale(self, max_dim: int = 512) -> "Bitmap":
        """Nearest-neighbour downscale so analysis cost stays bounded."""
        m = max(self.width, self.height)
        if m <= max_dim:
            return self
        s = max_dim / m
        w = max(1, round(self.width * s))
        h = max(1, round(self.height * s))
        out = bytearray(w * h * 4)
        for y in range(h):
            sy = min(self.height - 1, int(y / s))
            for x in range(w):
                sx = min(self.width - 1, int(x / s))
                si = (sy * self.width + sx) * 4
                di = (y * w + x) * 4
                out[di:di + 4] = self.data[si:si + 4]
        return Bitmap(w, h, out)


def decode(raw: bytes) -> Bitmap:
    if raw[:8] == PNG_MAGIC:
        return _decode_png(raw)
    if raw[:2] == BMP_MAGIC:
        return _decode_bmp(raw)
    if raw[:2] == JPEG_MAGIC or raw[:4] == b"RIFF":
        return _decode_pillow(raw)
    raise UnsupportedFormat("not a PNG / BMP / JPEG / WEBP image")


def encode_png(bmp: Bitmap) -> bytes:
    stride = bmp.width * 4
    rows = bytearray()
    for y in range(bmp.height):
        rows.append(0)  # filter: none
        rows += bmp.data[y * stride:(y + 1) * stride]
    ihdr = struct.pack(">IIBBBBB", bmp.width, bmp.height, 8, 6, 0, 0, 0)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        c = struct.pack(">I", len(payload)) + tag + payload
        return c + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    return (PNG_MAGIC + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
            + chunk(b"IEND", b""))


def _decode_pillow(raw: bytes) -> Bitmap:
    try:
        import io

        from PIL import Image  # type: ignore
    except ImportError as e:
        raise UnsupportedFormat(
            "JPEG/WEBP need Pillow; PNG and BMP work without it") from e
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    w, h = img.size
    return Bitmap(w, h, bytearray(img.tobytes()))


def _decode_png(raw: bytes) -> Bitmap:
    pos = 8
    width = height = bit_depth = color_type = interlace = None
    idat = bytearray()
    palette = b""
    trns = b""
    while pos + 8 <= len(raw):
        (length,) = struct.unpack(">I", raw[pos:pos + 4])
        tag = raw[pos + 4:pos + 8]
        payload = raw[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            (width, height, bit_depth, color_type,
             _comp, _filt, interlace) = struct.unpack(">IIBBBBB", payload)
        elif tag == b"PLTE":
            palette = payload
        elif tag == b"tRNS":
            trns = payload
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            break
    if width is None:
        raise UnsupportedFormat("PNG missing IHDR")
    if interlace:
        raise UnsupportedFormat("interlaced (Adam7) PNG not supported")
    if bit_depth != 8:
        raise UnsupportedFormat(f"PNG bit depth {bit_depth} not supported (8 only)")

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise UnsupportedFormat(f"PNG color type {color_type} not supported")

    data = zlib.decompress(bytes(idat))
    stride = width * channels
    out = bytearray(width * height * 4)
    prev = bytearray(stride)
    o = di = 0
    for _y in range(height):
        filt = data[o]
        o += 1
        row = bytearray(data[o:o + stride])
        o += stride
        if filt == 1:  # sub
            for i in range(channels, stride):
                row[i] = (row[i] + row[i - channels]) & 0xFF
        elif filt == 2:  # up
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif filt == 3:  # average
            for i in range(stride):
                a = row[i - channels] if i >= channels else 0
                row[i] = (row[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif filt == 4:  # paeth
            for i in range(stride):
                a = row[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                row[i] = (row[i] + pr) & 0xFF
        elif filt != 0:
            raise UnsupportedFormat(f"PNG filter {filt} unknown")
        for x in range(width):
            s = x * channels
            if color_type == 0:
                g = row[s]
                px = (g, g, g, 255)
            elif color_type == 2:
                px = (row[s], row[s + 1], row[s + 2], 255)
            elif color_type == 3:
                idx = row[s]
                pi = idx * 3
                a = trns[idx] if idx < len(trns) else 255
                px = (palette[pi], palette[pi + 1], palette[pi + 2], a)
            elif color_type == 4:
                g = row[s]
                px = (g, g, g, row[s + 1])
            else:  # 6
                px = (row[s], row[s + 1], row[s + 2], row[s + 3])
            out[di:di + 4] = bytes(px)
            di += 4
        prev = row
    return Bitmap(width, height, out)


def _decode_bmp(raw: bytes) -> Bitmap:
    if len(raw) < 54:
        raise UnsupportedFormat("truncated BMP")
    offset, = struct.unpack("<I", raw[10:14])
    header_size, = struct.unpack("<I", raw[14:18])
    if header_size < 40:
        raise UnsupportedFormat("OS/2 BMP not supported")
    width, height, planes, bpp, comp = struct.unpack("<iiHHI", raw[18:34])
    if comp != 0 or planes != 1 or bpp not in (24, 32):
        raise UnsupportedFormat("only uncompressed 24/32-bit BMP supported")
    top_down = height < 0
    height = abs(height)
    stride = ((width * bpp + 31) // 32) * 4
    out = bytearray(width * height * 4)
    for y in range(height):
        sy = y if top_down else height - 1 - y
        row = offset + sy * stride
        for x in range(width):
            si = row + x * (bpp // 8)
            di = (y * width + x) * 4
            a = raw[si + 3] if bpp == 32 else 255
            out[di:di + 4] = bytes((raw[si + 2], raw[si + 1], raw[si], a))
    return Bitmap(width, height, out)
