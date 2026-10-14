"""Deterministic PNG renders of FPGA artifacts, stdlib only.

Six views give a vision model something to look at: the package pin map,
utilization against budgets, timing against requirements, the placed
floorplan, simulation waveforms (VCD) and the gate report itself. Renders
are advisory; a render failure never changes a gate verdict. Every image
is RGB8 PNG with filter 0 and zlib level 9, so bytes are deterministic.

The canvas records the bounding box of every text run and every
registered solid region; ``layout_problems()`` reports overlaps and
clipping so tests can assert every view is clean.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import struct
import zlib
from pathlib import Path
from typing import Any, Literal, cast

from .contract import FpgaContract
from .devices import DeviceProfile
from .gates import Check

MAX_DIM = 4096
VCD_MAX_BYTES = 64 * 1024 * 1024
VCD_MAX_CHANGES = 2_000_000
MAX_WAVE_ROWS = 32
MIN_TEXT_SCALE = 2

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GREY = (200, 200, 200)
DARK_GREY = (110, 110, 110)
RED = (210, 40, 40)
GREEN = (30, 150, 70)
BLUE = (40, 110, 220)
ORANGE = (235, 160, 30)
PURPLE = (150, 80, 200)
TEAL = (30, 160, 160)
LIGHT_GREEN = (220, 240, 220)

Color = tuple[int, int, int]

# Classic 5x7 ASCII font (HD44780/GLCD style), 0x20-0x7E, each glyph seven
# rows of five pixels. Lowercase is a real alphabet, not a fold.
_FONT_LIST = [
    (" ", ["     "] * 7),
    ("!", ["  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "     ", "  #  "]),
    ('"', [" # # ", " # # ", " # # ", "     ", "     ", "     ", "     "]),
    ("#", [" # # ", " # # ", "#####", " # # ", "#####", " # # ", " # # "]),
    ("$", ["  #  ", " ####", "# #  ", " ### ", "  # #", "#### ", "  #  "]),
    ("%", ["##   ", "##  #", "   # ", "  #  ", " #   ", "#  ##", "   ##"]),
    ("&", [" ##  ", "#  # ", "# #  ", " #   ", "# # #", "#  # ", " ## #"]),
    ("'", ["  #  ", "  #  ", " #   ", "     ", "     ", "     ", "     "]),
    ("(", ["   # ", "  #  ", " #   ", " #   ", " #   ", "  #  ", "   # "]),
    (")", [" #   ", "  #  ", "   # ", "   # ", "   # ", "  #  ", " #   "]),
    ("*", ["     ", "  #  ", "# # #", " ### ", "# # #", "  #  ", "     "]),
    ("+", ["     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     "]),
    (",", ["     ", "     ", "     ", "     ", " ##  ", " ##  ", " #   "]),
    ("-", ["     ", "     ", "     ", "#####", "     ", "     ", "     "]),
    (".", ["     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "]),
    ("/", ["     ", "    #", "   # ", "  #  ", " #   ", "#    ", "     "]),
    ("0", [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "]),
    ("1", ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "]),
    ("2", [" ### ", "#   #", "    #", "  ## ", " #   ", "#    ", "#####"]),
    ("3", ["#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "]),
    ("4", ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "]),
    ("5", ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "]),
    ("6", ["  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "]),
    ("7", ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "]),
    ("8", [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "]),
    ("9", [" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "]),
    (":", ["     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     "]),
    (";", ["     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", " #   "]),
    ("<", ["   # ", "  #  ", " #   ", "#    ", " #   ", "  #  ", "   # "]),
    ("=", ["     ", "     ", "#####", "     ", "#####", "     ", "     "]),
    (">", [" #   ", "  #  ", "   # ", "    #", "   # ", "  #  ", " #   "]),
    ("?", [" ### ", "#   #", "    #", "   # ", "  #  ", "     ", "  #  "]),
    ("@", [" ### ", "#   #", "# ###", "# # #", "# ###", "#    ", " ### "]),
    ("A", [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"]),
    ("B", ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "]),
    ("C", [" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "]),
    ("D", ["###  ", "#  # ", "#   #", "#   #", "#   #", "#  # ", "###  "]),
    ("E", ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"]),
    ("F", ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "]),
    ("G", [" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ####"]),
    ("H", ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"]),
    ("I", [" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "]),
    ("J", ["    #", "    #", "    #", "    #", "    #", "#   #", " ### "]),
    ("K", ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"]),
    ("L", ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"]),
    ("M", ["#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"]),
    ("N", ["#   #", "#   #", "##  #", "# # #", "#  ##", "#   #", "#   #"]),
    ("O", [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "]),
    ("P", ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "]),
    ("Q", [" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"]),
    ("R", ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"]),
    ("S", [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "]),
    ("T", ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "]),
    ("U", ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "]),
    ("V", ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "]),
    ("W", ["#   #", "#   #", "#   #", "# # #", "# # #", "# # #", " # # "]),
    ("X", ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"]),
    ("Y", ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "]),
    ("Z", ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"]),
    ("[", [" ### ", " #   ", " #   ", " #   ", " #   ", " #   ", " ### "]),
    ("\\", ["     ", "#    ", " #   ", "  #  ", "   # ", "    #", "     "]),
    ("]", [" ### ", "   # ", "   # ", "   # ", "   # ", "   # ", " ### "]),
    ("^", ["  #  ", " # # ", "#   #", "     ", "     ", "     ", "     "]),
    ("_", ["     ", "     ", "     ", "     ", "     ", "     ", "#####"]),
    ("`", [" #   ", "  #  ", "     ", "     ", "     ", "     ", "     "]),
    ("a", ["     ", "     ", " ### ", "    #", " ####", "#   #", " ####"]),
    ("b", ["#    ", "#    ", "#### ", "#   #", "#   #", "#   #", "#### "]),
    ("c", ["     ", "     ", " ### ", "#   #", "#    ", "#   #", " ### "]),
    ("d", ["    #", "    #", " ####", "#   #", "#   #", "#   #", " ####"]),
    ("e", ["     ", "     ", " ### ", "#   #", "#####", "#    ", " ### "]),
    ("f", ["  ## ", " #  #", " #   ", "###  ", " #   ", " #   ", " #   "]),
    ("g", ["     ", "     ", " ####", "#   #", " ####", "    #", " ### "]),
    ("h", ["#    ", "#    ", "#### ", "#   #", "#   #", "#   #", "#   #"]),
    ("i", ["  #  ", "     ", " ##  ", "  #  ", "  #  ", "  #  ", " ### "]),
    ("j", ["   # ", "     ", "  ## ", "   # ", "   # ", "#  # ", " ##  "]),
    ("k", ["#    ", "#    ", "#  # ", "# #  ", "##   ", "# #  ", "#  # "]),
    ("l", [" ##  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "]),
    ("m", ["     ", "     ", "## # ", "# # #", "# # #", "#   #", "#   #"]),
    ("n", ["     ", "     ", "#### ", "#   #", "#   #", "#   #", "#   #"]),
    ("o", ["     ", "     ", " ### ", "#   #", "#   #", "#   #", " ### "]),
    ("p", ["     ", "     ", "#### ", "#   #", "#### ", "#    ", "#    "]),
    ("q", ["     ", "     ", " ####", "#   #", " ####", "    #", "    #"]),
    ("r", ["     ", "     ", "# ## ", "##  #", "#    ", "#    ", "#    "]),
    ("s", ["     ", "     ", " ####", "#    ", " ### ", "    #", "#### "]),
    ("t", [" #   ", " #   ", "#### ", " #   ", " #   ", " #  #", "  ## "]),
    ("u", ["     ", "     ", "#   #", "#   #", "#   #", "#   #", " ####"]),
    ("v", ["     ", "     ", "#   #", "#   #", "#   #", " # # ", "  #  "]),
    ("w", ["     ", "     ", "#   #", "#   #", "# # #", "# # #", " # # "]),
    ("x", ["     ", "     ", "#   #", " # # ", "  #  ", " # # ", "#   #"]),
    ("y", ["     ", "     ", "#   #", "#   #", " ####", "    #", " ### "]),
    ("z", ["     ", "     ", "#####", "   # ", "  #  ", " #   ", "#####"]),
    ("{", ["   # ", "  #  ", "  #  ", " #   ", "  #  ", "  #  ", "   # "]),
    ("|", ["  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "]),
    ("}", [" #   ", "   # ", "   # ", "    #", "   # ", "   # ", " #   "]),
    ("~", ["     ", " ##  ", "#  ##", "     ", "     ", "     ", "     "]),
]
FONT_ROWS: dict[str, str] = {char: "".join(rows) for char, rows in _FONT_LIST}
_FALLBACK = FONT_ROWS["?"]
_CELL = 6  # glyph advance in pixels at scale 1


def text_width(text: str, scale: int) -> int:
    """Pixel width of a text run at the given scale."""
    return len(text) * _CELL * scale


def text_height(scale: int) -> int:
    """Pixel height of a text run at the given scale."""
    return 7 * scale


class Canvas:
    """RGB8 pixel canvas with rects, Bresenham lines and 5x7 text.

    Every ``text`` call records its bounding box; ``solid`` registers a
    drawn region (bars, trace bands) that text must not cover.
    ``layout_problems()`` reports text-on-text overlaps, text-on-solid
    overlaps and boxes extending past the canvas.
    """

    def __init__(self, width: int, height: int, background: Color = WHITE) -> None:
        if not (0 < width <= MAX_DIM and 0 < height <= MAX_DIM):
            raise ValueError(f"canvas {width}x{height} exceeds the {MAX_DIM}px cap")
        self.width = width
        self.height = height
        self.pixels = bytearray(background * (width * height))
        self.text_boxes: list[tuple[int, int, int, int, str]] = []
        self._solids: list[tuple[int, int, int, int]] = []

    def _set(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset : offset + 3] = bytes(color)

    def fill_rect(self, x: int, y: int, width: int, height: int, color: Color) -> None:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + width), min(self.height, y + height)
        if x1 <= x0 or y1 <= y0:
            return
        row = bytes(color) * (x1 - x0)
        for yy in range(y0, y1):
            offset = (yy * self.width + x0) * 3
            self.pixels[offset : offset + len(row)] = row

    def solid(self, x: int, y: int, width: int, height: int) -> None:
        """Register a drawn region that later text must not cover."""
        self._solids.append((x, y, x + width, y + height))

    def rect(self, x: int, y: int, width: int, height: int, color: Color) -> None:
        self.fill_rect(x, y, width, 1, color)
        self.fill_rect(x, y + height - 1, width, 1, color)
        self.fill_rect(x, y, 1, height, color)
        self.fill_rect(x + width - 1, y, 1, height, color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        error = dx + dy
        while True:
            self._set(x0, y0, color)
            if x0 == x1 and y0 == y1:
                return
            e2 = 2 * error
            if e2 >= dy:
                error += dy
                x0 += sx
            if e2 <= dx:
                error += dx
                y0 += sy

    def text(self, x: int, y: int, text: str, color: Color, scale: int = 2) -> None:
        scale = max(MIN_TEXT_SCALE, min(4, scale))
        cx = x
        for char in text:
            glyph = FONT_ROWS.get(char, _FALLBACK)
            for row in range(7):
                bits = glyph[row * 5 : row * 5 + 5]
                for col, mark in enumerate(bits):
                    if mark != " ":
                        self.fill_rect(cx + col * scale, y + row * scale, scale, scale, color)
            cx += _CELL * scale
        if text:
            self.text_boxes.append(
                (x, y, x + text_width(text, scale) - scale, y + text_height(scale), text)
            )

    def layout_problems(self) -> list[str]:
        """Overlapping text boxes, text over registered solids, clipping."""
        problems: list[str] = []
        for i, (x0, y0, x1, y1, text) in enumerate(self.text_boxes):
            if x0 < 0 or y0 < 0 or x1 > self.width or y1 > self.height:
                problems.append(f"text {text!r} clipped at ({x0},{y0})-({x1},{y1})")
            for sx0, sy0, sx1, sy1 in self._solids:
                if x0 < sx1 and x1 > sx0 and y0 < sy1 and y1 > sy0:
                    problems.append(f"text {text!r} overlaps solid ({sx0},{sy0})-({sx1},{sy1})")
            for x2, y2, x3, y3, other in self.text_boxes[i + 1 :]:
                if x0 < x3 and x1 > x2 and y0 < y3 and y1 > y2:
                    problems.append(f"text {text!r} overlaps text {other!r}")
        return problems

    def png_bytes(self) -> bytes:
        raw = b"".join(
            b"\x00" + bytes(self.pixels[y * self.width * 3 : (y + 1) * self.width * 3])
            for y in range(self.height)
        )
        compressed = zlib.compress(raw, 9)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", compressed)
            + chunk(b"IEND", b"")
        )


def write_png(canvas: Canvas, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canvas.png_bytes()
    path.write_bytes(data)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": canvas.width,
        "height": canvas.height,
    }


def _legend(
    canvas: Canvas, x: int, y: int, entries: list[tuple[Color, str]], scale: int = 2
) -> int:
    """Swatch + label per entry; returns the y below the last entry."""
    for i, (color, label) in enumerate(entries):
        yy = y + i * (text_height(scale) + 8)
        canvas.fill_rect(x, yy, 14, 14, color)
        canvas.rect(x, yy, 14, 14, BLACK)
        canvas.text(x + 20, yy, label, BLACK, scale)
    return y + len(entries) * (text_height(scale) + 8)


_BGA_NAME = re.compile(r"^[A-Z]{1,2}[0-9]+$")


def _pin_color(contract: FpgaContract, profile: DeviceProfile, name: str) -> Color:
    by_pin = {p.package_pin: p for p in contract.pins}
    clocks = {c.port for c in contract.clocks}
    entry = by_pin.get(name)
    device_pin = profile.pin(name)
    if entry is not None:
        if device_pin is not None and device_pin.caution is not None:
            return ORANGE
        return GREEN if entry.port in clocks else BLUE
    if device_pin is None:
        return DARK_GREY
    if device_pin.caution is not None:
        return RED
    return GREY


def pinmap_canvas(contract: FpgaContract, profile: DeviceProfile) -> Canvas:
    """Package drawing: a BGA grid or a perimeter + pin table."""
    names = [p.name for p in profile.pins]
    scale = 2
    side = 0
    pad, pitch = 40, 46
    is_bga = bool(names) and all(_BGA_NAME.match(n) for n in names)
    margin, top_y = 16, 48
    positions: dict[str, tuple[int, int]] = {}
    if is_bga:
        letters = sorted(
            {match.group(0) for n in names if (match := re.match(r"[A-Z]+", n)) is not None}
        )
        digits = sorted(
            {int(match.group(0)) for n in names if (match := re.search(r"[0-9]+", n)) is not None}
        )
        for row, letter in enumerate(letters):
            for col, digit in enumerate(digits):
                name = f"{letter}{digit}"
                positions[name] = (margin + col * pitch, top_y + row * pitch)
        grid_w = margin + len(digits) * pitch
        grid_h = top_y + len(letters) * pitch
    else:
        match = re.search(r"(\d+)\s*$", profile.package)
        if match:
            count = int(match.group(1))
        else:
            numeric = [int(n) for n in names if n.isdigit()]
            count = max(numeric) if numeric else len(names)
        count = max(count, len(names))
        side = max(1, math.ceil(count / 4))
        edge = side * pitch
        # All package pins 1..count, pin 1 at the top-left, counter-clockwise.
        ordered = [str(n) for n in range(1, count + 1)]
        for index, name in enumerate(ordered):
            edge_i, slot = divmod(index, side)
            if edge_i == 0:  # left edge, top to bottom
                x, y = margin, top_y + slot * pitch
            elif edge_i == 1:  # bottom, left to right
                x, y = margin + slot * pitch, top_y + edge
            elif edge_i == 2:  # right edge, bottom to top
                x, y = margin + edge, top_y + (side - slot) * pitch
            else:  # top edge, right to left (last pin beside pin 1)
                x, y = margin + (side - slot) * pitch, top_y
            positions[name] = (x, y)
        grid_w = margin + edge + pitch
        grid_h = top_y + edge + pitch
    # Table columns measured on content.
    rows = [
        (pin.package_pin, pin.port, pin.net or "-", pin.io_standard or "-", pin.pull or "-")
        for pin in contract.pins
    ]
    headers = ("pin", "port", "net", "io_std", "pull")
    col_w = [
        max(text_width(headers[i], scale), *[text_width(r[i], scale) for r in rows] or [0]) + 16
        for i in range(5)
    ]
    table_w = sum(col_w)
    table_x = grid_w + 32
    row_h = text_height(scale) + 8
    legend_h = 6 * (text_height(scale) + 8) + 16
    height = max(grid_h + legend_h + 24, 48 + (len(rows) + 3) * row_h + 32)
    width = table_x + table_w + 40
    canvas = Canvas(min(width, MAX_DIM), min(height, MAX_DIM))
    title = (
        f"{contract.name} pin map - {profile.part} {profile.package} "
        f"({len(contract.pins)}/{len(names)} user I/O)"
    )
    canvas.text(margin, 12, title, BLACK, scale)
    if is_bga:
        for name, (x, y) in positions.items():
            canvas.fill_rect(x, y, pad, pad, _pin_color(contract, profile, name))
            canvas.rect(x, y, pad, pad, BLACK)
            canvas.text(
                x + max(2, (pad - text_width(name, scale)) // 2),
                y + (pad - text_height(scale)) // 2,
                name,
                BLACK,
                scale,
            )
    else:
        inner = margin + pitch
        inner_size = side * pitch - pitch
        canvas.fill_rect(inner, top_y + pitch, inner_size, inner_size, (245, 245, 245))
        canvas.rect(inner, top_y + pitch, inner_size, inner_size, BLACK)
        canvas.text(inner + 12, top_y + pitch + 12, profile.package, BLACK, scale)
        for name, (x, y) in positions.items():
            color = _pin_color(contract, profile, name)
            canvas.fill_rect(x, y, pad, pad, color)
            canvas.rect(x, y, pad, pad, BLACK)
            canvas.text(
                x + max(2, (pad - text_width(name, scale)) // 2),
                y + (pad - text_height(scale)) // 2,
                name,
                BLACK,
                scale,
            )
    legend_y = grid_h + 12
    used_colors = {_pin_color(contract, profile, n) for n in positions}
    legend_entries: list[tuple[Color, str]] = [
        entry
        for entry in [
            (BLUE, "used signal"),
            (GREEN, "used clock"),
            (ORANGE, "used caution pin (acknowledged)"),
            (RED, "unused caution pin"),
            (GREY, "free user I/O"),
            (DARK_GREY, "not user I/O"),
        ]
        if entry[0] in used_colors
    ]
    _legend(canvas, margin, legend_y, legend_entries, scale)
    x = table_x
    for i, header in enumerate(headers):
        canvas.text(x, 48, header, BLACK, scale)
        x += col_w[i]
    rule_y = 48 + text_height(scale) + 4
    canvas.line(table_x, rule_y, table_x + table_w, rule_y, BLACK)
    for r, row in enumerate(rows):
        y = 48 + text_height(scale) + 12 + r * row_h
        canvas.fill_rect(table_x - 24, y, 14, 14, _pin_color(contract, profile, row[0]))
        canvas.rect(table_x - 24, y, 14, 14, BLACK)
        x = table_x
        for i, value in enumerate(row):
            canvas.text(x, y, value, BLACK, scale)
            x += col_w[i]
    return canvas


def utilization_canvas(
    contract: FpgaContract, profile: DeviceProfile, report: dict[str, Any]
) -> Canvas:
    """One bar per used cell type against its resource-class budget."""
    utilization = cast(dict[str, dict[str, int]], report.get("utilization", {}))
    entries = sorted(
        (cell, int(e.get("used", 0)), int(e.get("available", 0)))
        for cell, e in utilization.items()
        if int(e.get("used", 0)) > 0
    )
    scale = 2
    rows: list[tuple[str, int, int, str, float | None, float]] = []
    for cell, used, available in entries:
        resource_class = profile.resources.get(cell, "other")
        pct = 100.0 * used / available if available else 100.0
        budget = contract.build.budget.limit(resource_class)
        rows.append((cell, used, available, resource_class, budget, pct))
    label_w = max((text_width(f"{c} ({r})", scale) for c, _, _, r, _, _ in rows), default=100)
    value_w = max(
        (
            text_width(
                f"{u}/{a} {p:.1f}%" + (f" budget {b:g}%" if b is not None else ""),
                scale,
            )
            for _, u, a, _, b, p in rows
        ),
        default=100,
    )
    margin, row_h = 16, 34
    bar_x = margin + label_w + 24
    bar_w = 420
    width = bar_x + bar_w + 24 + value_w + 16
    height = 60 + max(1, len(rows)) * row_h + 16
    canvas = Canvas(min(width, MAX_DIM), min(height, MAX_DIM))
    canvas.text(margin, 12, f"{contract.name} utilization - {profile.part}", BLACK, scale)
    for r, (cell, used, available, resource_class, budget, pct) in enumerate(rows):
        y = 60 + r * row_h
        over = budget is not None and pct > budget
        canvas.text(margin, y + 2, f"{cell} ({resource_class})", RED if over else BLACK, scale)
        canvas.fill_rect(bar_x, y, bar_w, 18, (235, 235, 235))
        fill = int(bar_w * min(pct, 100.0) / 100.0)
        canvas.fill_rect(bar_x, y, fill, 18, RED if over else BLUE)
        canvas.solid(bar_x, y, bar_w, 18)
        if budget is not None:
            marker = bar_x + int(bar_w * budget / 100.0)
            canvas.line(marker, y - 3, marker, y + 21, BLACK)
        value = f"{used}/{available} {pct:.1f}%" + (
            f" budget {budget:g}%" if budget is not None else ""
        )
        canvas.text(bar_x + bar_w + 16, y + 2, value, RED if over else BLACK, scale)
    if not rows:
        canvas.text(margin, 60, "no utilization data", RED, scale)
    return canvas


def _path_delay(segments: list[Any]) -> float | None:
    total = 0.0
    seen = False
    for segment in segments:
        if isinstance(segment, dict):
            delay = cast(dict[str, Any], segment).get("delay")
            if isinstance(delay, (int, float)):
                total += float(delay)
                seen = True
    return total if seen else None


def timing_canvas(contract: FpgaContract, report: dict[str, Any]) -> Canvas:
    """Achieved vs required MHz per clock net, with slack and worst paths."""
    fmax = cast(dict[str, dict[str, float]], report.get("fmax", {}))
    paths_data: Any = report.get("critical_paths", {})
    scale = 2
    rows: list[tuple[str, float, float]] = []
    for clock in contract.clocks:
        base = clock.port.split("[", 1)[0]
        pattern = re.compile(rf"(^|[^A-Za-z0-9]){re.escape(base)}([^A-Za-z0-9]|$)")
        nets = [net for net in fmax if pattern.search(net)]
        for net in nets or [clock.port]:
            achieved = float(fmax.get(net, {}).get("achieved", 0.0))
            rows.append((net, achieved, clock.frequency_mhz))

    def _row_text(net: str, achieved: float, required: float) -> str:
        if achieved:
            slack = 1000.0 / required - 1000.0 / achieved
            return f"{achieved:.2f} MHz achieved / {required:g} MHz required, slack {slack:+.1f} ns"
        return f"no timing data / {required:g} MHz required"

    worst: list[tuple[str, dict[str, Any]]] = []
    if isinstance(paths_data, dict):
        for net, info in cast(dict[str, Any], paths_data).items():
            if isinstance(info, dict):
                worst.append((str(net), cast(dict[str, Any], info)))
    elif isinstance(paths_data, list):
        for entry in cast(list[Any], paths_data)[:6]:
            if isinstance(entry, dict):
                item = cast(dict[str, Any], entry)
                merged = dict(item)
                merged.setdefault("segments", item.get("path", []))
                worst.append((str(item.get("from", "?")), merged))

    seg_lines: list[list[str]] = []
    for label, info in worst:
        segments = info.get("segments", [])
        total = info.get("total_delay_ns")
        if not isinstance(total, (int, float)):
            total = _path_delay(cast(list[Any], segments) if isinstance(segments, list) else [])
        total_text = f"{float(total):.2f} ns" if isinstance(total, (int, float)) else "? ns"
        lines = [
            f"worst path on {label}",
            f"{info.get('from', '?')} -> {info.get('to', '?')}  total {total_text}",
        ]
        top_segments: list[dict[str, Any]] = []
        if isinstance(segments, list):
            segment_dicts = [
                cast(dict[str, Any], s) for s in cast(list[Any], segments) if isinstance(s, dict)
            ]
            top_segments = sorted(
                segment_dicts, key=lambda seg: float(seg.get("delay", 0.0) or 0.0), reverse=True
            )[:5]
        for seg in top_segments:
            net = str(seg.get("net") or seg.get("from", {}).get("cell", "?"))[:48]
            lines.append(f"  {seg.get('type', '?')} {float(seg.get('delay', 0.0)):.2f} ns {net}")
        seg_lines.append(lines)

    label_w = max((text_width(n, scale) for n, _, _ in rows), default=80)
    value_w = max((text_width(_row_text(n, a, r), scale) for n, a, r in rows), default=80)
    worst_w = max(
        (text_width(line, scale) for lines in seg_lines for line in lines),
        default=80,
    )
    margin, row_h = 16, 36
    bar_x = margin + label_w + 24
    bar_w = 320
    width = max(bar_x + bar_w + 24 + value_w + 24, margin + worst_w + 24)
    seg_h = sum(len(lines) * (text_height(scale) + 6) + 14 for lines in seg_lines)
    height = 60 + max(1, len(rows)) * row_h + seg_h + 40
    canvas = Canvas(min(width, MAX_DIM), min(height, MAX_DIM))
    canvas.text(margin, 12, f"{contract.name} timing", BLACK, scale)
    for r, (net, achieved, required) in enumerate(rows):
        y = 60 + r * row_h
        ok = achieved >= required
        canvas.text(margin, y + 2, net, GREEN if ok else RED, scale)
        scale_max = max(required, achieved, 1.0) * 1.2
        canvas.fill_rect(bar_x, y, bar_w, 18, (235, 235, 235))
        canvas.fill_rect(
            bar_x,
            y,
            int(bar_w * min(achieved, scale_max) / scale_max),
            18,
            GREEN if ok else RED,
        )
        canvas.solid(bar_x, y, bar_w, 18)
        marker = bar_x + int(bar_w * required / scale_max)
        canvas.line(marker, y - 3, marker, y + 21, BLACK)
        canvas.text(
            bar_x + bar_w + 16,
            y + 2,
            _row_text(net, achieved, required),
            GREEN if ok else RED,
            scale,
        )
    if not rows:
        canvas.text(margin, 60, "contract declares no clocks", DARK_GREY, scale)
    y = 60 + max(1, len(rows)) * row_h + 16
    for lines in seg_lines:
        for i, line in enumerate(lines):
            canvas.text(
                margin + (24 if i >= 2 else 0),
                y,
                line,
                DARK_GREY if i else BLACK,
                scale,
            )
            y += text_height(scale) + 6
        y += 14
    return canvas


_BEL = re.compile(r"X(\d+)/?Y(\d+)")
# Constant/global pseudo-cells carry BEL-looking attributes but are not
# placement: skip them rather than classing them "other".
_PSEUDO_CELLS = {"VCC", "GND", "GOWIN_VCC", "GOWIN_GND", "GSR", "PUR"}
_CLASS_COLORS: dict[str, Color] = {
    "logic": BLUE,
    "ram": PURPLE,
    "dsp": ORANGE,
    "io": TEAL,
    "clock": GREEN,
    "pll": GREEN,
    "other": DARK_GREY,
}


def _bel_coords(attributes: dict[str, Any]) -> tuple[int, int] | None:
    for key in ("NEXTPNR_BEL", "bel", "BEL"):
        value = attributes.get(key)
        if isinstance(value, str):
            match = _BEL.search(value)
            if match:
                return int(match.group(1)), int(match.group(2))
    return None


def floorplan_canvas(placed: dict[str, Any], profile: DeviceProfile, top: str) -> Canvas:
    """Tile grid of placed cells, colored by the dominant resource class."""
    modules = cast(dict[str, Any], placed.get("modules", {}))
    module = cast(dict[str, Any] | None, modules.get(top))
    if module is None:
        module = next(iter(modules.values()), None)
    cells = cast(dict[str, Any], (module or {}).get("cells", {}))
    grid: dict[tuple[int, int], dict[str, int]] = {}
    for cell in cells.values():
        cell = cast(dict[str, Any], cell)
        coords = _bel_coords(cast(dict[str, Any], cell.get("attributes", {})))
        cell_type = str(cell.get("type", ""))
        if coords is None or cell_type in _PSEUDO_CELLS:
            continue
        resource_class = profile.resources.get(cell_type, "other")
        bucket = grid.setdefault(coords, {})
        bucket[resource_class] = bucket.get(resource_class, 0) + 1
    if not grid:
        canvas = Canvas(560, 120)
        canvas.text(12, 40, "no NEXTPNR_BEL placement data", RED, 2)
        return canvas
    min_x = min(x for x, _ in grid)
    max_x = max(x for x, _ in grid)
    min_y = min(y for _, y in grid)
    max_y = max(y for _, y in grid)
    nx, ny = max_x - min_x + 1, max_y - min_y + 1
    scale = 2
    tile = 22
    tick_label_w = text_width(str(max(max_x, max_y)), scale) + 8
    origin_x, origin_y = 24 + tick_label_w, 64
    counts: dict[str, int] = {}
    for bucket in grid.values():
        for resource_class, n in bucket.items():
            counts[resource_class] = counts.get(resource_class, 0) + n
    legend_x = origin_x + nx * tile + 32
    legend_w = max(text_width(f"{k} ({v})", scale) for k, v in counts.items()) + 40
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    width = max(legend_x + legend_w, 48 + text_width(f"{top} floorplan ({summary})", scale))
    height = origin_y + ny * tile + 70
    canvas = Canvas(min(width, MAX_DIM), min(height, MAX_DIM))
    canvas.text(24, 12, f"{top} floorplan ({summary})", BLACK, scale)
    for (gx, gy), bucket in sorted(grid.items()):
        dominant, _ = max(bucket.items(), key=lambda kv: kv[1])
        total = sum(bucket.values())
        base = _CLASS_COLORS.get(dominant, DARK_GREY)
        intensity = 255 - min(160, 60 * total)
        color = tuple(min(255, c * intensity // 255 + (255 - intensity)) for c in base)
        canvas.fill_rect(
            origin_x + (gx - min_x) * tile,
            origin_y + (gy - min_y) * tile,
            tile - 1,
            tile - 1,
            cast(Color, color),
        )
    step_x = max(1, math.ceil(nx / 10))
    for gx in range(min_x, max_x + 1, step_x):
        x = origin_x + (gx - min_x) * tile
        label = str(gx)
        canvas.line(x, origin_y - 4, x, origin_y, BLACK)
        canvas.text(x, origin_y - 4 - text_height(scale) - 4, label, BLACK, scale)
    step_y = max(1, math.ceil(ny / 10))
    for gy in range(min_y, max_y + 1, step_y):
        y = origin_y + (gy - min_y) * tile
        label = str(gy)
        canvas.line(origin_x - 4, y, origin_x, y, BLACK)
        canvas.text(
            origin_x - 8 - text_width(label, scale),
            y - text_height(scale) // 2,
            label,
            BLACK,
            scale,
        )
    canvas.text(origin_x + nx * tile // 2, origin_y + ny * tile + 12, "X", BLACK, scale)
    canvas.text(origin_x - 16 - tick_label_w, origin_y + ny * tile // 2, "Y", BLACK, scale)
    _legend(
        canvas,
        legend_x,
        origin_y,
        [(_CLASS_COLORS.get(k, DARK_GREY), f"{k} ({v})") for k, v in sorted(counts.items())],
        scale,
    )
    return canvas


def _bus_text(value: str) -> str:
    """Display text for a bus/string change value."""
    if set(value) <= {"0", "1"}:
        return _hex(value)
    if value.startswith("s"):
        return value[1:]
    if set(value) <= {"x", "z"}:
        return "x"
    return value


def _hex(bits: str) -> str:
    if set(bits) - {"0", "1"}:
        return "x"
    value = int(bits, 2) if bits else 0
    width = max(1, (len(bits) + 3) // 4)
    return f"{value:0{width}x}"


class VcdSignal:
    __slots__ = ("changes", "name", "scope", "width")

    def __init__(self, name: str, scope: str, width: int) -> None:
        self.name = name
        self.scope = scope
        self.width = width
        self.changes: list[tuple[int, str]] = []


def parse_vcd(path: Path) -> tuple[list[VcdSignal], str, bool]:
    """Parse a VCD file; returns (signals, timescale, truncated)."""
    size = path.stat().st_size
    truncated = size > VCD_MAX_BYTES
    with path.open("rb") as stream:
        data = stream.read(VCD_MAX_BYTES)
    text = data.decode("ascii", "replace")
    header, _, body = text.partition("$enddefinitions")
    signals: dict[str, VcdSignal] = {}
    timescale = "1ns"
    scopes: list[str] = []
    top_scope: str | None = None
    for match in re.finditer(r"\$(timescale|scope|upscope|var)\b([^$]*)", header):
        kind, rest = match.group(1), match.group(2)
        if kind == "timescale":
            parts = rest.replace("$end", "").split()
            if parts:
                timescale = parts[0]
        elif kind == "scope":
            parts = rest.split()
            if len(parts) >= 2:
                scopes.append(parts[1])
                if top_scope is None:
                    top_scope = parts[1]
        elif kind == "upscope":
            if scopes:
                scopes.pop()
        elif kind == "var":
            parts = rest.replace("$end", "").split()
            if len(parts) >= 4:
                ident = parts[2]
                name = parts[3].split("[")[0]
                is_string = parts[0].strip() in ("string", "real")
                width = int(parts[1]) if parts[1].isdigit() else 1
                if is_string:
                    width = 0
                signals[ident] = VcdSignal(name, ".".join(scopes), width)
    changes = 0
    time = 0
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0] == "#":
            with contextlib.suppress(ValueError):
                time = int(line[1:])
            continue
        if line[0] == "$":
            continue
        if line[0] in "bBrRsS":
            value, _, ident = line[1:].partition(" ")
            signal = signals.get(ident)
            if signal is not None:
                signal.changes.append((time, value.lower() if line[0] in "bBrR" else value))
                changes += 1
        else:
            signal = signals.get(line[1:])
            if signal is not None:
                signal.changes.append((time, line[0].lower()))
                changes += 1
        if changes > VCD_MAX_CHANGES:
            truncated = True
            break
    ordered = sorted(
        signals.values(),
        key=lambda s: (s.scope.count("."), s.scope != top_scope, s.scope, s.name),
    )
    return ordered[:MAX_WAVE_ROWS], timescale, truncated


def _timescale_unit(timescale: str) -> tuple[float, str]:
    """Numeric value of one VCD time unit in seconds and the timescale text."""
    match = re.match(r"(\d+)\s*(fs|ps|ns|us|ms|s)", timescale.lower())
    if match:
        base = {"fs": 1e-15, "ps": 1e-12, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
        return float(match.group(1)) * base[match.group(2)], match.group(2)
    return 1.0, "s"


def _eng_unit(span_seconds: float) -> tuple[float, str]:
    for factor, unit in (
        (1e-15, "fs"),
        (1e-12, "ps"),
        (1e-9, "ns"),
        (1e-6, "us"),
        (1e-3, "ms"),
        (1.0, "s"),
    ):
        if span_seconds * (1.0 / factor) < 1000 or unit == "s":
            return factor, unit
    return 1.0, "s"


def waveform_canvas(vcd_path: Path, title: str) -> Canvas:
    """Digital traces from a VCD file: scalars waveforms, buses/text boxes."""
    signals, timescale, truncated = parse_vcd(vcd_path)
    unit_s, _ = _timescale_unit(timescale)
    t0 = min((c[0] for s in signals for c in s.changes), default=0)
    t1 = max((c[0] for s in signals for c in s.changes), default=1)
    span = max(1, t1 - t0)
    factor, unit = _eng_unit(span * unit_s)
    span_text = f"{t0 * unit_s / factor:g} {unit} .. {t1 * unit_s / factor:g} {unit}"
    scale = 2
    # Signal labels: qualify duplicates with their scope relative to the top.
    name_counts: dict[str, int] = {}
    for signal in signals:
        name_counts[signal.name] = name_counts.get(signal.name, 0) + 1
    top_scope = signals[0].scope.split(".")[0] if signals else ""

    def _label(signal: VcdSignal) -> str:
        if name_counts.get(signal.name, 0) <= 1:
            base = signal.name
        else:
            scope = signal.scope
            if top_scope and scope.startswith(top_scope + "."):
                scope = scope[len(top_scope) + 1 :]
            elif scope == top_scope:
                scope = ""
            base = f"{scope}.{signal.name}" if scope else signal.name
        if len(base) > 32:
            base = "..." + base[-29:]
        suffix = f" [{signal.width}]" if signal.width > 1 else ""
        return base + suffix

    label_w = max((text_width(_label(s), scale) for s in signals), default=120)
    margin = 16
    plot_x = margin + label_w + 16
    plot_w = 720
    tick_max_w = 0
    tick_values = [t0 + span * i // 10 for i in range(11)]
    tick_labels = [f"{t * unit_s / factor:g} {unit}" for t in tick_values]
    if tick_labels:
        tick_max_w = max(text_width(label, scale) for label in tick_labels)
    width = plot_x + plot_w + 16 + tick_max_w // 2
    row_h = 34
    axis_y = 64 + max(1, len(signals)) * row_h + 10
    height = axis_y + 44
    canvas = Canvas(min(width, MAX_DIM), min(height, MAX_DIM))
    heading = f"{title} {span_text}" + (" TRUNCATED" if truncated else "")
    canvas.text(margin, 12, heading, BLACK, scale)

    def x_of(t: int) -> int:
        return plot_x + int(plot_w * (t - t0) / span)

    for row, signal in enumerate(signals):
        y = 64 + row * row_h
        mid, hi, lo = y + 16, y + 6, y + 26
        canvas.text(margin, y + 4, _label(signal), BLACK, scale)
        changes = signal.changes
        if not changes:
            canvas.text(plot_x + 6, y + 4, "x", RED, scale)
            continue
        if len(changes) > plot_w * 2:
            xs = {x_of(c[0]) for c in changes}
            canvas.fill_rect(min(xs), hi, max(xs) - min(xs) + 1, lo - hi, (190, 215, 255))
            canvas.rect(min(xs), hi, max(xs) - min(xs) + 1, lo - hi, BLUE)
            canvas.solid(min(xs), hi, max(xs) - min(xs) + 1, lo - hi)
            continue
        if signal.width == 1:
            prev_t, prev_v = changes[0]
            for t, v in [*changes[1:], (t1, prev_v)]:
                x0, x1 = x_of(prev_t), x_of(t)
                if prev_v in ("x", "z"):
                    canvas.line(x0, mid, x1, mid, RED)
                else:
                    level = hi if prev_v == "1" else lo
                    canvas.line(x0, level, x1, level, BLACK)
                    if t != prev_t:
                        canvas.line(x0, hi if prev_v != "1" else lo, x0, level, BLACK)
                prev_t, prev_v = t, v
            if prev_v in ("x", "z"):
                canvas.line(x_of(prev_t), mid, plot_x + plot_w, mid, RED)
        else:
            prev_t, prev_v = changes[0]
            for t, v in [*changes[1:], (t1, prev_v)]:
                x0, x1 = x_of(prev_t), x_of(t)
                if x1 - x0 < 2:
                    prev_t, prev_v = t, v
                    continue
                canvas.line(x0, hi, x1, hi, BLACK)
                canvas.line(x0, lo, x1, lo, BLACK)
                canvas.line(x0, hi, x0, lo, BLACK)
                canvas.line(x1, hi, x1, lo, BLACK)
                value = _bus_text(prev_v)
                if x1 - x0 > text_width(value, scale) + 8:
                    canvas.text(x0 + 4, mid - text_height(scale) // 2, value, DARK_GREY, scale)
                prev_t, prev_v = t, v
    canvas.line(plot_x, axis_y, plot_x + plot_w, axis_y, BLACK)
    step = max(1, math.ceil((tick_max_w + 16) * 10 / plot_w))
    for i in range(0, 11, step):
        t, label = tick_values[i], tick_labels[i]
        x = plot_x + plot_w * i // 10
        canvas.line(x, axis_y, x, axis_y + 6, BLACK)
        x_text = max(margin, x - text_width(label, scale) // 2)
        canvas.text(x_text, axis_y + 12, label, DARK_GREY, scale)
    return canvas


def report_canvas(report: dict[str, Any]) -> Canvas:
    """One row per gate check with a status chip, plus a verdict banner."""
    checks = cast(list[dict[str, Any]], report.get("checks", []))
    verdict = str(report.get("verdict", "fail"))
    scale = 2
    chip_w = text_width("not applicable", scale) + 16
    lines = [
        (
            f"{check.get('id', '?')}  {check.get('subject') or ''}  {check.get('detail') or ''}"
        ).rstrip()
        for check in checks
    ]
    detail_w = max((text_width(line, scale) for line in lines), default=200)
    width = 24 + chip_w + 24 + detail_w + 16
    row_h = text_height(scale) + 14
    height = 70 + max(1, len(checks)) * row_h + 16
    canvas = Canvas(min(width, MAX_DIM), min(height, MAX_DIM))
    banner = GREEN if verdict == "pass" else RED
    canvas.fill_rect(0, 0, canvas.width, 48, banner)
    canvas.text(
        16,
        14,
        f"{report.get('design', '?')} gate report: {verdict.upper()} ({report.get('scope', '?')})",
        WHITE,
        scale,
    )
    chips = {"pass": GREEN, "fail": RED, "not_applicable": GREY}
    for row, (check, line) in enumerate(zip(checks, lines, strict=True)):
        y = 70 + row * row_h
        status = str(check.get("status", "fail"))
        canvas.fill_rect(12, y, chip_w, text_height(scale) + 8, chips.get(status, GREY))
        canvas.text(20, y + 4, status.replace("_", " "), WHITE, scale)
        canvas.text(12 + chip_w + 24, y + 4, line, RED if status == "fail" else BLACK, scale)
    if not checks:
        canvas.text(16, 70, "no checks", DARK_GREY, scale)
    return canvas


RenderView = Literal["pinmap", "utilization", "timing", "floorplan", "waveform", "report"]


def placed_cells(text: str, top: str) -> dict[str, Any]:
    """Parse the placed netlist; degrade to regex when quotes break JSON."""
    try:
        return cast(dict[str, Any], json.loads(text))
    except json.JSONDecodeError:
        pass
    # nextpnr writes attributes verbatim, so quotes inside HDL attributes
    # can make the JSON unparseable; recover each cell's type and BEL.
    types = [(m.start(), m.group(1)) for m in re.finditer(r'"type":\s*"([^"]+)"', text)]
    cells: dict[str, Any] = {}
    for i, match in enumerate(re.finditer(r'"NEXTPNR_BEL":\s*"([^"]+)"', text)):
        cell_type = ""
        for pos, value in reversed(types):
            if pos < match.start():
                cell_type = value
                break
        cells[f"cell{i}"] = {
            "type": cell_type,
            "attributes": {"NEXTPNR_BEL": match.group(1)},
        }
    return {"modules": {top: {"cells": cells}}}


def render_view_pngs(
    contract: FpgaContract,
    profile: DeviceProfile,
    out_dir: Path,
    *,
    build_dir: Path,
    views: list[str] | None = None,
    gate_report_path: Path | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Render requested views from existing artifacts; never raises."""
    wanted = set(views or ["all"])
    rendered: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    def want(view: str) -> bool:
        return "all" in wanted or view in wanted

    def emit(view: str, path: Path, canvas: Canvas) -> None:
        info = write_png(canvas, path)
        rendered.append({"view": view, **info, "path": str(path)})

    def attempt(view: str, fn: Any, *args: Any) -> None:
        try:
            fn(*args)
        except Exception as exc:  # renders never fail the run
            skipped.append({"view": view, "reason": str(exc)})

    if want("pinmap"):
        attempt(
            "pinmap",
            lambda: emit(
                "pinmap",
                out_dir / f"{contract.name}.fpga-pinmap.png",
                pinmap_canvas(contract, profile),
            ),
        )
    report_json = build_dir / f"{contract.name}.nextpnr-report.json"
    if want("utilization"):

        def _util() -> None:
            report = json.loads(report_json.read_text(encoding="utf-8"))
            emit(
                "utilization",
                out_dir / f"{contract.name}.fpga-utilization.png",
                utilization_canvas(contract, profile, cast(dict[str, Any], report)),
            )

        if report_json.is_file():
            attempt("utilization", _util)
        else:
            skipped.append({"view": "utilization", "reason": "no nextpnr report"})
    if want("timing"):

        def _timing() -> None:
            report = json.loads(report_json.read_text(encoding="utf-8"))
            emit(
                "timing",
                out_dir / f"{contract.name}.fpga-timing.png",
                timing_canvas(contract, cast(dict[str, Any], report)),
            )

        if report_json.is_file():
            attempt("timing", _timing)
        else:
            skipped.append({"view": "timing", "reason": "no nextpnr report"})
    placed = build_dir / f"{contract.name}.placed.json"
    if not placed.is_file():
        placed = build_dir / f"{contract.name}.pnr.json"
    if want("floorplan"):

        def _floorplan() -> None:
            data = placed_cells(placed.read_text(encoding="utf-8"), contract.top)
            emit(
                "floorplan",
                out_dir / f"{contract.name}.fpga-floorplan.png",
                floorplan_canvas(data, profile, contract.top),
            )

        if placed.is_file():
            attempt("floorplan", _floorplan)
        else:
            skipped.append({"view": "floorplan", "reason": "no placed netlist"})
    if want("waveform"):
        vcds = sorted(out_dir.glob("sim-*.vcd"))
        if not vcds:
            skipped.append({"view": "waveform", "reason": "no sim-*.vcd captures"})
        for vcd in vcds:
            sim_id = vcd.stem.removeprefix("sim-")
            attempt(
                f"waveform:{sim_id}",
                lambda vcd=vcd, sim_id=sim_id: emit(
                    "waveform",
                    out_dir / f"{contract.name}.fpga-wave-{sim_id}.png",
                    waveform_canvas(vcd, f"{contract.name} wave {sim_id}"),
                ),
            )
    report_path = gate_report_path or out_dir / f"{contract.name}.fpga-report.json"
    if want("report"):

        def _report() -> None:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            emit(
                "report",
                out_dir / f"{contract.name}.fpga-report.png",
                report_canvas(cast(dict[str, Any], data)),
            )

        if report_path.is_file():
            attempt("report", _report)
        else:
            skipped.append({"view": "report", "reason": "no gate report"})
    return rendered, skipped


def skipped_note(view: str, reason: str) -> str:
    return f"render {view} skipped: {reason}"


def related_check(view: str) -> str | None:
    """Gate check a failed/skipped render should be noted on."""
    base = view.split(":", 1)[0]
    return {
        "pinmap": "fpga.pins",
        "utilization": "fpga.utilization",
        "timing": "fpga.timing",
        "floorplan": "fpga.pnr",
        "waveform": f"fpga.sim.{view.split(':', 1)[1]}" if ":" in view else "fpga.sim",
        "report": None,
    }.get(base)


__all__ = [
    "Canvas",
    "Check",
    "VcdSignal",
    "floorplan_canvas",
    "parse_vcd",
    "pinmap_canvas",
    "placed_cells",
    "related_check",
    "render_view_pngs",
    "report_canvas",
    "skipped_note",
    "text_height",
    "text_width",
    "timing_canvas",
    "utilization_canvas",
    "waveform_canvas",
    "write_png",
]
