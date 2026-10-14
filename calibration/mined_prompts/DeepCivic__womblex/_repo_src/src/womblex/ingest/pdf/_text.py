"""The pdfium text engine: characters in content order, rebuilt into the seam's shapes.

pdfium reports a page's text as characters with boxes and fonts, not as lines or
blocks, so this module does the segmentation MuPDF does natively. Phase 0
(`docs/decisions.md`) chose this over pdfminer's layout analysis on speed. The
segmentation here (`segment`) is pure — it takes `Char`s — so it is tested
without a PDF; `read_chars` is the only part that touches pdfium.

Reading order is content-stream order, as in MuPDF; a column jump starts a new
block but no reordering is attempted, so multi-column pages differ from MuPDF
exactly where Phase 0 measured the divergence tail.

Dehyphenation follows MuPDF's measured behaviour, which is to join nothing: the
locked MuPDF (1.27.2) keeps every line-end hyphen and its break under
`TEXT_DEHYPHENATE`, on synthetic pages and in the vendored corpus alike, so the
`dehyphenate` flag is accepted and changes nothing here either.

Font size is the effective size, the font's size scaled by the character's
matrix as MuPDF reports it: pdfium's own figure is the `Tf` operand, which is 1
wherever a producer scales text through the text matrix.
"""

from __future__ import annotations

import ctypes
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pypdfium2.raw as pdfium_c  # type: ignore[import-untyped]

from womblex.ingest.pdf.types import Block, Line, Rect, Span, Word

if TYPE_CHECKING:
    from collections.abc import Callable

    import pypdfium2 as pdfium  # type: ignore[import-untyped]

#: PDF font descriptor flag ForceBold, and the weight from which a font is bold.
_FORCE_BOLD = 1 << 18
_BOLD_WEIGHT = 600
_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")
_LTR = (1.0, 0.0)


@dataclass(frozen=True)
class Char:
    """One character, its box in top-left page space, its font, and its writing
    direction there as a unit vector (left to right is ``(1, 0)``)."""

    text: str
    box: Rect
    size: float
    font: str
    bold: bool
    direction: tuple[float, float] = _LTR
    #: The character's origin in page space; ``None`` falls back to its box's bottom-left.
    origin: tuple[float, float] | None = None


@dataclass
class _Line:
    chars: list[Char]

    @property
    def box(self) -> Rect:
        return _union([c.box for c in self.chars])

    @property
    def text(self) -> str:
        return "".join(c.text for c in self.chars)


def _union(boxes: list[Rect]) -> Rect:
    return Rect(
        min(b.x0 for b in boxes), min(b.y0 for b in boxes),
        max(b.x1 for b in boxes), max(b.y1 for b in boxes),
    )


def _is_bold(weight: int, flags: int, font: str) -> bool:
    return weight >= _BOLD_WEIGHT or bool(flags & _FORCE_BOLD) or "bold" in font.lower()


def read_chars(
    page: pdfium.PdfPage, to_rect: Callable[[tuple[float, float, float, float]], Rect], clip: Rect,
    text_boxes: Callable[[pdfium.PdfTextPage], list[tuple[Rect, str]]] | None = None,
) -> list[Char]:
    """Every printable character of *page* that touches *clip*, in content order.

    *to_rect* turns a pdfium user-space box into top-left page space, where
    *clip* is the page: MuPDF drops each character lying off it. Control
    characters (pdfium's generated line breaks) are dropped: lines are rebuilt
    from geometry. A surrogate pair is one character, boxed by its first unit.

    pdfium sorts vertical text by position, where MuPDF keeps content order.
    *text_boxes* (the page's text objects with their text, in content order, read
    only when needed) puts the vertical characters back in the order they were drawn.
    """
    textpage = page.get_textpage()
    try:
        raw = textpage.raw
        out: list[Char] = []
        high = 0
        name = ctypes.create_string_buffer(256)
        flags = ctypes.c_int(0)
        matrix = pdfium_c.FS_MATRIX()
        ox, oy = ctypes.c_double(), ctypes.c_double()
        for i in range(textpage.count_chars()):
            code = pdfium_c.FPDFText_GetUnicode(raw, i)
            if 0xD800 <= code < 0xDC00:
                high = code
                continue
            if 0xDC00 <= code < 0xE000 and high:
                text = chr(0x10000 + ((high - 0xD800) << 10) + (code - 0xDC00))
            else:
                text = chr(code)
            high = 0
            if text == "\x02":
                # pdfium's stand-in for a hyphen at a line end.
                text = "-"
            if text in "\r\n\x00￾" or (ord(text) < 0x20 and text != "\t"):
                continue
            box = to_rect(textpage.get_charbox(i, loose=True))
            if not (box.x1 > clip.x0 and box.x0 < clip.x1 and box.y1 > clip.y0 and box.y0 < clip.y1):
                continue
            origin: tuple[float, float] | None = None
            if pdfium_c.FPDFText_GetCharOrigin(raw, i, ctypes.byref(ox), ctypes.byref(oy)):
                point = to_rect((ox.value, oy.value, ox.value, oy.value))
                origin = (point.x0, point.y0)
            length = pdfium_c.FPDFText_GetFontInfo(raw, i, name, len(name), ctypes.byref(flags))
            font = _SUBSET_PREFIX.sub("", name.value.decode("latin-1")) if length > 0 else ""
            weight = pdfium_c.FPDFText_GetFontWeight(raw, i)
            pdfium_c.FPDFText_GetMatrix(raw, i, ctypes.byref(matrix))
            scale = math.sqrt(abs(matrix.a * matrix.d - matrix.b * matrix.c))
            # User space is y-up, so the direction's y flips into page space.
            norm = math.hypot(matrix.a, matrix.b) or 1.0
            direction = (round(matrix.a / norm, 2) + 0.0, round(-matrix.b / norm, 2) + 0.0)
            out.append(Char(
                text=" " if text == "\t" else text,
                box=box,
                size=float(pdfium_c.FPDFText_GetFontSize(raw, i)) * scale,
                font=font,
                bold=_is_bold(weight, flags.value, font),
                direction=direction,
                origin=origin,
            ))
        if text_boxes is not None:
            _restore_content_order(out, lambda: text_boxes(textpage))
        return out
    finally:
        textpage.close()


def _restore_content_order(chars: list[Char], text_boxes: Callable[[], list[tuple[Rect, str]]]) -> None:
    """Reorder the vertical characters in place by the text object that drew them.

    Characters keep their slots among the horizontal ones; within the vertical
    slots they sort by the text object, in content order, whose text has the
    character and whose smallest box holds all of it (so text overprinting another's
    box keeps its own object). One that no object holds takes the key of the
    character before it.
    """
    slots = [i for i, c in enumerate(chars) if c.direction[0] == 0.0]
    if len(slots) < 2:
        return
    objects = text_boxes()
    keys: list[int] = []
    last = 0
    for i in slots:
        char = chars[i]
        slack = 0.3 * char.size
        best = -1.0
        for k, (box, content) in enumerate(objects):
            if char.text in content and (
                box.x0 - slack <= char.box.x0 and char.box.x1 <= box.x1 + slack
                and box.y0 - slack <= char.box.y0 and char.box.y1 <= box.y1 + slack
            ):
                area = box.width * box.height
                if best < 0 or area < best:
                    best, last = area, k
        keys.append(last)
    ordered = [chars[i] for _, i in sorted(zip(keys, slots, strict=True), key=lambda kv: kv[0])]
    for i, char in zip(slots, ordered, strict=True):
        chars[i] = char


def _extent(box: Rect, axis: tuple[float, float]) -> tuple[float, float]:
    """The interval *box* covers along *axis*."""
    ends = [x * axis[0] + y * axis[1] for x in (box.x0, box.x1) for y in (box.y0, box.y1)]
    return min(ends), max(ends)


def _on_row(line: _Line, char: Char) -> bool:
    """The character's box shares some of the line's row, measured across its direction."""
    along = line.chars[-1].direction
    across = (-along[1], along[0])
    (a0, a1), (b0, b1) = _extent(line.box, across), _extent(char.box, across)
    return min(a1, b1) >= max(a0, b0)


def _same_line(line: _Line, char: Char) -> bool:
    """The character sits on the line's row, abutting it rather than across a gutter.

    Row and gap are measured in the writing direction, so rotated text forms lines too.
    """
    prev = line.chars[-1]
    if prev.direction != char.direction:
        return False
    along = char.direction
    across = (-along[1], along[0])
    (a0, a1), (b0, b1) = _extent(prev.box, across), _extent(char.box, across)
    overlap = min(a1, b1) - max(a0, b0)
    end, start = _extent(prev.box, along)[1], _extent(char.box, along)[0]
    return overlap > 0.5 * min(a1 - a0, b1 - b0) and end - char.size <= start <= end + _MAX_GAP * char.size


#: How many earlier lines a stray fragment may rejoin; content order seldom
#: strays further than the previous few text objects.
_REJOIN_WINDOW = 8
#: A gap wider than this many font sizes is a column gutter, not a word gap.
_MAX_GAP = 1.5


def _join(left: _Line, right: _Line) -> _Line:
    """*right*, which sits beyond *left*'s end, appended with a space if none shows."""
    a, b = left.chars[-1], right.chars[0]
    chars = left.chars + right.chars
    if a.text != " " and b.text != " " and b.box.x0 - a.box.x1 > 0.2 * a.size:
        gap = Rect(a.box.x1, a.box.y0, b.box.x0, a.box.y1)
        chars = [*left.chars, Char(" ", gap, a.size, a.font, a.bold), *right.chars]
    return _Line(chars)


def _rejoin(lines: list[_Line], boxes: list[Rect], line: _Line, box: Rect) -> bool:
    """Merge a fragment into the recent line on its row that it abuts.

    *boxes* holds each line's box, kept in step so no line is re-measured.
    """
    size = line.chars[0].size
    if line.chars[0].direction != _LTR:
        return False
    for i in range(len(lines) - 1, max(len(lines) - 1 - _REJOIN_WINDOW, -1), -1):
        if lines[i].chars[0].direction != _LTR:
            continue
        other = boxes[i]
        overlap = min(box.y1, other.y1) - max(box.y0, other.y0)
        if overlap <= 0.5 * min(box.height, other.height):
            continue
        if -0.3 * size <= box.x0 - other.x1 <= _MAX_GAP * size:
            lines[i] = _join(lines[i], line)
        elif -0.3 * size <= other.x0 - box.x1 <= _MAX_GAP * size:
            lines[i] = _join(line, lines[i])
        else:
            continue
        boxes[i] = _union([other, box])
        return True
    return False


def _lines(chars: list[Char]) -> list[_Line]:
    runs: list[_Line] = []
    pending: Char | None = None  # a space that did not abut: a word gap, or pdfium's stray separator
    for char in chars:
        if runs and _same_line(runs[-1], char) and (pending is None or not _on_row(runs[-1], pending)):
            runs[-1].chars.append(char)
            pending = None  # the row carried on across a space lying off it: pdfium's stray one
            continue
        if pending is not None:
            runs.append(_Line([pending]))
            pending = None
        if runs and _same_line(runs[-1], char):
            runs[-1].chars.append(char)
        elif char.text == " " and runs:
            pending = char
        else:
            runs.append(_Line([char]))
    if pending is not None:
        runs.append(_Line([pending]))
    lines: list[_Line] = []
    boxes: list[Rect] = []
    for run in runs:
        box = run.box
        if not _rejoin(lines, boxes, run, box):
            lines.append(run)
            boxes.append(box)
    for line in lines:
        # Edge spaces are separators between layout runs, not content.
        while line.chars and line.chars[-1].text == " ":
            line.chars.pop()
        while line.chars and line.chars[0].text == " ":
            line.chars.pop(0)
    return [line for line in lines if line.chars]


#: A line whose baseline is more than this many of its own font sizes below the
#: previous one starts a new block. Measured on MuPDF over `fixtures/synthetic/`:
#: 1.47 stays in the block and 1.64 breaks it, whatever the size or weight of the
#: lines either side (a bold run-in label or a larger closing line set tight stays put).
_MAX_PITCH = 1.5
#: Vertical text continues a block through a line this many font sizes *ahead* of
#: the baseline, on the side MuPDF treats as the way back (probed at 7pt: 3.5pt
#: joins, 6pt splits). Up-running text takes it on one side of `_MAX_PITCH`'s
#: window and downward text on the other.
_MAX_AHEAD = 0.5
#: Horizontal text: a line whose origin sits more than this many points right of
#: the previous line's opens a block, whichever way the text runs (MuPDF: 0 joins,
#: 1 splits, at 7pt and 10pt, at 0 and 180 degrees alike).
_INDENT = 0.5
_UP = (0.0, -1.0)
_DOWN = (0.0, 1.0)
_HORIZONTAL = (_LTR, (-1.0, 0.0))


def _origin(char: Char) -> tuple[float, float]:
    return (char.box.x0, char.box.y1) if char.origin is None else char.origin


def _same_block(prev: _Line, line: _Line) -> bool:
    """*line* continues *prev*'s block, as MuPDF measured in the unrotated page frame.

    Horizontal text (either way up): a line sharing the previous line's row, even
    one overprinting it, continues; otherwise it continues within 1.5 sizes of the
    baseline unless its origin lies right of the previous one's. Vertical text has
    no start condition, only a baseline window that favours opposite sides for up
    and down. Oblique text continues only as the next cell of a row.
    """
    first, last = prev.chars[0], line.chars[0]
    if first.direction != last.direction:
        return False
    along = first.direction
    across = (-along[1], along[0])
    (px, py), (qx, qy) = _origin(first), _origin(last)
    pitch = (qx - px) * across[0] + (qy - py) * across[1]
    if along == _UP:
        return -_MAX_PITCH * last.size <= pitch <= _MAX_AHEAD * last.size
    if along == _DOWN:
        return -_MAX_AHEAD * last.size <= pitch <= _MAX_PITCH * last.size
    (a0, a1), (b0, b1) = _extent(prev.box, across), _extent(line.box, across)
    if min(a1, b1) - max(a0, b0) > 0.5 * min(a1 - a0, b1 - b0):
        # The same row: MuPDF keeps a row's fragments, and an overprint, in one block.
        return along in _HORIZONTAL or _extent(line.box, along)[0] >= _extent(prev.box, along)[1]
    return along in _HORIZONTAL and abs(pitch) <= _MAX_PITCH * last.size and qx - px <= _INDENT


def segment(chars: list[Char]) -> list[list[_Line]]:
    """Characters into blocks of lines."""
    blocks: list[list[_Line]] = []
    for line in _lines(chars):
        if blocks and _same_block(blocks[-1][-1], line):
            blocks[-1].append(line)
        else:
            blocks.append([line])
    return blocks


def plain_text(chars: list[Char]) -> str:
    return "".join(f"{line.text}\n" for block in segment(chars) for line in block)


def text_blocks(chars: list[Char]) -> list[Block]:
    return [
        Block(
            bbox=_union([line.box for line in block]), number=number,
            text="".join(f"{line.text}\n" for line in block),
        )
        for number, block in enumerate(segment(chars))
    ]


def _spans(line: _Line) -> tuple[Span, ...]:
    runs: list[list[Char]] = []
    for char in line.chars:
        key = (char.font, char.size, char.bold)
        if runs and (runs[-1][0].font, runs[-1][0].size, runs[-1][0].bold) == key:
            runs[-1].append(char)
        else:
            runs.append([char])
    return tuple(
        Span(
            text="".join(c.text for c in run), bbox=_union([c.box for c in run]),
            size=run[0].size, font=run[0].font, bold=run[0].bold,
        )
        for run in runs
    )


def text_dict(chars: list[Char]) -> list[Block]:
    """Blocks with lines and spans."""
    return [
        Block(
            bbox=_union([line.box for line in block]), number=number,
            lines=tuple(Line(bbox=line.box, spans=_spans(line)) for line in block),
        )
        for number, block in enumerate(segment(chars))
    ]


def words(chars: list[Char]) -> list[Word]:
    out: list[Word] = []
    for block_no, block in enumerate(segment(chars)):
        for line_no, line in enumerate(block):
            word_no = 0
            run: list[Char] = []
            for char in [*line.chars, None]:
                if char is not None and not char.text.isspace():
                    run.append(char)
                    continue
                if run:
                    box = _union([c.box for c in run])
                    out.append(Word(
                        box.x0, box.y0, box.x1, box.y1,
                        "".join(c.text for c in run), block_no, line_no, word_no,
                    ))
                    word_no += 1
                    run = []
    return out
