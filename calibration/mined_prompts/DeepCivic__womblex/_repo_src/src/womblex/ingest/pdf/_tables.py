"""The pdfium table finder: an adapter onto pdfplumber's `TableFinder`.

pdfplumber finds tables from two inputs it normally reads through pdfminer: ruled
edges and character boxes. This module feeds it both from pdfium instead, so a
page is parsed once and pdfminer's layout pass never runs — `find_tables` is
called on every page by `page_profile`, so time per page is the constraint.

`read_polylines` is the only part that touches pdfium; everything after it is
pure (`Char`s and point lists in, `FoundTable`s out). Both inputs are built
lazily: the text strategy never reads edges, and a page with no ruled edges
never builds the character table the cell extraction needs.

Coordinates are the seam's (top-left origin, y down), which are also
pdfplumber's, so nothing is converted on the way in or out.
"""

from __future__ import annotations

import ctypes
from dataclasses import replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import pypdfium2.raw as pdfium_c  # type: ignore[import-untyped]

from womblex.ingest.pdf.types import FoundTable, Rect

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    import pypdfium2 as pdfium  # type: ignore[import-untyped]

    from womblex.ingest.pdf._text import Char
    from womblex.ingest.pdf.types import Matrix, TableStrategy

Polyline = list[tuple[float, float]]
Edge = dict[str, Any]

#: Two coordinates this close are one: a matrix leaves float noise on an edge
#: the producer drew exactly axis-aligned.
_AXIS_TOLERANCE = 1e-3


def read_polylines(
    paths: Iterable[tuple[pdfium.PdfObject, Matrix]], origin: tuple[float, float],
) -> Iterator[Polyline]:
    """The straight runs of each path object, in top-left page space.

    *paths* pairs a path object with the transform from its own space to
    pdfium's user space (its matrix composed with its containers'); *origin* is
    the crop box's top-left corner in that space. A curve ends a run rather than
    being approximated: a table's rules are straight. A close returns to the
    subpath's start, which a curve may have taken out of the run.
    """
    left, top = origin
    x, y = ctypes.c_float(0), ctypes.c_float(0)
    for obj, (a, b, c, d, e, f) in paths:
        run: Polyline = []
        start: tuple[float, float] | None = None
        for i in range(pdfium_c.FPDFPath_CountSegments(obj)):
            segment = pdfium_c.FPDFPath_GetPathSegment(obj, i)
            if not segment or not pdfium_c.FPDFPathSegment_GetPoint(segment, ctypes.byref(x), ctypes.byref(y)):
                continue
            point = (a * x.value + c * y.value + e - left, top - (b * x.value + d * y.value + f))
            kind = pdfium_c.FPDFPathSegment_GetType(segment)
            if kind == pdfium_c.FPDF_SEGMENT_MOVETO:
                if len(run) > 1:
                    yield run
                run, start = [point], point
            elif kind == pdfium_c.FPDF_SEGMENT_LINETO:
                run.append(point)
            else:
                if len(run) > 1:
                    yield run
                run = [point]
            if pdfium_c.FPDFPathSegment_GetClose(segment) and start is not None:
                if run[-1] != start:
                    run.append(start)
                if len(run) > 1:
                    yield run
                run = [start]
        if len(run) > 1:
            yield run


def edges_from_polylines(polylines: Iterable[Polyline]) -> list[Edge]:
    """pdfplumber edges for every horizontal or vertical step of each run."""
    edges: list[Edge] = []
    for run in polylines:
        for (x0, y0), (x1, y1) in pairwise(run):
            if abs(y0 - y1) <= _AXIS_TOLERANCE:
                y = (y0 + y1) / 2
                left, right = sorted((x0, x1))
                edges.append({
                    "x0": left, "x1": right, "top": y, "bottom": y, "width": right - left, "height": 0.0,
                    "orientation": "h", "object_type": "line",
                })
            elif abs(x0 - x1) <= _AXIS_TOLERANCE:
                x = (x0 + x1) / 2
                top, bottom = sorted((y0, y1))
                edges.append({
                    "x0": x, "x1": x, "top": top, "bottom": bottom, "width": 0.0, "height": bottom - top,
                    "orientation": "v", "object_type": "line",
                })
    return edges


def rotate_chars(chars: Iterable[Char], matrix: Matrix) -> list[Char]:
    """*chars* carried onto the rotated page: box through *matrix*, direction through its linear part."""
    a, b, c, d, _, _ = matrix
    return [
        replace(ch, box=ch.box.transform(matrix), direction=(
            round(a * ch.direction[0] + c * ch.direction[1], 2) + 0.0,
            round(b * ch.direction[0] + d * ch.direction[1], 2) + 0.0,
        ))
        for ch in chars
    ]


def rotate_polylines(polylines: Iterable[Polyline], matrix: Matrix) -> Iterator[Polyline]:
    a, b, c, d, e, f = matrix
    for run in polylines:
        yield [(a * x + c * y + e, b * x + d * y + f) for x, y in run]


def char_dicts(chars: Iterable[Char]) -> list[dict[str, Any]]:
    """The keys pdfplumber's word and cell extraction read from a character."""
    return [
        {
            "text": ch.text, "x0": ch.box.x0, "x1": ch.box.x1, "top": ch.box.y0, "bottom": ch.box.y1,
            "doctop": ch.box.y0, "width": ch.box.width, "height": ch.box.height, "size": ch.size,
            "fontname": ch.font, "upright": abs(ch.direction[1]) < 0.01,
        }
        for ch in chars
    ]


class _PlumberPage:
    """What `TableFinder` and `Table.extract` ask of a pdfplumber page: a bbox,
    edges, characters and words. Edges and characters are built on first use."""

    def __init__(
        self, bbox: Rect, chars: Callable[[], list[Char]], polylines: Callable[[], Iterable[Polyline]],
    ) -> None:
        self.bbox = bbox.as_tuple()
        self._chars = chars
        self._polylines = polylines
        self._char_dicts: list[dict[str, Any]] | None = None
        self._edges: list[Edge] | None = None

    @property
    def chars(self) -> list[dict[str, Any]]:
        if self._char_dicts is None:
            self._char_dicts = char_dicts(self._chars())
        return self._char_dicts

    @property
    def edges(self) -> list[Edge]:
        if self._edges is None:
            self._edges = edges_from_polylines(self._polylines())
        return self._edges

    def extract_words(self, **settings: Any) -> list[dict[str, Any]]:
        from pdfplumber.utils import extract_words

        return list(extract_words(self.chars, **settings))


def find_tables(
    strategy: TableStrategy,
    bbox: Rect,
    chars: Callable[[], list[Char]],
    polylines: Callable[[], Iterable[Polyline]],
    rotation: Matrix | None = None,
) -> list[FoundTable]:
    """Tables on a page of size *bbox*, by "lines" (ruled edges) or "text"
    (alignment). pdfplumber's defaults throughout — the engine is the feature.

    *rotation* carries the unrotated inputs onto the displayed page, where *bbox*
    lives: MuPDF finds tables in that frame, so a landscape table on a rotated
    portrait page reads as horizontal text rather than one column per glyph row."""
    from pdfplumber.table import TableFinder

    if rotation is not None:
        chars_unrotated, polylines_unrotated = chars, polylines
        chars = lambda: rotate_chars(chars_unrotated(), rotation)
        polylines = lambda: rotate_polylines(polylines_unrotated(), rotation)
    page = _PlumberPage(bbox, chars, polylines)
    settings = {"vertical_strategy": strategy, "horizontal_strategy": strategy}
    out: list[FoundTable] = []
    for table in TableFinder(page, settings).tables:  # type: ignore[arg-type]
        rows = tuple(tuple(None if cell is None else str(cell) for cell in row) for row in table.extract())
        out.append(FoundTable(
            bbox=Rect.of(table.bbox),
            row_count=len(table.rows),
            col_count=len(table.columns),
            rows=rows,
            cell_boxes=tuple(
                tuple(None if box is None else Rect.of(box) for box in row.cells)
                for row in table.rows
            ),
        ))
    return out
