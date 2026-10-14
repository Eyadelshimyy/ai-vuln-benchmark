"""Round-trip a reviewer-corrected baseline back to element-keyed corrections.

:mod:`womblex.process.renderer` projects a segment's element run *forward*
into the markdown a human reads and corrects. This module is the reverse:
given the original element run and the (possibly edited) markdown, it
attributes each corrected block back to the element it was rendered from,
so ground truth is stored keyed to element identity — not to a rendering
that reading-order and formatting changes regenerate.

The two directions share one coordinate: the renderer separates every
contributing element with exactly one blank line
(:data:`~womblex.process.chunker.NARRATIVE_JOIN`), the only element
delimiter in its closed structural set, so

    ``render_elements(run) == NARRATIVE_JOIN.join(one block per contributing element)``

holds for any run whose element blocks carry no blank line of their own —
which the renderer's design already assumes (tables join rows with a
single newline, forms join fields the same way, and a narrative element
that carried a blank line would give a reviewer no way to see the element
boundary either). :func:`split_rendered` therefore recovers the blocks by
splitting on that delimiter and pairing them with the contributing
elements in output order. A reviewer who changes the *number* of blocks —
merging two paragraphs or splitting one — has changed element structure,
which an element-keyed store cannot represent without re-segmentation, so
that is a :class:`RoundTripError` rather than a silent re-attribution.

:func:`apply_corrections` returns corrected copies of the elements: a
narrative element carries its corrected text, a ``table`` its corrected
cells (parsed back from the GFM grid, so a later table-markdown format
re-renders correctly), a ``form`` its corrected fields. Re-rendering is
then the unchanged :func:`~womblex.process.renderer.render_elements` over
those copies in whatever order is wanted — each element still carries the
text stored against its identity, so a changed element order moves the
blocks without reassigning their text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from womblex.ingest.elements import TEXT_KINDS, Cell, Element, FieldEntry
from womblex.process.chunker import NARRATIVE_JOIN, element_spans
from womblex.process.segmenter import FIELD_JOIN, element_text

__all__ = ["RoundTripError", "apply_corrections", "split_rendered"]

_STRUCTURAL_KINDS = ("table", "form")
# A GFM header-separator cell: dashes, optional alignment colons.
_SEPARATOR_CELL = re.compile(r"\A:?-+:?\Z")


class RoundTripError(ValueError):
    """The markdown cannot be attributed to the element run block-for-block."""


def split_rendered(
    elements: Sequence[Element], markdown: str
) -> list[tuple[Element, str]]:
    """Pair each contributing element with its rendered block, in output order.

    ``elements`` is the run the baseline was rendered from; ``markdown`` is
    that baseline, edited or not. The contributing elements are exactly the
    ones :func:`~womblex.process.renderer.render_elements` emits a block for
    — narrative-bearing elements with text, plus ``table`` / ``form``
    elements that project to something — in the same order. Splitting the
    markdown on :data:`NARRATIVE_JOIN` must yield one block per contributing
    element; a different count is a :class:`RoundTripError`.
    """
    ordered = sorted(elements, key=lambda e: e.order)
    slots = _contributing(ordered)
    blocks = markdown.split(NARRATIVE_JOIN) if markdown else []
    if len(blocks) != len(slots):
        raise RoundTripError(
            f"markdown split into {len(blocks)} block(s) but the element run "
            f"renders {len(slots)}; a reviewer changed the element structure "
            "(merged or split blocks), which needs re-segmentation, not round-trip"
        )
    return list(zip(slots, blocks))


def apply_corrections(elements: Sequence[Element], markdown: str) -> list[Element]:
    """Return the element run with each contributing element's content corrected.

    Contributing elements are replaced with copies carrying the reviewer's
    text (narrative), cells (``table``) or fields (``form``); every other
    element is carried through unchanged so the run still tiles its document
    and re-renders in element order. Identity — ``order``, ``page``,
    ``bbox`` — is preserved, since that is the key the correction is stored
    against.
    """
    corrected = {elem.order: _correct(elem, block) for elem, block in split_rendered(elements, markdown)}
    return [corrected.get(e.order, e) for e in sorted(elements, key=lambda e: e.order)]


def _contributing(ordered: list[Element]) -> list[Element]:
    """The elements that render a block, in output order.

    Mirrors :func:`~womblex.process.renderer.rendered_order`'s predicates —
    :func:`element_spans` decides narrative contribution and
    :func:`element_text` decides a table's or form's — so the two cannot
    disagree about what was rendered.
    """
    narrative_orders = {order for order, _start, _end in element_spans(ordered)}
    out: list[Element] = []
    for e in ordered:
        if e.order in narrative_orders or e.kind in _STRUCTURAL_KINDS and element_text(e).strip():
            out.append(e)
    return out


def _correct(elem: Element, block: str) -> Element:
    """A copy of *elem* carrying the corrected content parsed from *block*."""
    if elem.kind in TEXT_KINDS:
        return replace(elem, text=block)
    if elem.kind == "table":
        headers, rows = _parse_grid(block)
        cells = _carry_boxes(_grid_to_cells(headers, rows), elem.cells or [])
        return replace(elem, cells=cells, header_rows=[0] if headers else [])
    if elem.kind == "form":
        return replace(elem, fields=_parse_fields(block, elem.fields or []))
    return elem  # unreachable: a slot is always narrative, table or form


def _parse_grid(block: str) -> tuple[list[str], list[list[str]]]:
    """Split a GFM table block back into ``(headers, rows)``.

    The inverse of :func:`~womblex.process.chunker.table_to_markdown`, which
    always emits the header on row 0 and the dash separator on row 1: so row
    0 is headers, row 1 is discarded and everything below is data. Checked at
    that fixed position rather than searched for, so a header cell that is
    itself all dashes is not mistaken for the separator. Cells are ``|``-split
    and stripped, matching the renderer's un-escaped, space-padded output; a
    block missing the separator row (a degenerate edit) is all data.
    """
    grid = [_split_row(line) for line in block.split("\n") if line.strip()]
    if len(grid) >= 2 and _is_separator(grid[1]):
        return grid[0], grid[2:]
    return [], grid


def _split_row(line: str) -> list[str]:
    inner = line.strip()
    inner = inner.removeprefix("|")
    inner = inner.removesuffix("|")
    return [cell.strip() for cell in inner.split("|")]


def _is_separator(row: list[str]) -> bool:
    return bool(row) and all(_SEPARATOR_CELL.match(cell) for cell in row)


def _grid_to_cells(headers: list[str], rows: list[list[str]]) -> list[Cell]:
    """Dense ``Cell`` list, header on row 0 — as :func:`table_to_element` builds it."""
    cells = [Cell(row=0, col=col, value=value) for col, value in enumerate(headers)]
    start = 1 if headers else 0
    for r, row in enumerate(rows, start=start):
        cells.extend(Cell(row=r, col=col, value=value) for col, value in enumerate(row))
    return cells


def _carry_boxes(cells: list[Cell], original: list[Cell]) -> list[Cell]:
    """Carry each original cell's ``bbox`` onto the corrected cell at its (row, col).

    Only when the corrected grid keeps the original's row and column count: a
    reviewer who adds or drops a row or column shifts positions, and a box
    carried by position would then locate the wrong value.
    """
    boxes = {(c.row, c.col): c.bbox for c in original if c.bbox is not None}
    if not boxes or _grid_shape(cells) != _grid_shape(original):
        return cells
    return [replace(c, bbox=boxes.get((c.row, c.col))) for c in cells]


def _grid_shape(cells: list[Cell]) -> tuple[int, int]:
    return (max((c.row for c in cells), default=-1) + 1, max((c.col for c in cells), default=-1) + 1)


def _parse_fields(block: str, original: list[FieldEntry]) -> list[FieldEntry]:
    """Split ``label: value`` lines back into fields.

    Each line is partitioned on the first ``": "`` — the separator the
    renderer emits — so a value carrying a colon survives whole. ``field_type``
    is not rendered, so it is carried from the field at the same position.
    """
    out: list[FieldEntry] = []
    for i, line in enumerate(block.split(FIELD_JOIN)):
        name, sep, value = line.partition(": ")
        if not sep:
            name, value = line, ""
        field_type = original[i].field_type if i < len(original) else "text"
        out.append(FieldEntry(name=name, value=value, field_type=field_type))
    return out
