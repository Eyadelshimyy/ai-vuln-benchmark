"""Round-trip a corrected baseline back to element-keyed corrections (U5b)."""

from __future__ import annotations

import pytest

from womblex.ingest.elements import BBox, Cell, Element, FieldEntry
from womblex.process.ground_truth_roundtrip import (
    RoundTripError,
    apply_corrections,
    split_rendered,
)
from womblex.process.renderer import render_elements


def para(order: int, text: str, page: int | None = 0) -> Element:
    return Element(order=order, kind="paragraph", extractor="test", page=page, text=text)


def table(order: int, page: int | None = 0) -> Element:
    cells = [Cell(row=r, col=c, value=f"r{r}c{c}") for r in range(2) for c in range(2)]
    return Element(
        order=order, kind="table", extractor="test", page=page,
        cells=cells, header_rows=[0],
    )


def form(order: int, fields: list[tuple[str, str]], page: int | None = 0) -> Element:
    return Element(
        order=order, kind="form", extractor="test", page=page,
        fields=[FieldEntry(name=n, value=v) for n, v in fields],
    )


def _cell(element: Element, row: int, col: int) -> str:
    return next(c.value for c in element.cells or [] if c.row == row and c.col == col)


# ---------------------------------------------------------------------------
# Lossless identity: the pristine baseline round-trips unchanged
# ---------------------------------------------------------------------------


def test_identity_roundtrip_reproduces_the_baseline() -> None:
    elements = [
        para(0, "First paragraph."),
        table(1),
        para(2, "Second paragraph."),
        form(3, [("Name", "Jane Roe"), ("Ref", "A/17")]),
    ]
    baseline = render_elements(elements)
    corrected = apply_corrections(elements, baseline)
    assert render_elements(corrected) == baseline


def test_split_pairs_blocks_with_contributing_elements_in_order() -> None:
    elements = [para(0, "alpha"), table(1), para(2, "gamma")]
    pairs = split_rendered(elements, render_elements(elements))
    assert [e.order for e, _block in pairs] == [0, 1, 2]
    assert pairs[0][1] == "alpha"
    assert pairs[2][1] == "gamma"


def test_non_contributing_elements_are_not_slots() -> None:
    # An empty-text paragraph and a page break render nothing, so they are
    # carried through but never consume a block.
    elements = [
        para(0, "kept"),
        para(1, ""),
        Element(order=2, kind="page_break", extractor="test", page=1),
        para(3, "also kept"),
    ]
    baseline = render_elements(elements)
    assert baseline == "kept\n\nalso kept"
    corrected = apply_corrections(elements, baseline)
    assert render_elements(corrected) == baseline
    assert [e.order for e in corrected] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Corrections are keyed to element identity
# ---------------------------------------------------------------------------


def test_narrative_correction_is_stored_against_its_element() -> None:
    elements = [para(0, "typo hear"), para(1, "unchanged")]
    edited = "typo here\n\nunchanged"
    corrected = {e.order: e for e in apply_corrections(elements, edited)}
    assert corrected[0].text == "typo here"
    assert corrected[1].text == "unchanged"


def test_table_cell_correction_keyed_to_parent_and_position() -> None:
    elements = [table(0)]
    baseline = render_elements(elements)  # r0c0 header row, r1c* data
    edited = baseline.replace("r1c1", "corrected")
    corrected = apply_corrections(elements, edited)[0]
    assert _cell(corrected, 1, 1) == "corrected"
    assert _cell(corrected, 1, 0) == "r1c0"  # neighbours untouched
    assert _cell(corrected, 0, 0) == "r0c0"


def test_header_row_of_dashes_is_not_mistaken_for_the_separator() -> None:
    # The separator is at the fixed row-1 position, so a header cell that is
    # itself dashes round-trips as content rather than being discarded.
    elements = [
        Element(
            order=0, kind="table", extractor="test", page=0,
            cells=[
                Cell(row=0, col=0, value="---"), Cell(row=0, col=1, value="note"),
                Cell(row=1, col=0, value="a"), Cell(row=1, col=1, value="b"),
            ],
            header_rows=[0],
        )
    ]
    baseline = render_elements(elements)
    corrected = apply_corrections(elements, baseline)
    assert _cell(corrected[0], 0, 0) == "---"
    assert _cell(corrected[0], 1, 0) == "a"
    assert render_elements(corrected) == baseline


def test_header_text_correction_round_trips() -> None:
    elements = [table(0)]
    edited = render_elements(elements).replace("r0c1", "Corrected Header")
    corrected = apply_corrections(elements, edited)[0]
    assert _cell(corrected, 0, 1) == "Corrected Header"


def test_form_field_value_correction_preserves_type() -> None:
    elements = [
        Element(
            order=0, kind="form", extractor="test", page=0,
            fields=[
                FieldEntry(name="Signed", value="no", field_type="checkbox"),
                FieldEntry(name="Date", value="2020-01-01", field_type="text"),
            ],
        )
    ]
    baseline = render_elements(elements)
    edited = baseline.replace("no", "yes").replace("2020-01-01", "2021-06-30")
    fields = apply_corrections(elements, edited)[0].fields or []
    assert (fields[0].name, fields[0].value, fields[0].field_type) == ("Signed", "yes", "checkbox")
    assert (fields[1].name, fields[1].value, fields[1].field_type) == ("Date", "2021-06-30", "text")


# ---------------------------------------------------------------------------
# Re-rendering: order and table format are regenerable
# ---------------------------------------------------------------------------


def test_rerender_in_changed_order_moves_text_by_identity() -> None:
    elements = [para(0, "one"), para(1, "two"), para(2, "three")]
    corrected = apply_corrections(elements, "ONE\n\ntwo\n\nthree")
    reordered = sorted(corrected, key=lambda e: e.order, reverse=True)
    assert render_elements(reordered) == "three\n\ntwo\n\nONE"


def test_rerender_under_changed_table_format(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stored corrections are structured cells, so a later table-markdown
    # format re-renders correctly with no reviewer intervention.
    elements = [table(0)]
    corrected = apply_corrections(elements, render_elements(elements))

    from womblex.process import segmenter

    def bang_format(headers: list[str], rows: list[list[str]]) -> str:
        lines = ["!" + "!".join(headers) + "!"]
        lines += ["!" + "!".join(r) + "!" for r in rows]
        return "\n".join(lines)

    monkeypatch.setattr(segmenter, "table_to_markdown", bang_format)
    assert render_elements(corrected) == "!r0c0!r0c1!\n!r1c0!r1c1!"


# ---------------------------------------------------------------------------
# A changed block count is refused, not silently re-attributed
# ---------------------------------------------------------------------------


def test_merged_blocks_raise_rather_than_reassign() -> None:
    elements = [para(0, "first"), para(1, "second")]
    with pytest.raises(RoundTripError):
        split_rendered(elements, "first second")  # two elements, one block


def test_extra_block_raises() -> None:
    elements = [para(0, "solo")]
    with pytest.raises(RoundTripError):
        apply_corrections(elements, "solo\n\nspurious")


def test_empty_render_roundtrips_to_no_blocks() -> None:
    elements = [para(0, ""), Element(order=1, kind="page_break", extractor="test", page=1)]
    assert render_elements(elements) == ""
    assert split_rendered(elements, "") == []
    assert apply_corrections(elements, "") == elements


def _boxed_table() -> Element:
    elem = table(0)
    for c in elem.cells or []:
        c.bbox = BBox(x=0.1 * c.col, y=0.1 * c.row, width=0.1, height=0.1)
    return elem


def test_table_correction_keeps_cell_boxes_when_the_grid_shape_holds() -> None:
    elements = [_boxed_table()]
    edited = render_elements(elements).replace("r1c1", "corrected")
    corrected = apply_corrections(elements, edited)[0]
    assert [c.bbox for c in corrected.cells or []] == [c.bbox for c in elements[0].cells or []]


def test_table_correction_drops_cell_boxes_when_a_row_is_added() -> None:
    elements = [_boxed_table()]
    edited = render_elements(elements) + "\n| new | row |"
    corrected = apply_corrections(elements, edited)[0]
    assert len(corrected.cells or []) == 6
    assert all(c.bbox is None for c in corrected.cells or [])
