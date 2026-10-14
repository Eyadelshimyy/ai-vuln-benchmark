"""Segmentation of an element stream into ground-truth units."""

from __future__ import annotations

from itertools import pairwise

import pytest

from womblex.config import SegmentationConfig
from womblex.ingest.elements import Cell, Element, FieldEntry
from womblex.process.segmenter import element_text, segment_elements


def words(texts: list[str]) -> list[int]:
    """Deterministic, offline stand-in for a real tokeniser."""
    return [len(t.split()) for t in texts]


def para(order: int, text: str, page: int | None = 0) -> Element:
    return Element(order=order, kind="paragraph", extractor="test", page=page, text=text)


def table(order: int, rows: int, cols: int, page: int | None = 0) -> Element:
    cells = [
        Cell(row=r, col=c, value=f"r{r}c{c}")
        for r in range(rows)
        for c in range(cols)
    ]
    return Element(
        order=order, kind="table", extractor="test", page=page,
        cells=cells, header_rows=[0],
    )


def cfg(**kwargs: object) -> SegmentationConfig:
    base: dict[str, object] = {"token_budget": 10, "page_ceiling": 100}
    base.update(kwargs)
    return SegmentationConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def test_every_segment_within_budget() -> None:
    elements = [para(i, " ".join(["w"] * 4)) for i in range(10)]
    segments = segment_elements(elements, words, cfg(token_budget=10))

    assert len(segments) > 1
    assert all(s.tokens <= 10 for s in segments)
    assert not any(s.oversize for s in segments)


def test_oversize_element_is_emitted_alone_and_flagged() -> None:
    elements = [
        para(0, "one two"),
        para(1, " ".join(["w"] * 25)),
        para(2, "three four"),
    ]
    segments = segment_elements(elements, words, cfg(token_budget=10))

    big = [s for s in segments if s.oversize]
    assert len(big) == 1
    assert big[0].element_range == (1, 2)
    assert big[0].tokens == 25
    # Its neighbours are not dragged over budget with it.
    assert all(s.tokens <= 10 for s in segments if not s.oversize)


def test_element_exactly_at_budget_is_not_oversize() -> None:
    elements = [para(0, " ".join(["w"] * 10))]
    segments = segment_elements(elements, words, cfg(token_budget=10))

    assert len(segments) == 1
    assert segments[0].tokens == 10
    assert not segments[0].oversize


def test_word_count_recorded_alongside_tokens() -> None:
    elements = [para(0, "alpha beta gamma")]
    segments = segment_elements(elements, lambda ts: [len(t) for t in ts], cfg())

    assert segments[0].tokens == len("alpha beta gamma")  # characters, per count_fn
    assert segments[0].words == 3


# ---------------------------------------------------------------------------
# Structure is never split
# ---------------------------------------------------------------------------


def test_table_and_its_cells_stay_in_one_segment() -> None:
    tbl = table(1, rows=8, cols=4)
    els = [para(0, "intro"), tbl, para(2, "outro")]
    segments = segment_elements(els, words, cfg(token_budget=10))

    holding = [s for s in segments if s.element_range[0] <= 1 < s.element_range[1]]
    assert len(holding) == 1
    # Every cell measured inside that one segment — a table is one element,
    # so no boundary can fall between its cells.
    assert holding[0].tokens == len(element_text(tbl).split())


def test_form_fields_stay_with_their_form() -> None:
    form = Element(
        order=1, kind="form", extractor="test", page=0,
        fields=[FieldEntry(name=f"field {i}", value=f"value {i}") for i in range(20)],
    )
    segments = segment_elements([para(0, "intro"), form], words, cfg(token_budget=10))

    holding = [s for s in segments if s.element_range[0] <= 1 < s.element_range[1]]
    assert len(holding) == 1
    # Over budget as one element, so emitted solo with every field measured.
    assert holding[0].oversize
    assert holding[0].tokens == len(element_text(form).split())


def test_element_text_projects_tables_and_forms() -> None:
    assert "r0c0" in element_text(table(0, rows=2, cols=2))
    form = Element(
        order=0, kind="form", extractor="test",
        fields=[FieldEntry(name="Name", value="Ada")],
    )
    assert element_text(form) == "Name: Ada"
    assert element_text(Element(order=0, kind="page_break", extractor="test", page=1)) == ""


# ---------------------------------------------------------------------------
# Page ceiling
# ---------------------------------------------------------------------------


def test_no_segment_spans_more_than_the_page_ceiling() -> None:
    elements = [para(i, "a", page=i) for i in range(12)]
    segments = segment_elements(elements, words, cfg(token_budget=1000, page_ceiling=3))

    assert len(segments) == 4
    for s in segments:
        assert s.page_range is not None
        assert s.page_range[1] - s.page_range[0] <= 3


def test_page_ceiling_and_budget_bind_independently() -> None:
    elements = [para(i, " ".join(["w"] * 6), page=i // 2) for i in range(8)]
    tight_budget = segment_elements(elements, words, cfg(token_budget=6, page_ceiling=100))
    tight_pages = segment_elements(elements, words, cfg(token_budget=1000, page_ceiling=1))

    assert len(tight_budget) == 8
    assert len(tight_pages) == 4


def test_document_with_no_page_concept_segments_on_the_budget_alone() -> None:
    elements = [para(i, " ".join(["w"] * 6), page=None) for i in range(6)]
    segments = segment_elements(elements, words, cfg(token_budget=12, page_ceiling=1))

    assert len(segments) == 3
    assert all(s.page_range is None for s in segments)


def test_page_range_is_half_open_over_zero_based_pages() -> None:
    elements = [para(0, "a", page=3), para(1, "b", page=4)]
    segments = segment_elements(elements, words, cfg(token_budget=1000))

    assert segments[0].page_range == (3, 5)


def test_page_less_tail_of_a_paged_document_reports_no_page_range() -> None:
    """A spreadsheet-print PDF tails its stream with page-less tables, so a
    None page_range is a fact about the segment, not about the source."""
    elements = [para(0, "a", page=0), para(1, "b", page=0), table(2, 4, 3, page=None)]
    segments = segment_elements(elements, words, cfg(token_budget=4))

    assert segments[0].page_range == (0, 1)
    assert segments[-1].page_range is None
    # The paged elements are still there, so the document plainly has pages.
    assert any(s.page_range is not None for s in segments)


# ---------------------------------------------------------------------------
# Configuration, not constants
# ---------------------------------------------------------------------------


def test_budget_and_ceiling_come_from_config() -> None:
    elements = [para(i, " ".join(["w"] * 5), page=i) for i in range(8)]

    wide = segment_elements(elements, words, cfg(token_budget=40, page_ceiling=8))
    narrow = segment_elements(elements, words, cfg(token_budget=10, page_ceiling=8))

    assert len(wide) == 1
    assert len(narrow) == 4


def test_oversize_error_refuses_to_segment() -> None:
    elements = [para(0, " ".join(["w"] * 50))]

    with pytest.raises(ValueError, match="over the 10-token budget"):
        segment_elements(elements, words, cfg(token_budget=10, oversize="error"))


# ---------------------------------------------------------------------------
# Tiling and determinism
# ---------------------------------------------------------------------------


def test_segments_tile_the_document_without_gaps_or_overlaps() -> None:
    elements = [
        para(0, "intro", page=0),
        table(1, rows=6, cols=3, page=0),
        Element(order=2, kind="page_break", extractor="test", page=1),
        para(3, " ".join(["w"] * 12), page=1),
        para(4, "tail", page=2),
    ]
    segments = segment_elements(elements, words, cfg(token_budget=8, page_ceiling=2))

    assert segments[0].element_range[0] == 0
    assert segments[-1].element_range[1] == 5
    for prev, curr in pairwise(segments):
        assert prev.element_range[1] == curr.element_range[0]


def test_empty_stream_segments_to_nothing() -> None:
    assert segment_elements([], words, cfg()) == []


def test_unordered_stream_is_refused() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        segment_elements([para(1, "a"), para(0, "b")], words, cfg())


def test_filtered_stream_is_refused() -> None:
    """A gap means a subset, whose segments could not tile the document."""
    whole = [para(0, "a"), Element(order=1, kind="page_break", extractor="t", page=1), para(2, "b")]
    filtered = [e for e in whole if e.kind != "page_break"]

    assert len(segment_elements(whole, words, cfg())) == 1
    with pytest.raises(ValueError, match="must be contiguous"):
        segment_elements(filtered, words, cfg())


def test_rerun_produces_identical_boundaries() -> None:
    elements = [para(i, " ".join(["w"] * (i % 5 + 1)), page=i // 3) for i in range(20)]
    config = cfg(token_budget=9, page_ceiling=2)

    assert segment_elements(elements, words, config) == segment_elements(
        elements, words, config
    )

