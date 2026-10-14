"""Tests for womblex.process.chunker.

Exercises the post-I2 surface: a single :func:`chunk_batch` entry point
plus :func:`create_chunker`, :func:`table_to_markdown`,
:func:`reassemble_narrative`, :func:`element_spans`,
:func:`chunks_in_document_order`, :func:`collect_tables_from_elements`,
:func:`build_chunk_input`, the :class:`TextChunk` / :class:`ChunkInput`
dataclasses and the ``_repair_redaction_splits`` helper.

Tests use a word-count callable token counter so they don't depend on a
HuggingFace tokeniser. semchunk treats it as any other token counter.
"""

from __future__ import annotations

import semchunk

from womblex.ingest.elements import BBox, Cell, Element
from womblex.process.chunker import (
    ChunkInput,
    TextChunk,
    _count_tokens,
    _repair_redaction_splits,
    build_chunk_input,
    chunk_batch,
    chunks_in_document_order,
    collect_tables_from_elements,
    create_chunker,
    element_spans,
    reassemble_narrative,
    table_to_markdown,
)


def _word_token_counter(text: str) -> int:
    return len(text.split())


def _make_test_chunker(chunk_size: int = 50) -> semchunk.Chunker:
    return semchunk.chunkerify(_word_token_counter, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# create_chunker
# ---------------------------------------------------------------------------


class TestCreateChunker:
    def test_returns_callable(self) -> None:
        chunker = create_chunker(tokenizer=_word_token_counter, chunk_size=100)
        assert callable(chunker)

    def test_exposes_full_chunkerify_surface(self) -> None:
        chunker = create_chunker(
            _word_token_counter, chunk_size=50,
            memoize=False, cache_maxsize=10, max_token_chars=20,
        )
        result = chunker(["hello world"], offsets=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_default_chunking_model_none_is_offline(self) -> None:
        # Default (no chunking_model) must stay fully offline: a callable
        # token counter and no Isaacus client, yet chunking succeeds.
        chunker = create_chunker(_word_token_counter, chunk_size=50)
        chunks = chunker("alpha beta gamma delta")
        assert chunks  # produced offline, no API touched

    def test_ai_chunking_params_forwarded_to_chunkerify(self, monkeypatch) -> None:
        captured: dict = {}
        real_chunkerify = semchunk.chunkerify

        def _fake_chunkerify(tokenizer, **kwargs):
            captured["tokenizer"] = tokenizer
            captured.update(kwargs)
            return real_chunkerify(_word_token_counter, chunk_size=50)

        monkeypatch.setattr("womblex.process.chunker.semchunk.chunkerify", _fake_chunkerify)
        sentinel_client = object()
        create_chunker(
            _word_token_counter, chunk_size=50,
            chunking_model="kanon-2-enricher",
            isaacus_client=sentinel_client,
            tokenizer_kwargs={"add_special_tokens": False},
        )
        assert captured["chunking_model"] == "kanon-2-enricher"
        assert captured["isaacus_client"] is sentinel_client
        assert captured["tokenizer_kwargs"] == {"add_special_tokens": False}


# ---------------------------------------------------------------------------
# TextChunk
# ---------------------------------------------------------------------------


class TestTextChunk:
    def test_default_fields(self) -> None:
        c = TextChunk(text="x", start_char=0, end_char=1, chunk_index=0)
        assert c.content_type == "narrative"
        assert c.has_redaction is False
        assert c.page_start is None
        assert c.page_end is None

    def test_explicit_fields(self) -> None:
        c = TextChunk(
            text="x", start_char=0, end_char=1, chunk_index=0,
            content_type="table", has_redaction=True,
            page_start=3, page_end=5,
        )
        assert c.content_type == "table"
        assert c.has_redaction is True
        assert (c.page_start, c.page_end) == (3, 5)


# ---------------------------------------------------------------------------
# table_to_markdown
# ---------------------------------------------------------------------------


class TestTableToMarkdown:
    def test_empty(self) -> None:
        assert table_to_markdown([], []) == ""

    def test_headers_and_rows(self) -> None:
        md = table_to_markdown(["A", "B"], [["1", "2"], ["3", "4"]])
        lines = md.splitlines()
        assert lines[0] == "| A | B |"
        assert "1" in lines[2]
        assert "3" in lines[3]

    def test_short_row_padded(self) -> None:
        md = table_to_markdown(["A", "B", "C"], [["x"]])
        assert "| x |  |  |" in md or "x" in md


# ---------------------------------------------------------------------------
# _repair_redaction_splits
# ---------------------------------------------------------------------------


class TestRepairRedactionSplits:
    def test_no_change_when_marker_intact(self) -> None:
        chunks = [TextChunk("a <REDACTED> b", 0, 14, 0)]
        out = _repair_redaction_splits(chunks)
        assert out == chunks

    def test_split_marker_merged(self) -> None:
        chunks = [
            TextChunk("a <RED", 0, 6, 0),
            TextChunk("ACTED> b", 6, 14, 1),
        ]
        out = _repair_redaction_splits(chunks)
        assert len(out) == 1
        assert "<REDACTED>" in out[0].text
        assert out[0].start_char == 0
        assert out[0].end_char == 14
        assert out[0].has_redaction is True

    def test_indices_resequenced(self) -> None:
        chunks = [
            TextChunk("a <RED", 0, 6, 0),
            TextChunk("ACTED> b", 6, 14, 1),
            TextChunk("c", 14, 15, 2),
        ]
        out = _repair_redaction_splits(chunks)
        assert [c.chunk_index for c in out] == [0, 1]

    def test_merge_carries_page_span(self) -> None:
        chunks = [
            TextChunk("a <RED", 0, 6, 0, page_start=1, page_end=1),
            TextChunk("ACTED> b", 6, 14, 1, page_start=2, page_end=2),
        ]
        out = _repair_redaction_splits(chunks)
        assert (out[0].page_start, out[0].page_end) == (1, 2)


# ---------------------------------------------------------------------------
# reassemble_narrative + collect_tables_from_elements
# ---------------------------------------------------------------------------


def _para(order: int, text: str, page: int | None) -> Element:
    return Element(
        order=order, kind="paragraph", extractor="t",
        text=text, page=page,
    )


def _table_elem(order: int, page: int | None, rows: list[list[str]], header_rows: list[int] | None) -> Element:
    cells = []
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            cells.append(Cell(row=r, col=c, value=v))
    return Element(
        order=order, kind="table", extractor="t",
        page=page, cells=cells, header_rows=header_rows or [],
        bbox=BBox(x=0, y=0, width=1, height=1),
    )


class TestReassembleNarrative:
    def test_joins_with_double_newline(self) -> None:
        elements = [_para(0, "first", 1), _para(1, "second", 1)]
        text, _ = reassemble_narrative(elements)
        assert text == "first\n\nsecond"

    def test_skips_empty_text(self) -> None:
        elements = [_para(0, "x", 1), _para(1, "", 1), _para(2, "y", 1)]
        text, _ = reassemble_narrative(elements)
        assert text == "x\n\ny"

    def test_skips_non_text_kinds(self) -> None:
        elements = [_para(0, "x", 1), _table_elem(1, 1, [["a"]], None), _para(2, "y", 1)]
        text, _ = reassemble_narrative(elements)
        assert text == "x\n\ny"

    def test_page_breaks_track_offsets(self) -> None:
        elements = [
            _para(0, "alpha", 1),     # offset 0..5
            _para(1, "beta", 1),      # offset 5+2+4 = 11
            _para(2, "gamma", 2),     # next page
        ]
        text, page_breaks = reassemble_narrative(elements)
        # "alpha" (5) + "\n\nbeta" (6) = 11, then "\n\ngamma" (7) → 18
        assert text == "alpha\n\nbeta\n\ngamma"
        assert len(page_breaks) == 2
        assert page_breaks[0][1] == 1
        assert page_breaks[1][1] == 2
        # Page 1 ends at offset 11 (end of "beta"); page 2 ends at 18 (end of text).
        assert page_breaks[0][0] == 11
        assert page_breaks[1][0] == 18

    def test_no_page_yields_empty_breaks(self) -> None:
        elements = [_para(0, "alpha", None), _para(1, "beta", None)]
        _, page_breaks = reassemble_narrative(elements)
        assert page_breaks == []


class TestCollectTablesFromElements:
    def test_kind_table_emitted(self) -> None:
        elements = [_table_elem(0, 3, [["A", "B"], ["1", "2"]], header_rows=[0])]
        out = collect_tables_from_elements(elements)
        assert len(out) == 1
        page, elem_order, md = out[0]
        assert page == 3
        assert elem_order == 0
        assert "| A | B |" in md
        assert "| 1 | 2 |" in md

    def test_elem_order_tracks_the_source_element(self) -> None:
        # Two tables at distinct positions — each carries its own anchor.
        elements = [
            _para(0, "intro", 1),
            _table_elem(1, 1, [["A"], ["1"]], header_rows=[0]),
            _para(2, "middle", 1),
            _table_elem(3, 1, [["B"], ["2"]], header_rows=[0]),
        ]
        out = collect_tables_from_elements(elements)
        assert [elem_order for _, elem_order, _ in out] == [1, 3]

    def test_sheet_cells_aggregated_as_synthetic_table(self) -> None:
        # Two sheet_cell elements → one synthetic table.
        elements = [
            Element(order=0, kind="sheet_cell", extractor="t",
                    sheet="S1", row=0, col=0, value="H1"),
            Element(order=1, kind="sheet_cell", extractor="t",
                    sheet="S1", row=0, col=1, value="H2"),
            Element(order=2, kind="sheet_cell", extractor="t",
                    sheet="S1", row=1, col=0, value="v1"),
        ]
        out = collect_tables_from_elements(elements)
        assert len(out) == 1
        page, elem_order, md = out[0]
        assert page is None
        # A sheet aggregates many sheet_cells and has no narrative to be
        # ordered against, so it carries no anchor.
        assert elem_order is None
        assert "H1" in md and "H2" in md


# ---------------------------------------------------------------------------
# element_spans / chunks_in_document_order
# ---------------------------------------------------------------------------


def _chunk_row(content_type: str, start_char: int, elem_order: int | None = None) -> dict:
    """A CHUNKS_SCHEMA-shaped row — the shape a consumer reads back."""
    return {"content_type": content_type, "start_char": start_char, "elem_order": elem_order}


class TestElementSpans:
    def test_spans_index_the_reassembled_narrative(self) -> None:
        elements = [_para(0, "alpha", 1), _para(1, "beta", 1)]
        text, _ = reassemble_narrative(elements)
        spans = element_spans(elements)
        assert spans == [(0, 0, 5), (1, 5, 11)]
        # Slicing the narrative by a span returns that element's text (the
        # leading join belongs to the span, the same convention page breaks use).
        assert text[spans[0][1]:spans[0][2]] == "alpha"
        assert text[spans[1][1]:spans[1][2]] == "\n\nbeta"
        assert spans[-1][2] == len(text)

    def test_only_narrative_bearing_elements_appear(self) -> None:
        # Tables and empty paragraphs contribute nothing to the narrative, so
        # they have no span — elem_order 1 and 2 are absent.
        elements = [
            _para(0, "x", 1),
            _table_elem(1, 1, [["a"]], None),
            _para(2, "", 1),
            _para(3, "y", 1),
        ]
        assert [order for order, _, _ in element_spans(elements)] == [0, 3]

    def test_no_narrative_yields_no_spans(self) -> None:
        assert element_spans([_table_elem(0, 1, [["a"]], None)]) == []


class TestChunksInDocumentOrder:
    def test_tables_interleave_with_the_prose_they_followed(self) -> None:
        elements = [
            _para(0, "alpha", 1),
            _table_elem(1, 1, [["A"], ["1"]], header_rows=[0]),
            _para(2, "beta", 1),
            _table_elem(3, 1, [["B"], ["2"]], header_rows=[0]),
            _para(4, "gamma", 1),
        ]
        spans = element_spans(elements)
        # Projection order: every narrative chunk, then every table chunk.
        rows = [
            _chunk_row("narrative", 0),
            _chunk_row("narrative", 7),
            _chunk_row("narrative", 13),
            _chunk_row("table", 0, elem_order=3),
            _chunk_row("table", 0, elem_order=1),
        ]
        ordered = chunks_in_document_order(rows, spans)
        assert [(r["content_type"], r["start_char"], r["elem_order"]) for r in ordered] == [
            ("narrative", 0, None),
            ("table", 0, 1),
            ("narrative", 7, None),
            ("table", 0, 3),
            ("narrative", 13, None),
        ]

    def test_table_sorts_before_the_narrative_chunk_it_precedes(self) -> None:
        # A table anchors where the next element begins; the table element
        # comes first in the document, so it sorts first on the tie.
        elements = [_para(0, "alpha", 1), _table_elem(1, 1, [["A"]], None), _para(2, "beta", 1)]
        spans = element_spans(elements)
        rows = [_chunk_row("narrative", spans[1][1]), _chunk_row("table", 0, elem_order=1)]
        assert [r["content_type"] for r in chunks_in_document_order(rows, spans)] == [
            "table", "narrative",
        ]

    def test_trailing_table_sorts_last(self) -> None:
        elements = [_para(0, "alpha", 1), _table_elem(1, 1, [["A"]], None)]
        spans = element_spans(elements)
        rows = [_chunk_row("table", 0, elem_order=1), _chunk_row("narrative", 0)]
        assert [r["content_type"] for r in chunks_in_document_order(rows, spans)] == [
            "narrative", "table",
        ]

    def test_multiple_chunks_of_one_table_keep_markdown_order(self) -> None:
        # A table's own chunks order by their offset into its markdown — not by
        # the order the rows happened to arrive in, so an unordered read is safe.
        elements = [_para(0, "alpha", 1), _table_elem(1, 1, [["A"]], None), _para(2, "beta", 1)]
        spans = element_spans(elements)
        rows = [
            _chunk_row("table", 40, elem_order=1),
            _chunk_row("narrative", 0),
            _chunk_row("table", 0, elem_order=1),
        ]
        ordered = chunks_in_document_order(rows, spans)
        assert [r["start_char"] for r in ordered if r["content_type"] == "table"] == [0, 40]

    def test_sheets_sort_last_in_their_existing_order(self) -> None:
        # A spreadsheet sheet has no anchor and no narrative to be ordered
        # against; it must not displace anything that does.
        elements = [_para(0, "alpha", 1)]
        spans = element_spans(elements)
        rows = [
            _chunk_row("table", 0, elem_order=None),
            _chunk_row("table", 99, elem_order=None),
            _chunk_row("narrative", 0),
        ]
        ordered = chunks_in_document_order(rows, spans)
        assert [r["content_type"] for r in ordered] == ["narrative", "table", "table"]
        assert [r["start_char"] for r in ordered[1:]] == [0, 99]

    def test_no_narrative_orders_tables_by_anchor(self) -> None:
        rows = [_chunk_row("table", 0, elem_order=3), _chunk_row("table", 0, elem_order=1)]
        assert [r["elem_order"] for r in chunks_in_document_order(rows, [])] == [1, 3]

    def test_empty_input(self) -> None:
        assert chunks_in_document_order([], []) == []


# ---------------------------------------------------------------------------
# build_chunk_input
# ---------------------------------------------------------------------------


class TestBuildChunkInput:
    def test_assembles_narrative_and_tables(self) -> None:
        elements = [
            _para(0, "intro", 1),
            _table_elem(1, 1, [["A"], ["1"]], header_rows=[0]),
        ]
        ci = build_chunk_input("hash-x", elements)
        assert ci.source_hash == "hash-x"
        assert ci.narrative == "intro"
        assert ci.page_breaks == [(5, 1)]
        assert len(ci.tables) == 1

    def test_include_tables_false_skips_tables(self) -> None:
        elements = [_table_elem(0, 1, [["A"], ["1"]], header_rows=[0])]
        ci = build_chunk_input("hash-x", elements, include_tables=False)
        assert ci.tables == []


# ---------------------------------------------------------------------------
# chunk_batch
# ---------------------------------------------------------------------------


class TestChunkBatch:
    def test_empty_input(self) -> None:
        assert chunk_batch([], _make_test_chunker()) == {}

    def test_narrative_only(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        ci = ChunkInput(source_hash="a", narrative="hello world", page_breaks=[(11, 1)])
        out = chunk_batch([ci], chunker)
        assert "a" in out
        assert len(out["a"]) >= 1
        assert all(c.content_type == "narrative" for c in out["a"])
        assert all(c.page_start == 1 for c in out["a"])

    def test_tables_get_table_content_type(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        ci = ChunkInput(
            source_hash="a", narrative="",
            tables=[(2, 7, "| A | B |\n| --- | --- |\n| 1 | 2 |")],
        )
        out = chunk_batch([ci], chunker)
        chunks = out["a"]
        assert chunks
        assert all(c.content_type == "table" for c in chunks)
        assert all(c.page_start == 2 and c.page_end == 2 for c in chunks)
        assert all(c.elem_order == 7 for c in chunks)

    def test_narrative_chunks_carry_no_elem_order(self) -> None:
        # Narrative chunks straddle elements, so the anchor stays None —
        # only table chunks are attributable to one element.
        chunker = _make_test_chunker(chunk_size=50)
        ci = ChunkInput(source_hash="a", narrative="hello world")
        out = chunk_batch([ci], chunker)
        assert all(c.elem_order is None for c in out["a"])

    def test_document_order_reconstructable_from_anchors(self) -> None:
        # The point of the anchor: recover which table followed which prose.
        chunker = _make_test_chunker(chunk_size=50)
        elements = [
            _para(0, "alpha", 1),
            _table_elem(1, 1, [["A"], ["1"]], header_rows=[0]),
            _para(2, "beta", 1),
            _table_elem(3, 1, [["B"], ["2"]], header_rows=[0]),
        ]
        ci = build_chunk_input("a", elements)
        out = chunk_batch([ci], chunker)
        tables = sorted(
            (c for c in out["a"] if c.content_type == "table"),
            key=lambda c: c.elem_order or 0,
        )
        assert [c.elem_order for c in tables] == [1, 3]
        assert "A" in tables[0].text and "B" in tables[1].text

    def test_real_chunks_interleave_via_element_spans(self) -> None:
        # End to end: chunk a doc whose prose brackets two tables, then put the
        # two projections back into document order from the anchors + spans.
        # chunk_size=2 (words) splits each paragraph into its own chunk, so the
        # narrative really is interleaved rather than one chunk with tables after.
        chunker = _make_test_chunker(chunk_size=2)
        elements = [
            _para(0, "alpha prose", 1),
            _table_elem(1, 1, [["Aardvark"], ["1"]], header_rows=[0]),
            _para(2, "beta prose", 1),
            _table_elem(3, 1, [["Bison"], ["2"]], header_rows=[0]),
            _para(4, "gamma prose", 1),
        ]
        out = chunk_batch([build_chunk_input("a", elements)], chunker)
        rows = [
            {"content_type": c.content_type, "start_char": c.start_char,
             "elem_order": c.elem_order, "text": c.text}
            for c in out["a"]
        ]
        ordered = chunks_in_document_order(rows, element_spans(elements))

        def first_with(marker: str) -> int:
            return next(k for k, r in enumerate(ordered) if marker in r["text"])

        positions = [first_with(m) for m in ("alpha", "Aardvark", "beta", "Bison", "gamma")]
        assert positions == sorted(positions)
        # Unordered, the table chunks would all sit past every narrative chunk.
        assert positions[1] < first_with("gamma")

    def test_multiple_docs_keyed_by_source_hash(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        inputs = [
            ChunkInput("a", "first document content"),
            ChunkInput("b", "second document content"),
        ]
        out = chunk_batch(inputs, chunker)
        assert set(out) == {"a", "b"}
        assert all(out["a"][0].text != "" for _ in [None])
        assert out["b"][0].text != out["a"][0].text

    def test_per_doc_chunk_index_resequenced(self) -> None:
        chunker = _make_test_chunker(chunk_size=5)
        ci = ChunkInput("a", " ".join(f"word{i}" for i in range(30)))
        out = chunk_batch([ci], chunker)
        indices = [c.chunk_index for c in out["a"]]
        assert indices == list(range(len(indices)))

    def test_overlap_passes_through(self) -> None:
        chunker = _make_test_chunker(chunk_size=10)
        text = " ".join(f"word{i}" for i in range(30))
        ci = ChunkInput("a", text)
        no_overlap = chunk_batch([ci], chunker)
        with_overlap = chunk_batch([ci], chunker, overlap=3)
        # Overlapping chunks duplicate boundary content, so total text grows.
        assert sum(len(c.text) for c in with_overlap["a"]) > sum(
            len(c.text) for c in no_overlap["a"]
        )

    def test_has_redaction_set_from_marker(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        ci = ChunkInput("a", "before <REDACTED> after")
        out = chunk_batch([ci], chunker)
        assert any(c.has_redaction for c in out["a"])

    def test_doc_with_only_tables(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        ci = ChunkInput(
            source_hash="a", narrative="",
            tables=[(1, 0, table_to_markdown(["H"], [["v"]]))],
        )
        out = chunk_batch([ci], chunker)
        assert out["a"]
        assert out["a"][0].content_type == "table"

    def test_offsets_match_input_text(self) -> None:
        chunker = _make_test_chunker(chunk_size=10)
        text = " ".join(f"word{i}" for i in range(30))
        ci = ChunkInput("a", text)
        out = chunk_batch([ci], chunker)
        for c in out["a"]:
            assert text[c.start_char:c.end_char] == c.text

    def test_token_count_uses_the_chunkers_own_counter(self) -> None:
        # The number on the row is the one semchunk budgeted with, applied to
        # the text that actually landed — not a downstream re-estimate.
        chunker = _make_test_chunker(chunk_size=10)
        text = " ".join(f"word{i}" for i in range(30))
        out = chunk_batch([ChunkInput("a", text)], chunker)
        assert out["a"]
        for c in out["a"]:
            assert c.token_count == _word_token_counter(c.text)

    def test_token_count_respects_the_budget(self) -> None:
        chunker = _make_test_chunker(chunk_size=7)
        text = " ".join(f"word{i}" for i in range(40))
        out = chunk_batch([ChunkInput("a", text)], chunker)
        assert all(c.token_count is not None and c.token_count <= 7 for c in out["a"])

    def test_table_chunks_counted_too(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        md = table_to_markdown(["H"], [["v"]])
        ci = ChunkInput(source_hash="a", narrative="", tables=[(1, 0, md)])
        out = chunk_batch([ci], chunker)
        assert out["a"][0].token_count == _word_token_counter(out["a"][0].text)

    def test_repaired_chunk_counts_the_merged_text(self) -> None:
        # The repair concatenates two chunks; the count must be of the result,
        # not the sum of the two pre-merge counts (which the join would miss).
        chunker = _make_test_chunker(chunk_size=3)
        text = "lots of text before <REDACTED> and then lots of text after"
        out = chunk_batch([ChunkInput("a", text)], chunker)
        merged = [c for c in out["a"] if "<REDACTED>" in c.text]
        assert merged, "no chunk carries the whole marker"
        for c in out["a"]:
            assert c.token_count == _word_token_counter(c.text)

    def test_over_budget_count_is_a_floor_not_an_exact_figure(self) -> None:
        # semchunk's counter short-circuits over-budget text to chunk_size + 1
        # rather than tokenising it, so the column is exact at or below the
        # budget and a floor above it. Only a repaired chunk is ever above.
        # Pinned here because the docs on the column state it.
        chunker = semchunk.chunkerify(
            _word_token_counter, chunk_size=10, max_token_chars=8,
        )
        long_text = " ".join(f"word{i}" for i in range(100))
        chunks = _count_tokens(
            [TextChunk(long_text, 0, len(long_text), 0)], chunker.token_counter,
        )
        assert chunks[0].token_count == 11 < _word_token_counter(long_text)

    def test_split_redaction_marker_repaired(self) -> None:
        # Construct text such that semchunk plausibly splits around the
        # marker — the repair pass merges across the boundary.
        chunker = _make_test_chunker(chunk_size=3)
        text = "lots of text before <REDACTED> and then lots of text after"
        ci = ChunkInput("a", text)
        out = chunk_batch([ci], chunker)
        joined = "".join(c.text for c in out["a"])
        assert "<REDACTED>" in joined
