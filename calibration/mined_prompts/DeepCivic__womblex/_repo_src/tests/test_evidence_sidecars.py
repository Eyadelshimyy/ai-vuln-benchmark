"""PII spans, entity links and money spans carry an evidence reference.

Each row is located at the most precise level that is reliable, checked at its
own precision before the batch is written, and older files read at the level
their old columns supported.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests._shard import DOC, NARRATIVE, contact_shard, table_markdown, write_contact_chunks
from tests.test_enrich_packing import _FakeClient, _FakeCounter
from womblex.analyse.enrich_stage import enrich_shards
from womblex.config import EnrichmentConfig, LinkingConfig, MoneyConfig, PIIConfig, ReferenceConfig
from womblex.link.stage import link_shards
from womblex.pii.pii_stage import pii_shards
from womblex.process.money_stage import money_shards
from womblex.store.enrichment_output import (
    ENTITY_SCHEMA,
    enrichment_entities_path_for,
    read_enrichment_entities,
)
from womblex.store.entity_links_output import read_entity_links
from womblex.store.evidence import EVIDENCE_COLUMNS
from womblex.store.money_output import MONEY_SPANS_SCHEMA, read_money_spans
from womblex.store.pii_output import read_pii_spans


def _pii(d, *, chunk_markdown: str | None = None, drop: str | None = None) -> list[dict]:
    base = contact_shard(d)
    enrich_shards(d, EnrichmentConfig(), client=_FakeClient(), token_counter=_FakeCounter())
    write_contact_chunks(base, chunk_markdown or table_markdown(base))
    if drop:
        (d / f"batch-0001.{drop}.parquet").unlink()
    pii_shards(d, PIIConfig(use_regex_backstop=False))
    return read_pii_spans(d).to_pylist()


def _levels(rows: list[dict]) -> dict[str, str]:
    return {r["content_type"]: r["anchor_level"] for r in rows}


def test_pii_spans_are_located_to_the_span_in_their_source(tmp_path):
    rows = _pii(tmp_path)

    assert _levels(rows) == {"narrative": "span", "table": "span"}
    narrative = next(r for r in rows if r["content_type"] == "narrative")
    table = next(r for r in rows if r["content_type"] == "table")
    assert NARRATIVE[narrative["char_start"]:narrative["char_end"]] == narrative["text"]
    assert (table["text_layer"], table["elem_order"]) == ("table_markdown", 1)


def test_a_narrative_chunk_out_of_step_with_its_source_is_located_to_the_document(tmp_path):
    base = contact_shard(tmp_path)
    enrich_shards(tmp_path, EnrichmentConfig(), client=_FakeClient(), token_counter=_FakeCounter())
    write_contact_chunks(base, table_markdown(base))
    # A repaired chunk: same text but for one character, so it is not the source's slice.
    chunks = pq.read_table(str(tmp_path / "batch-0001.chunks.parquet")).to_pylist()
    chunks[0]["text"] = chunks[0]["text"].replace("about", "abou t")
    import pyarrow as pa

    from womblex.store.output import CHUNKS_SCHEMA
    pq.write_table(pa.Table.from_pylist(chunks, schema=CHUNKS_SCHEMA),
                   str(tmp_path / "batch-0001.chunks.parquet"))

    pii_shards(tmp_path, PIIConfig(use_regex_backstop=False))

    narrative = next(r for r in read_pii_spans(tmp_path).to_pylist()
                     if r["content_type"] == "narrative")
    assert narrative["anchor_level"] == "document" and narrative["char_start"] is None
    assert narrative["text"] == "Cont"  # still masked: masking never depends on the receipt


def test_a_table_that_cannot_be_found_or_does_not_line_up_is_located_coarser(tmp_path):
    out_of_step = _pii(tmp_path / "a", chunk_markdown="| Name |\n| --- |\n| a repaired chunk |")
    # `_pii` with no elements at all: the chunk is all that can be named.
    no_elements = _pii(tmp_path / "b", drop="elements")

    assert not [r for r in out_of_step if r["content_type"] == "table"]  # takes no table spans
    assert _levels(no_elements)["narrative"] == "chunk"
    chunk_row = next(r for r in no_elements if r["content_type"] == "narrative")
    assert (chunk_row["page"], chunk_row["char_start"]) == (1, None)


# ---------------------------------------------------------------------------
# Entity links
# ---------------------------------------------------------------------------


@pytest.fixture
def reference(tmp_path) -> ReferenceConfig:
    csv = tmp_path / "register.csv"
    csv.write_text("Id,Name\nPR-1,Cont Holdings\nPR-2,John Citizen Pty\n", encoding="utf-8")
    return ReferenceConfig(
        path=csv, id_col="Id", name_col="Name", entity_type="provider", match_fuzzy_cols=["Name"],
    )


def _links(d, reference, *, kinds=("natural",), edit=None) -> list[dict]:
    base = contact_shard(d)
    enrich_shards(d, EnrichmentConfig(), client=_FakeClient(), token_counter=_FakeCounter())
    if edit:
        rows = [edit(r) for r in read_enrichment_entities(base).to_pylist()]
        pq.write_table(pa.Table.from_pylist(rows, schema=ENTITY_SCHEMA),
                       str(enrichment_entities_path_for(base)))
    link_shards(d, LinkingConfig(reference=reference, candidate_kinds=list(kinds)))
    return read_entity_links(d).to_pylist()


def test_links_are_located_to_their_mention_and_carry_the_enrichers_text(tmp_path, reference):
    rows = _links(tmp_path, reference)

    assert {(r["text_layer"], r["anchor_level"]) for r in rows} == {
        ("elements", "span"), ("table_markdown", "span")}
    table = next(r for r in rows if r["text_layer"] == "table_markdown")
    assert table["elem_order"] == 1 and table["mention_text"] == table_markdown(
        tmp_path / "batch-0001.parquet")[:4]


def test_a_mention_the_provider_gave_no_offsets_for_is_located_to_the_document(
    tmp_path, reference,
):
    rows = _links(tmp_path, reference, edit=lambda r: {
        **r, "mention_start": -1, "mention_end": -1, "mention_text": None})

    assert {r["anchor_level"] for r in rows} == {"document"}
    assert all(r["char_start"] is None for r in rows)


def test_a_link_whose_slice_is_not_its_mention_is_located_to_the_document(tmp_path, reference):
    rows = _links(tmp_path, reference, edit=lambda r: {**r, "mention_end": r["mention_end"] - 1})

    assert {(r["anchor_level"], r["evidence_check"]) for r in rows} == {("document", "holds")}


def test_a_link_whose_mention_text_is_not_in_the_source_is_written_labelled_fails(
    tmp_path, reference,
):
    rows = _links(tmp_path, reference,
                  edit=lambda r: {**r, "mention_text": "not the source text"})

    assert rows and {(r["anchor_level"], r["evidence_check"]) for r in rows} == {
        ("document", "fails")}


# ---------------------------------------------------------------------------
# Money, and files written before the evidence reference
# ---------------------------------------------------------------------------


def test_money_cells_are_located_to_the_span_in_their_cell(tmp_path):
    base = contact_shard(tmp_path)
    money_shards(tmp_path, MoneyConfig())

    cells = [r for r in read_money_spans(base).to_pylist() if r["locus"] == "table_cell"]

    assert cells and {r["anchor_level"] for r in cells} == {"span"}
    assert {r["text_layer"] for r in cells} == {"cell"}


def test_older_files_read_at_the_level_their_old_columns_supported(tmp_path):
    money = {f.name: pa.nulls(2, f.type) for f in MONEY_SPANS_SCHEMA
             if f.name not in EVIDENCE_COLUMNS}
    money |= {
        "source_hash": [DOC, DOC], "locus": ["narrative", "table_cell"],
        "text": ["$5,000", "1,200"], "text_source": ["elements", None],
        "start_char": pa.array([5, None], pa.int32()), "end_char": pa.array([11, None], pa.int32()),
        "page": pa.array([1, None], pa.int32()), "elem_order": pa.nulls(2, pa.int32()),
        "parent_elem_order": pa.array([None, 1], pa.int32()), "sheet": pa.nulls(2, pa.string()),
        "row": pa.array([None, 2], pa.int32()), "col": pa.array([None, 1], pa.int32()),
    }
    pq.write_table(pa.table(money), str(tmp_path / "batch-0001.money_spans.parquet"))
    pq.write_table(pa.table({
        "source_hash": [DOC], "chunk_index": pa.array([0], pa.int32()),
        "content_type": ["narrative"], "start": pa.array([8], pa.int32()),
        "end": pa.array([16], pa.int32()), "text": ["Jane Doe"], "entity_type": ["PERSON"],
        "entity_id": ["e1"], "detector": ["enrichment"], "score": pa.array([1.0], pa.float32()),
        "replacement": ["<PERSON_1>"],
    }), str(tmp_path / "batch-0001.pii_spans.parquet"))
    pq.write_table(pa.table({
        "source_hash": [DOC], "candidate_text": ["Acme"], "candidate_kind": ["corporate"],
        "mention_start": pa.array([0], pa.int32()), "mention_end": pa.array([4], pa.int32()),
        "entity_id": ["PR-1"], "entity_type": ["provider"], "canonical_name": ["Acme"],
        "parent_entity_id": [""], "confidence": pa.array([1.0], pa.float32()),
        "match_method": ["alias"], "matched": [True],
    }), str(tmp_path / "batch-0001.entity_links.parquet"))

    narrative, cell = read_money_spans(tmp_path).to_pylist()
    span, link = read_pii_spans(tmp_path).to_pylist()[0], read_entity_links(tmp_path).to_pylist()[0]

    assert (narrative["anchor_level"], narrative["char_start"], narrative["text_layer"]) == (
        "span", 5, "elements")
    assert (cell["anchor_level"], cell["elem_order"], cell["cell_row"], cell["cell_col"]) == (
        "element", 1, 2, 1)
    assert span["anchor_level"] == "chunk" and span["chunk_index"] == 0
    assert link["anchor_level"] == "document" and link["mention_text"] is None
    for row in (narrative, cell, span, link):
        assert set(EVIDENCE_COLUMNS) <= set(row) and "start_char" not in row
        assert "start" not in row and "mention_start" not in row and "parent_elem_order" not in row


def test_the_inspector_gets_a_span_position_only_for_a_span_level_receipt():
    from womblex.ui.readers import _chunk_relative_pii

    detail = {
        "chunks": [{"chunk_index": 3, "start_char": 100}],
        "pii_spans": [
            {"chunk_index": 3, "anchor_level": "span", "char_start": 104, "char_end": 112},
            {"chunk_index": 3, "anchor_level": "document", "char_start": None, "char_end": None},
        ],
    }

    _chunk_relative_pii(detail)

    assert [(s["start"], s["end"]) for s in detail["pii_spans"]] == [(4, 12), (-1, -1)]
