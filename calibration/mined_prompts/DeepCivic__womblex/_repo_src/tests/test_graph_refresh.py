"""Offline tests for the graph-edge refresh stage.

No Isaacus: a shard is built with the real records ingest (for the manifest
the stage discovers batches by), then entity + chunk sidecars are written
directly with known offsets so the refresh links mentions to chunks
deterministically.
"""

from __future__ import annotations

from pathlib import Path

from womblex.analyse.graph_refresh import refresh_graph_edges
from womblex.ingest.records import RecordFieldMapping, ingest_records, records_source_hash
from womblex.store.enrichment_output import (
    read_enrichment_entities,
    read_graph_edges,
    write_enrichment_entities_rows,
    write_graph_edges_rows,
)
from womblex.store.output import write_chunks

_MAPPING = RecordFieldMapping(id_field="version_id", text_field="text")
_DOC_ID = "doc-1"
_TEXT = "word " * 60  # arbitrary; the refresh works on offsets, not content


def _src() -> str:
    return records_source_hash(_DOC_ID, _TEXT)


def _entity_row(src: str, entity_id: str, start: int, end: int) -> dict:
    return {
        "source_hash": src, "entity_id": entity_id, "entity_label": "location",
        "name": "somewhere", "entity_type": "address", "role": "",
        "mention_start": start, "mention_end": end, "chunk_index": -1,
    }


def _chunk_row(src: str, chunk_index: int, start: int, end: int) -> dict:
    return {
        "source_hash": src, "chunk_index": chunk_index, "text": "x",
        "start_char": start, "end_char": end, "content_type": "narrative",
        "has_redaction": False, "page_start": None, "page_end": None,
    }


def _setup_shard(tmp_path: Path) -> tuple[Path, Path, str]:
    d = tmp_path / "documents"
    ingest_records([{"version_id": _DOC_ID, "text": _TEXT}], d, _MAPPING)
    src = _src()
    base = d / "batch-0001.parquet"
    # two chunks: [0,100) and [100,200); a mention at 120-130 lands in chunk 1
    write_chunks([_chunk_row(src, 0, 0, 100), _chunk_row(src, 1, 100, 200)], base)
    write_enrichment_entities_rows(
        [_entity_row(src, "loc:0", 10, 20), _entity_row(src, "loc:1", 120, 130)], base,
    )
    return d, base, src


def test_populates_chunk_index_and_edges(tmp_path):
    d, base, src = _setup_shard(tmp_path)
    result = refresh_graph_edges(d)
    assert result.batches_written == 1
    assert result.docs_refreshed == 1

    rows = {r["entity_id"]: r for r in read_enrichment_entities(base).to_pylist()}
    assert rows["loc:0"]["chunk_index"] == 0  # 10-20 → chunk 0
    assert rows["loc:1"]["chunk_index"] == 1  # 120-130 → chunk 1

    edges = read_graph_edges(base).to_pylist()
    mention_edges = [e for e in edges if e["relation"] == "mentioned_in"]
    targets = {e["target_id"] for e in mention_edges}
    assert f"{src}:chunk:0" in targets and f"{src}:chunk:1" in targets
    assert {e["source_id"] for e in mention_edges} == {f"{src}:loc:0", f"{src}:loc:1"}


def test_preserves_non_mention_edges_and_replaces_stale(tmp_path):
    d, base, src = _setup_shard(tmp_path)
    # a pre-existing 'cites' edge (must survive) + a stale 'mentioned_in' (dropped)
    write_graph_edges_rows([
        {"source_hash": src, "source_id": src, "target_id": f"{src}:ext:0",
         "relation": "cites", "prop_key": "", "prop_value": ""},
        {"source_hash": src, "source_id": f"{src}:loc:0", "target_id": f"{src}:chunk:9",
         "relation": "mentioned_in", "prop_key": "start", "prop_value": "999"},
    ], base)

    refresh_graph_edges(d)
    edges = read_graph_edges(base).to_pylist()
    assert any(e["relation"] == "cites" for e in edges)  # preserved
    mention_targets = {e["target_id"] for e in edges if e["relation"] == "mentioned_in"}
    assert f"{src}:chunk:9" not in mention_targets  # stale edge dropped


def test_idempotent(tmp_path):
    d, base, _ = _setup_shard(tmp_path)
    refresh_graph_edges(d)
    first = read_graph_edges(base).to_pylist()
    refresh_graph_edges(d)
    second = read_graph_edges(base).to_pylist()
    assert first == second


def test_skips_batch_without_chunks(tmp_path):
    d = tmp_path / "documents"
    ingest_records([{"version_id": _DOC_ID, "text": _TEXT}], d, _MAPPING)
    base = d / "batch-0001.parquet"
    write_enrichment_entities_rows([_entity_row(_src(), "loc:0", 10, 20)], base)
    # no chunks sidecar → nothing to link
    result = refresh_graph_edges(d)
    assert result.batches_written == 0
