"""Tests for the per-stage PII span detection (``pii_shards``) + sidecar IO.

Builds a minimal shard dir directly (manifest for batch discovery + chunks +
enrichment entities) — no heavy extraction. The Kanon-2 graph is mocked as
``enrichment_entities`` rows. Context-model scoring is monkeypatched to zero so
the tests are deterministic and don't need sentence-transformers; the
high-confidence honorific regex (no model) exercises the backstop.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.config import PIIConfig
from womblex.pii.cleaner import PIICleaner
from womblex.pii.pii_stage import pii_shards
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_output import (
    ENTITY_SCHEMA,
    enrichment_entities_path_for,
    enrichment_meta_path_for,
)
from womblex.store.output import MANIFEST_SCHEMA, write_chunks
from womblex.store.pii_output import (
    CLEAN_TEXT_SCHEMA,
    PII_SPANS_SCHEMA,
    pii_spans_path_for,
    read_clean_text,
    read_pii_spans,
    write_clean_text,
    write_pii_spans,
)

DOC = "doc1"
# Narrative chunk 0: "Jane Doe" is a graph person mention at offsets 12..20.
_NARRATIVE = "The officer Jane Doe signed the notice."
# Table chunk 1: same text body, separate offset space; honorific for backstop.
_TABLE = "Row contact Mr Smith approved."


def _chunk_row(idx: int, text: str, content_type: str) -> dict:
    return {
        "source_hash": DOC, "chunk_index": idx, "text": text,
        "start_char": 0, "end_char": len(text), "content_type": content_type,
        "has_redaction": False, "page_start": 1, "page_end": 1,
    }


def _entity_row(start: int, end: int, entity_type: str, eid: str = "e1") -> dict:
    return {
        "source_hash": DOC, "entity_id": eid, "entity_label": "person",
        "name": "Jane Doe", "entity_type": entity_type, "role": "other",
        "mention_start": start, "mention_end": end, "chunk_index": -1,
    }


def _build_shard(d: Path, *, chunk_rows: list[dict], entity_rows: list[dict]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    base = d / "batch-0001.parquet"
    # Manifest: required for _batch_bases discovery + checkpoint doc_ids.
    man = {f.name: None for f in MANIFEST_SCHEMA}
    man.update({"source_hash": DOC, "doc_id": DOC, "filename": "doc1.pdf", "status": "ok"})
    pq.write_table(
        pa.Table.from_pylist([man], schema=MANIFEST_SCHEMA),
        str(d / "batch-0001._manifest.parquet"),
    )
    write_chunks(chunk_rows, base)
    pq.write_table(
        pa.Table.from_pylist(entity_rows, schema=ENTITY_SCHEMA),
        str(enrichment_entities_path_for(base)),
    )
    return base


def _no_context_model(monkeypatch) -> None:
    """Force low-confidence (title-case) candidates to fail — no model load."""
    monkeypatch.setattr(
        PIICleaner, "_score_context_batch",
        lambda self, text, cands: [0.0] * len(cands),
    )


class TestPiiShards:
    def test_graph_span_on_narrative(self, tmp_path, monkeypatch):
        _no_context_model(monkeypatch)
        start, end = _NARRATIVE.index("Jane Doe"), _NARRATIVE.index("Jane Doe") + 8
        d = tmp_path / "documents"
        _build_shard(
            d,
            chunk_rows=[_chunk_row(0, _NARRATIVE, "narrative")],
            entity_rows=[_entity_row(start, end, "natural")],
        )
        result = pii_shards(d, PIIConfig(use_regex_backstop=False))
        assert result.batches_written == 1
        spans = read_pii_spans(d).to_pylist()
        assert len(spans) == 1
        s = spans[0]
        assert s["entity_type"] == "PERSON"
        assert s["detector"] == "enrichment"
        assert s["text"] == "Jane Doe"
        assert s["chunk_index"] == 0
        assert s["entity_id"] == "e1"
        assert s["replacement"] == "<PERSON_1>"
        # clean_text layer: span replaced with the typed+numbered tag
        clean = read_clean_text(d).to_pylist()
        assert len(clean) == 1
        assert clean[0]["text"] == "The officer <PERSON_1> signed the notice."
        assert clean[0]["n_masked"] == 1

    def test_numbering_is_per_entity_and_stable(self, tmp_path, monkeypatch):
        _no_context_model(monkeypatch)
        text = "Jane Doe met Mark Lee, then Jane Doe left."
        a0, a1 = 0, 8                      # first "Jane Doe"
        b0, b1 = 13, 21                    # "Mark Lee"
        a2, a3 = text.index("Jane Doe", 9), text.index("Jane Doe", 9) + 8  # 2nd "Jane Doe"
        d = tmp_path / "documents"
        _build_shard(
            d,
            chunk_rows=[_chunk_row(0, text, "narrative")],
            entity_rows=[
                _entity_row(a0, a1, "natural", eid="jane"),
                _entity_row(b0, b1, "natural", eid="mark"),
                _entity_row(a2, a3, "natural", eid="jane"),
            ],
        )
        pii_shards(d, PIIConfig(use_regex_backstop=False))
        clean = read_clean_text(d).to_pylist()[0]["text"]
        # distinct entities get distinct numbers; repeated entity keeps its number
        assert clean == "<PERSON_1> met <PERSON_2>, then <PERSON_1> left."

    def test_graph_span_not_applied_to_table_chunk(self, tmp_path, monkeypatch):
        _no_context_model(monkeypatch)
        # Graph mention offsets land inside the table chunk's local range, but
        # graph spans must only apply to narrative chunks.
        d = tmp_path / "documents"
        _build_shard(
            d,
            chunk_rows=[_chunk_row(1, _TABLE, "table")],
            entity_rows=[_entity_row(0, 8, "natural")],
        )
        result = pii_shards(d, PIIConfig(use_regex_backstop=False))
        assert result.spans_written == 0  # no graph on table, regex off

    def test_regex_backstop_covers_table_chunk(self, tmp_path, monkeypatch):
        _no_context_model(monkeypatch)
        d = tmp_path / "documents"
        _build_shard(
            d,
            chunk_rows=[_chunk_row(1, _TABLE, "table")],
            entity_rows=[],
        )
        on = pii_shards(d, PIIConfig(use_regex_backstop=True))
        spans = read_pii_spans(d).to_pylist()
        assert on.spans_written >= 1
        person = [s for s in spans if s["entity_type"] == "PERSON"]
        assert person and person[0]["detector"] == "regex_high"
        assert "Smith" in person[0]["text"]
        assert person[0]["entity_id"] == ""  # regex spans have no graph id
        assert person[0]["replacement"].startswith("<PERSON_")

    def test_backstop_off_yields_nothing_without_graph(self, tmp_path, monkeypatch):
        _no_context_model(monkeypatch)
        d = tmp_path / "documents"
        _build_shard(
            d, chunk_rows=[_chunk_row(1, _TABLE, "table")], entity_rows=[],
        )
        result = pii_shards(d, PIIConfig(use_regex_backstop=False))
        assert result.spans_written == 0

    def test_checkpoint_skips_on_resume(self, tmp_path, monkeypatch):
        _no_context_model(monkeypatch)
        start, end = _NARRATIVE.index("Jane Doe"), _NARRATIVE.index("Jane Doe") + 8
        d = tmp_path / "documents"
        _build_shard(
            d,
            chunk_rows=[_chunk_row(0, _NARRATIVE, "narrative")],
            entity_rows=[_entity_row(start, end, "natural")],
        )
        ckpt = CheckpointManager(tmp_path / ".pii-checkpoint", "pii")
        first = pii_shards(d, PIIConfig(use_regex_backstop=False), checkpoint_mgr=ckpt)
        assert first.batches_written == 1
        second = pii_shards(d, PIIConfig(use_regex_backstop=False), checkpoint_mgr=ckpt)
        assert second.batches_written == 0  # all docs checkpointed → skipped


class TestMaskStatus:
    def _statuses(self, d: Path) -> list[str | None]:
        rows = sorted(read_clean_text(d).to_pylist(), key=lambda r: r["chunk_index"])
        return [r["mask_status"] for r in rows]

    def _chunks(self) -> list[dict]:
        return [_chunk_row(0, _NARRATIVE, "narrative"), _chunk_row(1, _TABLE, "table")]

    def test_masked_and_uncovered_table(self, tmp_path, monkeypatch):
        _no_context_model(monkeypatch)
        s = _NARRATIVE.index("Jane Doe")
        d = tmp_path / "documents"
        _build_shard(d, chunk_rows=self._chunks(), entity_rows=[_entity_row(s, s + 8, "natural")])
        pii_shards(d, PIIConfig(use_regex_backstop=False))
        assert self._statuses(d) == ["masked", "not_masked"]

    def test_no_enrichment_is_not_masked(self, tmp_path, monkeypatch):
        _no_context_model(monkeypatch)
        d = tmp_path / "documents"
        _build_shard(d, chunk_rows=self._chunks(), entity_rows=[])
        pii_shards(d, PIIConfig(use_regex_backstop=False))
        assert self._statuses(d) == ["not_masked", "not_masked"]

    def test_enriched_doc_with_no_pii_is_no_entity(self, tmp_path, monkeypatch):
        _no_context_model(monkeypatch)
        d = tmp_path / "documents"
        base = _build_shard(d, chunk_rows=self._chunks(), entity_rows=[])
        pq.write_table(
            pa.table({"source_hash": [DOC]}), str(enrichment_meta_path_for(base)),
        )
        pii_shards(d, PIIConfig(use_regex_backstop=False))
        assert self._statuses(d) == ["no_entity", "not_masked"]

    def test_regex_backstop_covers_every_chunk(self, tmp_path, monkeypatch):
        _no_context_model(monkeypatch)
        d = tmp_path / "documents"
        _build_shard(d, chunk_rows=[_chunk_row(0, "Nothing here.", "table")], entity_rows=[])
        pii_shards(d, PIIConfig(use_regex_backstop=True))
        assert self._statuses(d) == ["no_entity"]


class TestPiiSpansIO:
    def test_round_trip(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        rows = [{
            "source_hash": DOC, "chunk_index": 0, "content_type": "narrative",
            "start": 12, "end": 20, "text": "Jane Doe", "entity_type": "PERSON",
            "entity_id": "e1", "detector": "enrichment", "score": 0.95,
            "replacement": "<PERSON_1>",
        }]
        write_pii_spans(rows, base)
        assert pii_spans_path_for(base).exists()
        table = read_pii_spans(base)
        assert table.schema.names == [f.name for f in PII_SPANS_SCHEMA]
        assert table.to_pylist()[0]["replacement"] == "<PERSON_1>"

    def test_empty_is_schema_correct(self, tmp_path):
        base = tmp_path / "batch-0002.parquet"
        write_pii_spans([], base)
        table = read_pii_spans(base)
        assert table.num_rows == 0
        assert table.schema.names == [f.name for f in PII_SPANS_SCHEMA]

    def test_clean_text_round_trip(self, tmp_path):
        base = tmp_path / "batch-0003.parquet"
        rows = [{
            "source_hash": DOC, "chunk_index": 0, "content_type": "narrative",
            "text": "The officer <PERSON_1> signed it.", "n_masked": 1,
            "mask_status": "masked",
        }]
        write_clean_text(rows, base)
        table = read_clean_text(base)
        assert table.schema.names == [f.name for f in CLEAN_TEXT_SCHEMA]
        assert table.to_pylist()[0]["n_masked"] == 1

    def test_older_file_without_mask_status_is_backfilled(self, tmp_path):
        base = tmp_path / "batch-0004.parquet"
        old = pa.table({
            "source_hash": [DOC, DOC], "chunk_index": pa.array([0, 1], pa.int32()),
            "content_type": ["narrative", "table"], "text": ["a", "b"],
            "n_masked": pa.array([2, 0], pa.int32()),
        })
        pq.write_table(old, str(tmp_path / "batch-0004.clean_text.parquet"))
        assert read_clean_text(base).column("mask_status").to_pylist() == ["masked", None]


class TestCmdPii:
    def test_cmd_writes_sidecar(self, tmp_path, monkeypatch):
        import argparse

        from womblex.cli.pii import cmd_pii

        _no_context_model(monkeypatch)
        start, end = _NARRATIVE.index("Jane Doe"), _NARRATIVE.index("Jane Doe") + 8
        d = tmp_path / "documents"
        _build_shard(
            d,
            chunk_rows=[_chunk_row(0, _NARRATIVE, "narrative")],
            entity_rows=[_entity_row(start, end, "natural")],
        )
        args = argparse.Namespace(
            shards=d, config=None, checkpoint_dir=None, dataset="pii-cli",
            no_resume=True, no_verify_resume=True,
        )
        assert cmd_pii(args) == 0
        spans = read_pii_spans(d).to_pylist()
        assert any(s["text"] == "Jane Doe" for s in spans)
