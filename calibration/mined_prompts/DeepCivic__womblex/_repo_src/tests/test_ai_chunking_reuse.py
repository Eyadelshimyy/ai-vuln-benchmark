"""Offline tests for semchunk-4 AI-chunking enrichment reuse.

Covers the Womblex-owned seams of the single-enrichment reuse design
(``docs/decisions.md``) without a live Isaacus key:

- ``store/enrichment_doc.py`` round-trips ``(source_hash, text_source,
  document_json)``;
- the byte-identity reuse guard in ``chunker._resolve_narrative_input`` /
  ``chunk_batch`` honours a matching Document and falls back to the plain
  string on any mismatch or absent sidecar;
- ``WomblexConfig`` auto-enables ``enrichment.persist_document`` only when AI
  chunking + enrich are both on.

semchunk's actual handling of a real SDK Document is covered end-to-end by the
live AI-chunking test (``tests/test_pipeline.py::TestAiChunkingLive``, real
``kanon-2-enricher``, skips without a key) and the live verification gates
recorded in ``docs/decisions.md``; a fake duck-typed Document cannot pass
semchunk's ``isinstance`` check, so the ``chunk_batch`` wiring is exercised here
with a recording fake chunker instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from womblex.config import ChunkingConfig, EnrichmentConfig, WomblexConfig
from womblex.process.chunker import ChunkInput, _resolve_narrative_input, chunk_batch
from womblex.store.enrichment_doc import (
    enrichment_doc_path_for,
    read_enrichment_docs,
    write_enrichment_doc_shard,
)


@dataclass
class _FakeDoc:
    """Duck-typed stand-in for an ILGS Document — only ``.text`` is read."""

    text: str


class _RecordingChunker:
    """Fake semchunk chunker: records inputs, returns one chunk per item."""

    def __init__(self) -> None:
        self.received: list[object] | None = None
        # A real semchunk.Chunker exposes the counter it budgeted with;
        # chunk_batch reads it to stamp each chunk's token_count.
        self.token_counter = lambda text: len(text.split())

    def __call__(
        self, texts, *, offsets=False, overlap=None, processes=1, progress=False,
    ):
        self.received = list(texts)
        chunks, offs = [], []
        for item in texts:
            s = getattr(item, "text", item)
            chunks.append([s])
            offs.append([(0, len(s))])
        return chunks, offs


# ---------------------------------------------------------------------------
# store/enrichment_doc.py
# ---------------------------------------------------------------------------


class TestEnrichmentDocStore:
    def test_round_trip(self, tmp_path: Path) -> None:
        base = tmp_path / "batch-0001.parquet"
        rows = [
            ("hashA", "elements", '{"text": "alpha"}'),
            ("hashB", "spellfix", '{"text": "beta"}'),
        ]
        out = write_enrichment_doc_shard(rows, base)
        assert out == enrichment_doc_path_for(base)
        assert out.exists()

        got = read_enrichment_docs(base)
        assert got == {
            "hashA": ("elements", '{"text": "alpha"}'),
            "hashB": ("spellfix", '{"text": "beta"}'),
        }

    def test_empty_writes_schema_correct_file(self, tmp_path: Path) -> None:
        base = tmp_path / "batch-0002.parquet"
        write_enrichment_doc_shard([], base)
        assert enrichment_doc_path_for(base).exists()
        assert read_enrichment_docs(base) == {}

    def test_missing_sidecar_returns_empty(self, tmp_path: Path) -> None:
        base = tmp_path / "batch-0003.parquet"
        assert read_enrichment_docs(base) == {}


# ---------------------------------------------------------------------------
# Byte-identity reuse guard
# ---------------------------------------------------------------------------


class TestReuseGuard:
    def test_matching_text_uses_override(self) -> None:
        doc = ChunkInput(source_hash="A", narrative="alpha beta gamma")
        override = _FakeDoc(text="alpha beta gamma")
        assert _resolve_narrative_input(doc, {"A": override}) is override

    def test_mismatched_text_falls_back_to_string(self, caplog) -> None:
        doc = ChunkInput(source_hash="A", narrative="alpha beta gamma")
        override = _FakeDoc(text="alpha beta DIFFERENT")
        with caplog.at_level(logging.WARNING):
            result = _resolve_narrative_input(doc, {"A": override})
        assert result == "alpha beta gamma"
        assert "reuse guard" in caplog.text

    def test_absent_override_uses_string(self) -> None:
        doc = ChunkInput(source_hash="A", narrative="alpha beta")
        assert _resolve_narrative_input(doc, {}) == "alpha beta"


# ---------------------------------------------------------------------------
# chunk_batch wiring
# ---------------------------------------------------------------------------


class TestChunkBatchReuse:
    def test_override_reaches_chunker_only_on_match(self) -> None:
        inputs = [
            ChunkInput(source_hash="A", narrative="alpha beta"),
            ChunkInput(source_hash="B", narrative="gamma delta"),
        ]
        overrides = {
            "A": _FakeDoc(text="alpha beta"),       # matches → reused
            "B": _FakeDoc(text="stale narrative"),  # mismatch → string
        }
        chunker = _RecordingChunker()
        out = chunk_batch(inputs, chunker, narrative_overrides=overrides)

        assert chunker.received is not None
        assert chunker.received[0] is overrides["A"]
        assert chunker.received[1] == "gamma delta"
        # Both still produce chunks keyed by source_hash.
        assert set(out) == {"A", "B"}

    def test_no_overrides_passes_plain_strings(self) -> None:
        inputs = [ChunkInput(source_hash="A", narrative="alpha beta")]
        chunker = _RecordingChunker()
        chunk_batch(inputs, chunker, narrative_overrides=None)
        assert chunker.received == ["alpha beta"]


# ---------------------------------------------------------------------------
# Config auto-enable
# ---------------------------------------------------------------------------


class TestConfigAutoEnable:
    _PATHS: ClassVar[dict[str, str]] = {
        "input_root": "/tmp/i", "output_root": "/tmp/o", "checkpoint_dir": "/tmp/c",
    }

    def test_auto_enabled_when_ai_chunking_and_enrich(self) -> None:
        cfg = WomblexConfig(
            dataset={"name": "t"}, paths=self._PATHS,
            chunking=ChunkingConfig(chunking_model="kanon-2-enricher"),
            enrichment=EnrichmentConfig(enabled=True),
        )
        assert cfg.enrichment.persist_document is True

    def test_off_by_default(self) -> None:
        cfg = WomblexConfig(dataset={"name": "t"}, paths=self._PATHS)
        assert cfg.enrichment.persist_document is False

    def test_not_enabled_when_enrich_off(self) -> None:
        cfg = WomblexConfig(
            dataset={"name": "t"}, paths=self._PATHS,
            chunking=ChunkingConfig(chunking_model="kanon-2-enricher"),
            enrichment=EnrichmentConfig(enabled=False),
        )
        assert cfg.enrichment.persist_document is False
