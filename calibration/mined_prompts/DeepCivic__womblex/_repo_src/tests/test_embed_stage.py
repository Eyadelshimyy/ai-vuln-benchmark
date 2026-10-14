"""Tests for the per-stage embed wiring over a shard directory.

Builds a real extraction shard from the budget-statement DOCX fixture, writes a
couple of chunks, and runs ``embed_shards`` against the **live** Isaacus
Kanon-2 embedder (no mocks — the repo validates against the real service
locally; tests skip cleanly when ``ISAACUS_API_KEY`` is unset). Asserts the
embeddings sidecar + checkpoint behaviour (skip-on-resume, no-checkpoint-on-
failure). The failure path uses a real invalid-key client to induce a genuine
API error rather than a stubbed exception.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._synthetic import BUDGET_DOCX as _BUDGET_DOCX
from womblex.analyse.embed_stage import embed_shards
from womblex.config import EmbeddingConfig
from womblex.ingest.strategies_file import DocxExtractor
from womblex.store.checkpoint import CheckpointManager
from womblex.store.embed_output import embeddings_path_for, read_embeddings
from womblex.store.output import read_manifest, write_chunks, write_results

_KANON2_DIM = 1792  # native dimensionality of kanon-2-embedder


@pytest.fixture
def shard_with_chunks(tmp_path) -> Path:
    d = tmp_path / "documents"
    d.mkdir()
    base = d / "batch-0001.parquet"
    write_results([("budget", str(_BUDGET_DOCX), DocxExtractor().extract_path(_BUDGET_DOCX))],
                  base, collection_id="test")
    src = read_manifest(base).column("source_hash").to_pylist()[0]
    write_chunks([
        {"source_hash": src, "chunk_index": 0, "text": "first chunk text",
         "start_char": 0, "end_char": 16, "content_type": "narrative",
         "has_redaction": False, "page_start": 1, "page_end": 1},
        {"source_hash": src, "chunk_index": 1, "text": "second chunk text",
         "start_char": 16, "end_char": 33, "content_type": "narrative",
         "has_redaction": False, "page_start": 1, "page_end": 1},
    ], base)
    return d


class TestEmbedShards:
    def test_writes_embeddings_sidecar(self, shard_with_chunks, isaacus_client):
        result = embed_shards(shard_with_chunks, EmbeddingConfig(), client=isaacus_client)
        assert result.chunks_embedded == 2
        assert embeddings_path_for(shard_with_chunks / "batch-0001.parquet").exists()
        t = read_embeddings(shard_with_chunks)
        assert t.num_rows == 2
        rows = t.to_pylist()
        # real Kanon-2 vectors, not stubs
        assert all(r["dim"] == _KANON2_DIM and len(r["vector"]) == _KANON2_DIM for r in rows)
        assert all(r["model"] == "kanon-2-embedder" for r in rows)
        assert {r["chunk_index"] for r in rows} == {0, 1}

    def test_checkpoint_skips_on_resume(self, shard_with_chunks, isaacus_client, tmp_path):
        ckpt = CheckpointManager(tmp_path / ".embed-ckpt", "t_embed")
        ckpt.load()
        first = embed_shards(shard_with_chunks, EmbeddingConfig(), client=isaacus_client,
                             checkpoint_mgr=ckpt)
        assert first.chunks_embedded == 2
        # Resume: batch is checkpointed → skipped, nothing re-embedded.
        second = embed_shards(shard_with_chunks, EmbeddingConfig(), client=isaacus_client,
                              checkpoint_mgr=ckpt)
        assert second.batches_written == 0
        assert second.chunks_embedded == 0

    def test_failure_not_checkpointed(self, shard_with_chunks, bad_isaacus_client, tmp_path):
        # A real API failure (invalid key) must leave the doc unprocessed so a
        # resume retries it rather than skipping it forever.
        ckpt = CheckpointManager(tmp_path / ".embed-ckpt", "t_embed")
        ckpt.load()
        embed_shards(shard_with_chunks, EmbeddingConfig(), client=bad_isaacus_client,
                     checkpoint_mgr=ckpt)
        assert "budget" not in ckpt.state.processed_ids
