"""Per-stage embedding over an existing shard directory.

Consumes ``*.chunks.parquet`` (written by the chunk stage) and writes a
``*.embeddings.parquet`` sibling per batch — one vector per chunk, joinable
back on ``(source_hash, chunk_index, content_type)``. Chunks are the right
granularity for retrieval. Mirrors :mod:`womblex.analyse.enrich_stage`:
per-stage ``CheckpointManager``, skip-existing on resume, and batch-level
failure isolation (a failed batch is left unprocessed so resume retries it).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from womblex.analyse.embed import MAX_TEXTS_PER_REQUEST, embed_texts
from womblex.config import EmbeddingConfig
from womblex.process.chunk_stage import _batch_bases
from womblex.store.checkpoint import CheckpointManager
from womblex.store.embed_output import embeddings_path_for, write_embeddings
from womblex.store.output import read_chunks, read_manifest
from womblex.utils.token_packer import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class EmbedStageResult:
    batches_written: int
    chunks_embedded: int


def embed_shards(
    shard_dir: Path,
    embedding_config: EmbeddingConfig,
    *,
    client: object,
    checkpoint_mgr: CheckpointManager | None = None,
) -> EmbedStageResult:
    """Embed every batch's chunks and write ``*.embeddings.parquet`` siblings."""
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("embed_shards: no batches found in %s", shard_dir)
        return EmbedStageResult(0, 0)

    batches_written = 0
    chunks_embedded = 0

    for base in bases:
        if checkpoint_mgr is not None and _all_docs_checkpointed(base, checkpoint_mgr):
            logger.info("embed_shards: skipping %s (all docs checkpointed)", base.stem)
            continue

        chunk_rows = _chunk_rows(base)
        rows: list[dict] = []
        errored = False
        if chunk_rows:
            texts = [r["text"] for r in chunk_rows]
            try:
                vectors = embed_texts(
                    texts, client,
                    model=embedding_config.model,
                    task=embedding_config.task,
                    dimensions=embedding_config.dimensions,
                    max_retries=embedding_config.max_retries,
                    retry_base_delay=embedding_config.retry_base_delay,
                )
            except Exception as e:  # transient — leave batch for retry
                logger.error("embed_shards: embedding failed for %s: %s", base.stem, e)
                errored = True
                vectors = []
            if not errored:
                for r, vec in zip(chunk_rows, vectors):
                    rows.append({
                        "source_hash": r["source_hash"],
                        "chunk_index": r["chunk_index"],
                        "content_type": r["content_type"],
                        "model": embedding_config.model,
                        "task": embedding_config.task or "",
                        "dim": len(vec),
                        "vector": list(vec),
                    })

        write_embeddings(rows, base)
        batches_written += 1
        chunks_embedded += len(rows)

        if checkpoint_mgr is not None and not errored:
            doc_ids = _doc_ids(base)
            if doc_ids:
                checkpoint_mgr.update(
                    doc_ids=doc_ids,
                    succeeded=len(rows),
                    failed=0,
                    batch_num=int(base.stem.replace("batch-", "") or 0),
                )

        logger.info("embed_shards: %s embedded %d chunks", base.stem, len(rows))

    return EmbedStageResult(batches_written=batches_written, chunks_embedded=chunks_embedded)


@dataclass
class EmbedPlan:
    """What `embed_shards` would send, counted offline."""

    chunks: int = 0
    tokens: int = 0
    requests: int = 0


def plan_embedding(shard_dir: Path, *, token_counter: TokenCounter | None = None) -> EmbedPlan:
    """Count the chunks, requests and tokens `embed_shards` would send; never calls the API."""
    counter = token_counter or TokenCounter()
    plan = EmbedPlan()
    for base in _batch_bases(shard_dir):
        texts = [r["text"] for r in _chunk_rows(base)]
        plan.chunks += len(texts)
        plan.tokens += sum(counter.count_batch(texts)) if texts else 0
        plan.requests += -(-len(texts) // MAX_TEXTS_PER_REQUEST)
    return plan


def _chunk_rows(base_path: Path) -> list[dict]:
    """Non-empty chunk rows for a batch (text must have a non-whitespace char)."""
    table = read_chunks(base_path)
    if table.num_rows == 0:
        return []
    return [r for r in table.to_pylist() if (r["text"] or "").strip()]


def _doc_ids(base_path: Path) -> list[str]:
    try:
        return list(read_manifest(base_path).column("doc_id").to_pylist())
    except Exception:
        return []


def _all_docs_checkpointed(base_path: Path, mgr: CheckpointManager) -> bool:
    if not embeddings_path_for(base_path).exists():
        return False
    doc_ids = _doc_ids(base_path)
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["EmbedStageResult", "embed_shards"]
