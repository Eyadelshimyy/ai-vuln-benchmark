"""Per-stage chunk-quality annotation over an existing shard directory.

Consumes ``*.chunks.parquet`` and writes a ``*.chunk_quality.parquet`` sibling
per batch. Unlike the per-batch-independent normalise stage, duplicate
clustering is **cross-batch**, so this runs as a single global pass: all chunks
are loaded together, ids are computed once, then sliced back to per-batch
sidecars (so the join grain stays per batch). It is a full-run stage — there is
no per-batch resume, because a partial corpus would change the dup clusters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from womblex.config import QualityConfig
from womblex.process.quality import (
    alpha_frac,
    boilerplate_flag,
    char_len,
    compile_patterns,
    exact_dup_ids,
    near_dup_ids,
)
from womblex.store.output import CHUNKS_SUFFIX, chunks_path_for, read_chunks
from womblex.store.quality_output import write_chunk_quality

logger = logging.getLogger(__name__)


def _chunk_bases(shard_dir: Path) -> list[Path]:
    """Canonical batch bases discovered from ``*.chunks.parquet`` siblings."""
    bases = []
    for p in sorted(shard_dir.glob(f"*{CHUNKS_SUFFIX}")):
        if p.name.endswith(".corrupt" + CHUNKS_SUFFIX):
            continue
        stem = p.name[: -len(CHUNKS_SUFFIX)]
        bases.append(shard_dir / f"{stem}.parquet")
    return bases


@dataclass
class QualityStageResult:
    batches_written: int
    chunks_annotated: int
    exact_dup_clusters: int
    near_dup_clusters: int


def quality_shards(shard_dir: Path, config: QualityConfig) -> QualityStageResult:
    """Annotate chunk quality for every batch; write ``*.chunk_quality.parquet``."""
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    bases = [b for b in _chunk_bases(shard_dir) if chunks_path_for(b).exists()]
    if not bases:
        logger.warning("quality_shards: no `*.chunks.parquet` found in %s", shard_dir)
        return QualityStageResult(0, 0, 0, 0)

    # global load (preserve batch grouping for per-batch write-back)
    batch_cols: list[tuple[Path, dict]] = []
    texts: list[str] = []
    for base in bases:
        tbl = read_chunks(chunks_path_for(base))
        cols = {c: tbl.column(c).to_pylist()
                for c in ("source_hash", "chunk_index", "content_type", "text")}
        batch_cols.append((base, cols))
        texts.extend(cols["text"])

    if config.dedup:
        exact = exact_dup_ids(texts)
        near = near_dup_ids(texts, config.minhash_permutations,
                            config.minhash_bands, config.shingle_words)
    else:
        exact = near = [None] * len(texts)

    patterns = compile_patterns(config.boilerplate_patterns)

    batches_written = 0
    off = 0
    for base, cols in batch_cols:
        n = len(cols["text"])
        rows = []
        for j in range(n):
            t = cols["text"][j]
            cl = char_len(t)
            rows.append({
                "source_hash": cols["source_hash"][j],
                "chunk_index": cols["chunk_index"][j],
                "content_type": cols["content_type"][j],
                "char_len": cl,
                "alpha_frac": round(alpha_frac(t), 4),
                "is_short": cl < config.short_chars,
                "boilerplate_flag": boilerplate_flag(t, patterns),
                "exact_dup_id": exact[off + j],
                "near_dup_id": near[off + j],
            })
        write_chunk_quality(rows, base)
        batches_written += 1
        off += n
        logger.info("quality_shards: %s annotated %d chunks", base.stem, n)

    return QualityStageResult(
        batches_written=batches_written,
        chunks_annotated=len(texts),
        exact_dup_clusters=len({i for i in exact if i is not None}),
        near_dup_clusters=len({i for i in near if i is not None}),
    )


__all__ = ["QualityStageResult", "quality_shards"]
