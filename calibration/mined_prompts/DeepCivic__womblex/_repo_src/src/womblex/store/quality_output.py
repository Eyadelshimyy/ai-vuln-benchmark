"""Parquet IO for the chunk-quality stage (``*.chunk_quality.parquet``).

Self-contained like :mod:`womblex.store.normalise_output`. One row per chunk,
joined back to ``*.chunks.parquet`` on ``(source_hash, chunk_index)``. Carries
ML-readiness shape flags plus cross-batch duplicate cluster ids
(``exact_dup_id`` / ``near_dup_id`` are nullable — null = singleton). Annotation
sidecar only; chunk text is never mutated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import _write_rows
from womblex.store.run_stamp import sidecar_footer

logger = logging.getLogger(__name__)

CHUNK_QUALITY_SUFFIX = ".chunk_quality.parquet"

CHUNK_QUALITY_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("chunk_index", pa.int32()),
    ("content_type", pa.string()),
    ("char_len", pa.int32()),
    ("alpha_frac", pa.float64()),
    ("is_short", pa.bool_()),
    ("boilerplate_flag", pa.bool_()),
    ("exact_dup_id", pa.int32()),   # nullable — null = singleton
    ("near_dup_id", pa.int32()),    # nullable — null = singleton
])


def chunk_quality_path_for(base_path: Path) -> Path:
    """Return ``<base>.chunk_quality.parquet`` sibling for a shard base path."""
    return base_path.parent / f"{base_path.stem}{CHUNK_QUALITY_SUFFIX}"


def write_chunk_quality(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's chunk-quality rows. Empty input -> empty schema-correct file."""
    target = chunk_quality_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, CHUNK_QUALITY_SCHEMA, metadata=sidecar_footer(output_path, "quality"))
    logger.info("Wrote chunk_quality shard %s: rows=%d", target.name, len(rows))
    return target


def read_chunk_quality(path: Path) -> pa.Table:
    """Read chunk quality from a single shard file or a shard-directory glob."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{CHUNK_QUALITY_SUFFIX}"))
        if not shards:
            return pa.table(
                {f.name: pa.array([], type=f.type) for f in CHUNK_QUALITY_SCHEMA},
                schema=CHUNK_QUALITY_SCHEMA,
            )
        return pa.concat_tables([_read_shard(s) for s in shards])
    q_p = p if p.name.endswith(CHUNK_QUALITY_SUFFIX) else chunk_quality_path_for(p)
    return _read_shard(q_p)


def _read_shard(path: Path) -> pa.Table:
    raw = pq.read_table(str(path))
    missing = [f.name for f in CHUNK_QUALITY_SCHEMA if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"chunk_quality shard {path} missing columns {missing}; "
            "schema bump without compat shim?"
        )
    return raw.select([f.name for f in CHUNK_QUALITY_SCHEMA]).cast(CHUNK_QUALITY_SCHEMA)


__all__ = [
    "CHUNK_QUALITY_SCHEMA",
    "CHUNK_QUALITY_SUFFIX",
    "chunk_quality_path_for",
    "read_chunk_quality",
    "write_chunk_quality",
]
