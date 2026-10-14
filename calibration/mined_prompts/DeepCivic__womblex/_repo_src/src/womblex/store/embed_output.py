"""``*.embeddings.parquet`` schema + IO — the embed stage's sidecar.

Self-contained, in the shape ``store/pii_output.py`` and
``store/money_output.py`` established: one module per stage sidecar, holding
its schema, its path convention and its reader/writer. It lived in
``store/output.py``, which is the *extraction* writer — a different concern,
and the reason that file outgrew its line budget.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import _write_rows
from womblex.store.run_stamp import sidecar_footer

logger = logging.getLogger(__name__)

EMBEDDINGS_SUFFIX = ".embeddings.parquet"

# Embedding sidecar (embed stage). One row per embedded chunk, joinable to
# `*.chunks.parquet` on (source_hash, chunk_index, content_type). The vector
# is a variable-length float32 list; `dim` is stored explicitly for readers.
EMBEDDINGS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("chunk_index", pa.int32()),
    ("content_type", pa.string()),
    ("model", pa.string()),
    ("task", pa.string()),
    ("dim", pa.int32()),
    ("vector", pa.list_(pa.float32())),
])


def embeddings_path_for(base_path: Path) -> Path:
    """Return ``<base>.embeddings.parquet`` sibling for a shard base path."""
    return base_path.parent / f"{base_path.stem}{EMBEDDINGS_SUFFIX}"


def write_embeddings(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's embedding rows to ``batch-NNNN.embeddings.parquet``.

    ``rows`` must match :data:`EMBEDDINGS_SCHEMA`. Empty input produces an
    empty-but-schema-correct file so downstream readers can glob safely.
    """
    target = embeddings_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, EMBEDDINGS_SCHEMA, metadata=sidecar_footer(output_path, "embed"))
    logger.info("Wrote embeddings shard %s: rows=%d", target.name, len(rows))
    return target


def read_embeddings(path: Path) -> pa.Table:
    """Read embeddings from a single shard file or a shard-directory glob."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{EMBEDDINGS_SUFFIX}"))
        if not shards:
            return pa.table(
                {f.name: pa.array([], type=f.type) for f in EMBEDDINGS_SCHEMA},
                schema=EMBEDDINGS_SCHEMA,
            )
        return pa.concat_tables([_read_embeddings_shard(s) for s in shards])
    emb_p = p if p.name.endswith(EMBEDDINGS_SUFFIX) else embeddings_path_for(p)
    return _read_embeddings_shard(emb_p)


def _read_embeddings_shard(path: Path) -> pa.Table:
    raw = pq.read_table(str(path))
    missing = [f.name for f in EMBEDDINGS_SCHEMA if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"embeddings shard {path} missing columns {missing}; schema bump without compat shim?"
        )
    return raw.select([f.name for f in EMBEDDINGS_SCHEMA]).cast(EMBEDDINGS_SCHEMA)


__all__ = [
    "EMBEDDINGS_SCHEMA",
    "EMBEDDINGS_SUFFIX",
    "embeddings_path_for",
    "read_embeddings",
    "write_embeddings",
]
