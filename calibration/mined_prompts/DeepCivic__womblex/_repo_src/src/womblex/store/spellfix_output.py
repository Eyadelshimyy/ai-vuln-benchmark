"""Parquet IO for the OCR-repair stage (``womblex spellfix``).

Two siblings per batch, both self-contained like :mod:`womblex.store.normalise_output`:

- ``*.spellfix_text.parquet`` — the repaired **element-text overlay**. One row
  per ``TEXT_KINDS`` element (``source_hash``, ``elem_order``, ``kind``,
  ``page``, ``text``, ``n_changes``), identical in shape to the normalise
  stage's ``*.normalised_text.parquet`` so both are consumable through the one
  ``process.text_overlay`` resolver. A complete narrative layer (verbatim
  passthrough where nothing fired), joinable on ``(source_hash, elem_order)``.
- ``*.spellfix_corrections.parquet`` — the audit trail: one row per applied
  rewrite (original → corrected), so every change is reviewable and the raw
  element text remains recoverable.

Repair lands at the **element** layer — the shared source both the chunk branch
(``build_chunk_input``) and the enrichment branch (``reassemble_narrative``)
fork from — so the repaired text composes across chunking, embeddings, Kanon-2
enrichment and PII in a single coordinate space (mention offsets stay aligned
with chunk offsets). See ``docs/decisions.md`` "Dictionary-gated OCR repair".
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import _write_rows
from womblex.store.run_stamp import sidecar_footer

logger = logging.getLogger(__name__)

SPELLFIX_TEXT_SUFFIX = ".spellfix_text.parquet"
SPELLFIX_CORRECTIONS_SUFFIX = ".spellfix_corrections.parquet"

# Same shape as NORMALISED_TEXT_SCHEMA so both overlays share one resolver.
SPELLFIX_TEXT_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("elem_order", pa.int32()),
    ("kind", pa.string()),
    ("page", pa.int32()),       # nullable — sources without page semantics
    ("text", pa.string()),      # repaired text (verbatim passthrough if unchanged)
    ("n_changes", pa.int32()),  # rewrites applied to this element
])

SPELLFIX_CORRECTIONS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("elem_order", pa.int32()),
    ("kind", pa.string()),
    ("offset", pa.int32()),       # char offset of the token within the element text
    ("original", pa.string()),
    ("corrected", pa.string()),
    ("method", pa.string()),      # homoglyph | edit1
])


def spellfix_text_path_for(base_path: Path) -> Path:
    return base_path.parent / f"{base_path.stem}{SPELLFIX_TEXT_SUFFIX}"


def spellfix_corrections_path_for(base_path: Path) -> Path:
    return base_path.parent / f"{base_path.stem}{SPELLFIX_CORRECTIONS_SUFFIX}"


def write_spellfix_text(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's repaired element-text rows (match :data:`SPELLFIX_TEXT_SCHEMA`)."""
    target = spellfix_text_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, SPELLFIX_TEXT_SCHEMA,
                metadata=sidecar_footer(output_path, "spellfix"))
    logger.info("Wrote spellfix_text shard %s: rows=%d", target.name, len(rows))
    return target


def write_spellfix_corrections(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's correction-audit rows (match :data:`SPELLFIX_CORRECTIONS_SCHEMA`)."""
    target = spellfix_corrections_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, SPELLFIX_CORRECTIONS_SCHEMA,
                metadata=sidecar_footer(output_path, "spellfix"))
    logger.info("Wrote spellfix corrections shard %s: rows=%d", target.name, len(rows))
    return target


def read_spellfix_text(path: Path) -> pa.Table:
    """Read repaired element text from a single shard file or a directory glob."""
    return _read(path, SPELLFIX_TEXT_SUFFIX, SPELLFIX_TEXT_SCHEMA, spellfix_text_path_for)


def read_spellfix_corrections(path: Path) -> pa.Table:
    """Read correction-audit rows from a single shard file or a directory glob."""
    return _read(
        path, SPELLFIX_CORRECTIONS_SUFFIX, SPELLFIX_CORRECTIONS_SCHEMA,
        spellfix_corrections_path_for,
    )


def _read(path: Path, suffix: str, schema: pa.Schema, path_for) -> pa.Table:
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{suffix}"))
        if not shards:
            return _empty(schema)
        return pa.concat_tables([_read_shard(s, schema) for s in shards])
    target = p if p.name.endswith(suffix) else path_for(p)
    return _read_shard(target, schema)


def _empty(schema: pa.Schema) -> pa.Table:
    return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)


def _read_shard(path: Path, schema: pa.Schema) -> pa.Table:
    raw = pq.read_table(str(path))
    missing = [f.name for f in schema if f.name not in raw.schema.names]
    if missing:
        raise ValueError(f"spellfix shard {path} missing columns {missing}; schema bump?")
    return raw.select([f.name for f in schema]).cast(schema)


__all__ = [
    "SPELLFIX_CORRECTIONS_SCHEMA",
    "SPELLFIX_CORRECTIONS_SUFFIX",
    "SPELLFIX_TEXT_SCHEMA",
    "SPELLFIX_TEXT_SUFFIX",
    "read_spellfix_corrections",
    "read_spellfix_text",
    "spellfix_corrections_path_for",
    "spellfix_text_path_for",
    "write_spellfix_corrections",
    "write_spellfix_text",
]
