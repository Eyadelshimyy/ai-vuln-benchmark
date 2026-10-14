"""Register-ingest manifest: a glob-free index of standalone-ingest output.

The standalone register ingests (``ingest-abn``, ``ingest-gnaf``,
``ingest-geo``) write one or more Parquet files per source into a flat
output directory, with no index. Consumers then have to glob fragile
patterns (``*_names.parquet`` vs the records files) to tell the outputs
apart. This module consolidates a directory of register outputs into a
single ``manifest.parquet`` — one row per output file — so discovery is
deterministic and glob-free.

It is deliberately separate from ``store/run_manifest.py``: that manifest
is the NLP pipeline's ``source_hash → doc_id`` document table and its
schema is extraction-specific. Register outputs have no extraction
metadata, so they get their own generic schema here. Self-contained and
SDK-free, mirroring ``store/run_manifest.py``'s shape.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

REGISTER_MANIFEST_FILENAME = "manifest.parquet"

# One row per output Parquet file. ``role`` distinguishes the records vs
# names sidecars (ABN) or the table type (G-NAF) without globbing.
REGISTER_MANIFEST_SCHEMA = pa.schema([
    ("source_file", pa.string()),
    ("output_file", pa.string()),
    ("role", pa.string()),
    ("row_count", pa.int64()),
    ("schema_version", pa.string()),
    ("source_md5", pa.string()),
    ("ingested_at_iso", pa.string()),
])


def _decode_meta(raw: dict[bytes, bytes] | None) -> dict[str, str]:
    if not raw:
        return {}
    return {k.decode(): v.decode() for k, v in raw.items()}


def _namespace(meta: dict[str, str]) -> str | None:
    """The footer-key namespace, taken from whichever ``<ns>.source_file``
    key is present (``abn.`` / ``gnaf.`` / ``geospatial.`` / any future
    ingest) — no per-register registry to keep in sync."""
    for key in meta:
        ns, _, field = key.partition(".")
        if field == "source_file":
            return ns
    return None


def _role_for(ns: str | None, meta: dict[str, str]) -> str:
    """Role from footer metadata only: an explicit ``<ns>.role`` (ABN
    records/names), else the G-NAF table type, else the primary records
    output. Filenames are never inspected."""
    if ns is not None:
        role = meta.get(f"{ns}.role") or meta.get(f"{ns}.table_name")
        if role:
            return role
    return "records"


def _manifest_row(output_path: Path) -> dict[str, object]:
    md = pq.read_metadata(str(output_path))
    meta = _decode_meta(md.metadata)
    ns = _namespace(meta)
    source_file = meta.get(f"{ns}.source_file", "") if ns else ""
    schema_version = meta.get(f"{ns}.schema_version", "") if ns else ""
    source_md5 = meta.get(f"{ns}.source_md5") if ns else None
    return {
        "source_file": source_file,
        "output_file": output_path.name,
        "role": _role_for(ns, meta),
        "row_count": md.num_rows,
        "schema_version": schema_version,
        "source_md5": source_md5,
        "ingested_at_iso": datetime.now(UTC).isoformat(),
    }


def write_register_manifest(output_dir: Path) -> Path:
    """Index all register-ingest Parquet outputs in *output_dir*.

    Scans ``*.parquet`` (excluding the manifest itself), reads each file's
    row count and footer metadata, and writes ``manifest.parquet`` into
    *output_dir*. Returns the path written. An empty directory still
    produces an empty, schema-correct file so downstream reads are safe.
    """
    files = sorted(
        p for p in output_dir.glob("*.parquet") if p.name != REGISTER_MANIFEST_FILENAME
    )
    rows = [_manifest_row(p) for p in files]
    table = pa.Table.from_pylist(rows, schema=REGISTER_MANIFEST_SCHEMA)
    target = output_dir / REGISTER_MANIFEST_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(target), compression="zstd", compression_level=3)
    logger.info("Wrote register manifest %s: outputs=%d", target, table.num_rows)
    return target


__all__ = [
    "REGISTER_MANIFEST_FILENAME",
    "REGISTER_MANIFEST_SCHEMA",
    "write_register_manifest",
]
