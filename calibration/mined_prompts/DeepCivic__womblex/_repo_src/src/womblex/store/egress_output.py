"""``source_index.parquet`` schema and IO for the egress bundle.

Self-contained, in the manner of the other ``store/*_output.py`` modules.
One row per document in the exported run (not deduplicated by
``source_hash`` — two documents sharing one hash each get their own row,
both naming the same ``raw_key``). ``status`` is drawn from
:mod:`womblex.store.source_resolver`'s vocabulary (``resolved`` /
``hash_mismatch`` / ``not_found`` / ``unsupported_basis``), plus
``upload_failed`` — the bundle builder's own, for a source that resolved but
whose copy to the destination raised.

``raw_key`` is the raw file's path relative to the bundle root
(:func:`raw_key_for`), so a consumer opens it as written; it is null wherever
no file was produced — an unresolved document, or one hashed over record id +
text rather than file bytes. ``ext`` carries its leading dot (``.pdf``), as
``MANIFEST_SCHEMA`` does.

The file carries the exported run's stamp in its footer, stage ``egress``, so
it stays attributable when copied out of the bundle on its own.

This module owns the schema and IO only. Building the bundle — resolving
sources, copying files, writing this index alongside ``egress_manifest.json``
— belongs to the bundle builder, not here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.contract import contract_footer
from womblex.store.run_stamp import RunStamp

logger = logging.getLogger(__name__)

SOURCE_INDEX_FILENAME = "source_index.parquet"
SOURCES_DIRNAME = "sources"
EGRESS_STAGE = "egress"

SOURCE_INDEX_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("doc_id", pa.string()),
    ("filename", pa.string()),
    ("ext", pa.string()),
    ("raw_key", pa.string()),
    ("status", pa.string()),
])


def raw_key_for(source_hash: str, ext: str) -> str:
    """Bundle-root-relative key of a raw file: ``sources/<source_hash><ext>``."""
    return f"{SOURCES_DIRNAME}/{source_hash}{ext}"


def write_source_index(
    rows: list[dict], bundle_dir: Path, *, stamp: RunStamp | None,
) -> Path:
    """Write a bundle's ``source_index.parquet``.

    ``rows`` must match :data:`SOURCE_INDEX_SCHEMA`. *stamp* is the exported
    run's, re-pointed at stage ``egress``; ``None`` writes the file unstamped,
    for a run that cannot be named. Empty input still produces a
    schema-correct file. Returns the path written.
    """
    target = Path(bundle_dir) / SOURCE_INDEX_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    footer = stamp.for_stage(EGRESS_STAGE).footer_metadata() if stamp is not None else {}
    schema = SOURCE_INDEX_SCHEMA.with_metadata({**footer, **contract_footer(target)})
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, str(target), compression="zstd", compression_level=3)
    logger.info("Wrote source index %s: rows=%d", target, table.num_rows)
    return target


def read_source_index(path: Path) -> pa.Table:
    """Read a bundle's source index, from the bundle directory or the file itself."""
    p = Path(path)
    target = p / SOURCE_INDEX_FILENAME if p.is_dir() else p
    raw = pq.read_table(str(target))
    missing = [f.name for f in SOURCE_INDEX_SCHEMA if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"source_index {target} missing columns {missing}; schema bump without compat shim?"
        )
    return raw.select([f.name for f in SOURCE_INDEX_SCHEMA]).cast(SOURCE_INDEX_SCHEMA)


__all__ = [
    "EGRESS_STAGE",
    "SOURCES_DIRNAME",
    "SOURCE_INDEX_FILENAME",
    "SOURCE_INDEX_SCHEMA",
    "raw_key_for",
    "read_source_index",
    "write_source_index",
]
