"""Per-batch raw-ILGS-Document sidecar (``*.enrichment_doc.parquet``).

Self-contained store module (mirrors ``store/enrichment_output.py`` /
``store/normalise_output.py``). The enrich stage optionally persists the
*raw* Isaacus ILGS Document — the full segment tree, headings and span set —
so the chunk stage can reuse it for semchunk-4 AI chunking instead of
re-enriching the same narrative (the single-enrichment reuse design in
``docs/decisions.md``). The flattened ``*.enrichment_entities.parquet`` /
``*.enrichment_meta.parquet`` siblings are lossy and cannot serve this.

The Document is a Stainless-generated pydantic model that round-trips
losslessly through ``model_dump_json()`` ↔ ``model_validate_json()``; this
module only stores/returns the JSON string and never imports the isaacus
SDK (which is a core dependency in any case), so it loads even on a partial
install where the SDK is missing. The ``text_source``
column records which cleaning overlay the persisted ``document.text`` was
reassembled under — audit metadata and a cheap reuse pre-filter; the
authoritative reuse guard is byte-identity of ``document.text`` against the
chunk stage's freshly reassembled narrative.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import _write_rows
from womblex.store.run_stamp import sidecar_footer

logger = logging.getLogger(__name__)

ENRICHMENT_DOC_SUFFIX = ".enrichment_doc.parquet"

ENRICHMENT_DOC_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("text_source", pa.string()),   # elements | normalised | spellfix (provenance)
    ("document_json", pa.string()),  # Document.model_dump_json()
])


def enrichment_doc_path_for(base_path: Path) -> Path:
    """Return ``<base>.enrichment_doc.parquet`` sibling for a shard base."""
    return base_path.parent / f"{base_path.stem}{ENRICHMENT_DOC_SUFFIX}"


def write_enrichment_doc_shard(
    rows: list[tuple[str, str, str]], base_path: Path,
) -> Path:
    """Write a batch's raw Documents to ``<base>.enrichment_doc.parquet``.

    ``rows`` is ``(source_hash, text_source, document_json)``. Empty input
    still writes a schema-correct empty file so downstream globs are safe.
    """
    target = enrichment_doc_path_for(base_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = [
        {"source_hash": sh, "text_source": ts, "document_json": dj}
        for sh, ts, dj in rows
    ]
    _write_rows(records, target, ENRICHMENT_DOC_SCHEMA,
                metadata=sidecar_footer(base_path, "enrich"))
    logger.info("Wrote enrichment doc shard %s: rows=%d", target.name, len(records))
    return target


def read_enrichment_docs(base_path: Path) -> dict[str, tuple[str, str]]:
    """Return ``{source_hash: (text_source, document_json)}`` for a shard base.

    Missing sidecar → empty dict (the chunk stage then falls back to
    self-enrich for that batch). Accepts either a shard base path or the
    sidecar path itself.
    """
    p = Path(base_path)
    target = p if p.name.endswith(ENRICHMENT_DOC_SUFFIX) else enrichment_doc_path_for(p)
    if not target.exists():
        return {}
    raw = pq.read_table(str(target))
    missing = [f.name for f in ENRICHMENT_DOC_SCHEMA if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"enrichment doc shard {target} missing columns {missing}; "
            "schema bump without compat shim?"
        )
    out: dict[str, tuple[str, str]] = {}
    for row in raw.to_pylist():
        out[row["source_hash"]] = (row["text_source"] or "", row["document_json"] or "")
    return out


__all__ = [
    "ENRICHMENT_DOC_SCHEMA",
    "ENRICHMENT_DOC_SUFFIX",
    "enrichment_doc_path_for",
    "read_enrichment_docs",
    "write_enrichment_doc_shard",
]
