"""Parquet IO for the normalisation stage (``*.normalised_text.parquet``).

Self-contained like :mod:`womblex.store.pii_output`. One row per
text-bearing element (``TEXT_KINDS``), carrying the normalised text — a
drop-in text layer over the narrative half of ``*.elements.parquet``.
Join back on ``(source_hash, elem_order)``; unchanged elements pass
through verbatim with ``n_changes=0`` so the sidecar is a complete
narrative layer, not a sparse diff.

Distinct from the PII stage's ``*.clean_text.parquet`` (masking layer):
this is the *cleaning* layer (whitespace / footer-glyph / typo fixes),
applied pre-chunk. See ``docs/decisions.md`` "Downstream text-cleaning op".
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import _write_rows
from womblex.store.run_stamp import sidecar_footer

logger = logging.getLogger(__name__)

NORMALISED_TEXT_SUFFIX = ".normalised_text.parquet"

NORMALISED_TEXT_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("elem_order", pa.int32()),
    ("kind", pa.string()),
    ("page", pa.int32()),       # nullable — sources without page semantics
    ("text", pa.string()),      # normalised text (verbatim passthrough if unchanged)
    ("n_changes", pa.int32()),  # transform hits applied to this element
])


def normalised_text_path_for(base_path: Path) -> Path:
    """Return ``<base>.normalised_text.parquet`` sibling for a shard base path."""
    return base_path.parent / f"{base_path.stem}{NORMALISED_TEXT_SUFFIX}"


def write_normalised_text(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's normalised-element rows to ``batch-NNNN.normalised_text.parquet``.

    ``rows`` must match :data:`NORMALISED_TEXT_SCHEMA`. Empty input produces an
    empty-but-schema-correct file so downstream readers can glob safely.
    """
    target = normalised_text_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, NORMALISED_TEXT_SCHEMA, metadata=sidecar_footer(output_path, "normalise"))
    logger.info("Wrote normalised_text shard %s: rows=%d", target.name, len(rows))
    return target


def read_normalised_text(path: Path) -> pa.Table:
    """Read normalised text from a single shard file or a shard-directory glob."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{NORMALISED_TEXT_SUFFIX}"))
        if not shards:
            return pa.table(
                {f.name: pa.array([], type=f.type) for f in NORMALISED_TEXT_SCHEMA},
                schema=NORMALISED_TEXT_SCHEMA,
            )
        return pa.concat_tables([_read_normalised_text_shard(s) for s in shards])
    nt_p = p if p.name.endswith(NORMALISED_TEXT_SUFFIX) else normalised_text_path_for(p)
    return _read_normalised_text_shard(nt_p)


def _read_normalised_text_shard(path: Path) -> pa.Table:
    raw = pq.read_table(str(path))
    missing = [f.name for f in NORMALISED_TEXT_SCHEMA if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"normalised_text shard {path} missing columns {missing}; "
            "schema bump without compat shim?"
        )
    return raw.select([f.name for f in NORMALISED_TEXT_SCHEMA]).cast(NORMALISED_TEXT_SCHEMA)


__all__ = [
    "NORMALISED_TEXT_SCHEMA",
    "NORMALISED_TEXT_SUFFIX",
    "normalised_text_path_for",
    "read_normalised_text",
    "write_normalised_text",
]
