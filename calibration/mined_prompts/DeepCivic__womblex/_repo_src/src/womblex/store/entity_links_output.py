"""``*.entity_links.parquet`` schema + IO — the link stage's sidecar.

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

from womblex.store.evidence import DOCUMENT, EVIDENCE_FIELDS, backfill_evidence
from womblex.store.normalise_output import NORMALISED_TEXT_SUFFIX
from womblex.store.output import ELEMENTS_SUFFIX, _write_rows
from womblex.store.run_stamp import sidecar_footer
from womblex.store.spellfix_output import SPELLFIX_TEXT_SUFFIX

logger = logging.getLogger(__name__)

ENTITY_LINKS_SUFFIX = ".entity_links.parquet"

# Entity-link sidecar (link stage). Generic by design: a link is a resolved
# (or attempted) attribution of a document mention to a canonical reference
# entity. ``entity_type`` is a free value (e.g. ``provider`` / ``service``) —
# there are NO domain-specific columns. Rows are at mention/span grain; the
# per-doc view is derived (see :func:`read_entity_links`). Unmatched candidates
# are written with ``matched=False`` so they stay inspectable.
ENTITY_LINKS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("candidate_text", pa.string()),    # surface form extracted from the doc
    ("candidate_kind", pa.string()),    # enrichment kind that produced it (e.g. corporate, address)
    *EVIDENCE_FIELDS,                   # where the mention is in the source
    ("mention_text", pa.string()),      # the enricher's text for the mention (null if not given)
    ("entity_id", pa.string()),         # canonical reference id (e.g. PR-/SE-); "" if unmatched
    ("entity_type", pa.string()),       # reference entity type (provider | service | ...)
    ("canonical_name", pa.string()),    # reference display name; "" if unmatched
    ("parent_entity_id", pa.string()),  # hierarchy FK (e.g. service -> provider); "" if none
    ("confidence", pa.float32()),
    ("match_method", pa.string()),      # address_exact | name_fuzzy | alias | unmatched
    ("matched", pa.bool_()),
])


def entity_links_path_for(base_path: Path) -> Path:
    """Return ``<base>.entity_links.parquet`` sibling for a shard base path."""
    return base_path.parent / f"{base_path.stem}{ENTITY_LINKS_SUFFIX}"


def write_entity_links(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's entity-link rows to ``batch-NNNN.entity_links.parquet``.

    ``rows`` must match :data:`ENTITY_LINKS_SCHEMA`. Empty input produces an
    empty-but-schema-correct file so downstream readers can glob safely.
    """
    target = entity_links_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, ENTITY_LINKS_SCHEMA, metadata=sidecar_footer(output_path, "link"))
    logger.info("Wrote entity_links shard %s: rows=%d", target.name, len(rows))
    return target


def read_entity_links(path: Path, *, grain: str = "span") -> pa.Table:
    """Read entity links from a single shard file or a shard-directory glob.

    ``grain='span'`` returns the persisted mention-level rows verbatim.
    ``grain='doc'`` returns a derived doc-level view: one row per
    ``(source_hash, entity_id)`` matched attribution, with mention count and
    max confidence — computed on read, never persisted (no second file to
    drift). Unmatched rows are excluded from the doc view.
    """
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{ENTITY_LINKS_SUFFIX}"))
        if not shards:
            table = pa.table(
                {f.name: pa.array([], type=f.type) for f in ENTITY_LINKS_SCHEMA},
                schema=ENTITY_LINKS_SCHEMA,
            )
        else:
            table = pa.concat_tables([_read_entity_links_shard(s) for s in shards])
    else:
        links_p = p if p.name.endswith(ENTITY_LINKS_SUFFIX) else entity_links_path_for(p)
        table = _read_entity_links_shard(links_p)

    if grain == "span":
        return table
    if grain == "doc":
        return _entity_links_doc_view(table)
    raise ValueError(f"grain must be 'span' or 'doc', got {grain!r}")


def _reanchor_legacy(raw: pa.Table, path: Path) -> pa.Table:
    """Place an older file's mentions in the source where that is certain.

    A contract 1.x link's `mention_start` / `mention_end` index the enrichment
    text, which is the verbatim narrative unless a text layer was selected. With
    no overlay sidecar beside the file the layer can only have been `elements`, so
    the offsets locate the mention exactly (`span`, unchecked: the file has no
    mention text). A shard with an overlay, or with no elements, stays
    document-located.
    """
    if "mention_start" not in raw.schema.names:
        return raw
    # Lazy: re-anchoring needs the elements, which only the process layer reads.
    from womblex.process.evidence import EvidenceIndexes

    base = path.parent / f"{path.name.removesuffix(ENTITY_LINKS_SUFFIX)}.parquet"
    if not (path.parent / f"{base.stem}{ELEMENTS_SUFFIX}").exists() or any(
            (path.parent / f"{base.stem}{s}").exists()
            for s in (NORMALISED_TEXT_SUFFIX, SPELLFIX_TEXT_SUFFIX)):
        return raw
    indexes = EvidenceIndexes(base)
    rows = raw.to_pylist()
    for r in rows:
        index = indexes.get(r["source_hash"])
        start, end = r["mention_start"], r["mention_end"]
        ref = index.narrative_ref(start, end) if index and start is not None and 0 <= start < end else None
        if ref:
            r.update({k: v for k, v in ref.items() if k != "evidence_check"})
    return pa.Table.from_pylist(rows, schema=raw.schema)


def _read_entity_links_shard(path: Path) -> pa.Table:
    # Contract 1.x links carried `mention_start` / `mention_end` into the enrichment
    # text; the evidence columns replace them. A file with no elements beside it
    # reads as document-located; see `_reanchor_legacy`.
    raw = _reanchor_legacy(backfill_evidence(pq.read_table(str(path)), level=DOCUMENT), path)
    if "mention_text" not in raw.schema.names:
        raw = raw.append_column("mention_text", pa.nulls(raw.num_rows, pa.string()))
    missing = [f.name for f in ENTITY_LINKS_SCHEMA if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"entity_links shard {path} missing columns {missing}; schema bump without compat shim?"
        )
    return raw.select([f.name for f in ENTITY_LINKS_SCHEMA]).cast(ENTITY_LINKS_SCHEMA)


def _entity_links_doc_view(table: pa.Table) -> pa.Table:
    """Group matched span rows to one row per (source_hash, entity_id)."""
    agg: dict[tuple[str, str], dict] = {}
    rows = table.to_pylist()
    for r in rows:
        if not r["matched"]:
            continue
        key = (r["source_hash"], r["entity_id"])
        cur = agg.get(key)
        if cur is None:
            agg[key] = {
                "source_hash": r["source_hash"],
                "entity_id": r["entity_id"],
                "entity_type": r["entity_type"],
                "canonical_name": r["canonical_name"],
                "parent_entity_id": r["parent_entity_id"],
                "mention_count": 1,
                "max_confidence": r["confidence"],
            }
        else:
            cur["mention_count"] += 1
            cur["max_confidence"] = max(cur["max_confidence"], r["confidence"])
    doc_schema = pa.schema([
        ("source_hash", pa.string()),
        ("entity_id", pa.string()),
        ("entity_type", pa.string()),
        ("canonical_name", pa.string()),
        ("parent_entity_id", pa.string()),
        ("mention_count", pa.int32()),
        ("max_confidence", pa.float32()),
    ])
    out = list(agg.values())
    if not out:
        return pa.table({f.name: pa.array([], type=f.type) for f in doc_schema}, schema=doc_schema)
    return pa.Table.from_pylist(out, schema=doc_schema)


__all__ = [
    "ENTITY_LINKS_SCHEMA",
    "ENTITY_LINKS_SUFFIX",
    "entity_links_path_for",
    "read_entity_links",
    "write_entity_links",
]
