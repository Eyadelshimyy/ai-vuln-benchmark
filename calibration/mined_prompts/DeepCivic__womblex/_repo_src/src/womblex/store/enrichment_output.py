"""Parquet output for enrichment results and graph data.

Provides two output schemas:
- **entities.parquet**: Flat entity mentions for fast filtering queries.
- **graph_edges.parquet**: Relationship edges for graph reconstruction.

Document-level enrichment metadata is appended to the existing
documents Parquet via ``enrichment_metadata_columns()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.analyse.graph import DocumentGraph
from womblex.analyse.models import EnrichmentResult
from womblex.store.evidence import TABLE_LAYER
from womblex.store.output import _write_rows
from womblex.store.run_stamp import sidecar_footer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity mention schema (flat, for Parquet-based filtering)
# ---------------------------------------------------------------------------

ENTITY_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("entity_id", pa.string()),
    ("entity_label", pa.string()),       # person | location | term | external_document
    ("name", pa.string()),
    ("entity_type", pa.string()),        # natural, corporate, politic, country, state, etc.
    ("role", pa.string()),               # seller, buyer, other, etc. (persons only)
    ("mention_start", pa.int32()),
    ("mention_end", pa.int32()),
    ("chunk_index", pa.int32()),         # -1 if not mapped to a chunk
    # Which text the offsets index: the narrative under its element-text layer
    # (elements | normalised | spellfix), or ``table_markdown`` for a table
    # enriched on its own. Null on files written before the table pass.
    ("text_layer", pa.string()),
    ("elem_order", pa.int32()),          # table element a table_markdown mention lies in
    ("sheet", pa.string()),              # spreadsheet sheet a table_markdown mention lies in
    # The text of this mention (``name`` is the entity's, the same for every mention).
    # Null on files written before contract 1.5.
    ("mention_text", pa.string()),
])

# Columns an older file may lack; read back as null.
_ENTITY_NULL_BACKFILL = ("text_layer", "elem_order", "sheet", "mention_text")
_META_NULL_BACKFILL = ("table_count",)

#: Meta footer key set when the batch's tables were sent to the enricher — the
#: resume guard's marker, since a null ``table_count`` cannot tell "tables off"
#: from "no tables".
TABLES_ENRICHED_KEY = b"womblex.tables_enriched"

# ---------------------------------------------------------------------------
# Graph edge schema (for relationship reconstruction)
# ---------------------------------------------------------------------------

GRAPH_EDGE_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("source_id", pa.string()),
    ("target_id", pa.string()),
    ("relation", pa.string()),
    ("prop_key", pa.string()),
    ("prop_value", pa.string()),
])

# ---------------------------------------------------------------------------
# Document enrichment metadata schema
# ---------------------------------------------------------------------------

ENRICHMENT_META_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("doc_type_enriched", pa.string()),  # statute | regulation | decision | contract | other
    ("jurisdiction", pa.string()),
    ("title", pa.string()),
    ("segment_count", pa.int32()),
    ("person_count", pa.int32()),
    ("location_count", pa.int32()),
    ("term_count", pa.int32()),
    ("external_doc_count", pa.int32()),
    ("date_count", pa.int32()),
    ("heading_count", pa.int32()),
    ("junk_span_count", pa.int32()),
    # Tables sent to the enricher on their own; null where none were (or the
    # file predates the table pass) and for a document with a table that failed.
    ("table_count", pa.int32()),
])


# ---------------------------------------------------------------------------
# Serialisation: entity mentions
# ---------------------------------------------------------------------------


def _entity_mentions_from_enrichment(
    source_hash: str,
    enrichment: EnrichmentResult,
    chunks: list[object] | None = None,
    *,
    text_layer: str | None = None,
    elem_order: int | None = None,
    sheet: str | None = None,
    id_prefix: str = "",
) -> list[dict[str, Any]]:
    """Extract flat entity mention rows from an enrichment result.

    ``text_layer`` / ``elem_order`` / ``sheet`` say which text the offsets index;
    ``id_prefix`` namespaces entity ids when several results share a
    ``source_hash`` (one per enriched table), which would otherwise all restart
    at the same ids.
    """
    from womblex.analyse.graph import _find_chunks_for_span

    rows: list[dict[str, Any]] = []
    chunk_list = chunks or []

    # Persons
    for per in enrichment.persons:
        name = per.name.decode(enrichment.text)
        for mention in per.mentions:
            chunk_indices = _find_chunks_for_span(mention, chunk_list) if chunk_list else []  # type: ignore[arg-type]
            rows.append({
                "source_hash": source_hash,
                "entity_id": per.id,
                "entity_label": "person",
                "name": name,
                "entity_type": per.type,
                "role": per.role,
                "mention_start": mention.start,
                "mention_end": mention.end,
                "mention_text": mention.decode(enrichment.text),
                "chunk_index": chunk_indices[0] if chunk_indices else -1,
            })

    # Locations
    for loc in enrichment.locations:
        name = loc.name.decode(enrichment.text)
        for mention in loc.mentions:
            chunk_indices = _find_chunks_for_span(mention, chunk_list) if chunk_list else []  # type: ignore[arg-type]
            rows.append({
                "source_hash": source_hash,
                "entity_id": loc.id,
                "entity_label": "location",
                "name": name,
                "entity_type": loc.type,
                "role": "",
                "mention_start": mention.start,
                "mention_end": mention.end,
                "mention_text": mention.decode(enrichment.text),
                "chunk_index": chunk_indices[0] if chunk_indices else -1,
            })

    # Terms
    for term in enrichment.terms:
        name = term.name.decode(enrichment.text)
        for mention in term.mentions:
            chunk_indices = _find_chunks_for_span(mention, chunk_list) if chunk_list else []  # type: ignore[arg-type]
            rows.append({
                "source_hash": source_hash,
                "entity_id": term.id,
                "entity_label": "term",
                "name": name,
                "entity_type": "",
                "role": "",
                "mention_start": mention.start,
                "mention_end": mention.end,
                "mention_text": mention.decode(enrichment.text),
                "chunk_index": chunk_indices[0] if chunk_indices else -1,
            })

    # External documents
    for exd in enrichment.external_documents:
        name = exd.name.decode(enrichment.text)
        for mention in exd.mentions:
            chunk_indices = _find_chunks_for_span(mention, chunk_list) if chunk_list else []  # type: ignore[arg-type]
            rows.append({
                "source_hash": source_hash,
                "entity_id": exd.id,
                "entity_label": "external_document",
                "name": name,
                "entity_type": exd.type,
                "role": "",
                "mention_start": mention.start,
                "mention_end": mention.end,
                "mention_text": mention.decode(enrichment.text),
                "chunk_index": chunk_indices[0] if chunk_indices else -1,
            })

    for row in rows:
        row.update(
            entity_id=f"{id_prefix}{row['entity_id']}",
            text_layer=text_layer, elem_order=elem_order, sheet=sheet,
        )
    return rows


# ---------------------------------------------------------------------------
# Serialisation: graph edges
# ---------------------------------------------------------------------------


def _graph_edges_to_rows(
    source_hash: str,
    graph: DocumentGraph,
) -> list[dict[str, Any]]:
    """Flatten graph edges to rows, one per edge-property pair."""
    rows: list[dict[str, Any]] = []
    for edge in graph.edges:
        if edge.properties:
            for key, value in edge.properties.items():
                rows.append({
                    "source_hash": source_hash,
                    "source_id": edge.source,
                    "target_id": edge.target,
                    "relation": edge.relation,
                    "prop_key": key,
                    "prop_value": str(value) if value is not None else "",
                })
        else:
            rows.append({
                "source_hash": source_hash,
                "source_id": edge.source,
                "target_id": edge.target,
                "relation": edge.relation,
                "prop_key": "",
                "prop_value": "",
            })
    return rows


# ---------------------------------------------------------------------------
# Serialisation: enrichment metadata
# ---------------------------------------------------------------------------


def _enrichment_meta_row(
    source_hash: str,
    enrichment: EnrichmentResult,
) -> dict[str, Any]:
    """Build a single enrichment metadata row."""
    title = enrichment.title.decode(enrichment.text) if enrichment.title else ""
    return {
        "source_hash": source_hash,
        "doc_type_enriched": enrichment.type,
        "jurisdiction": enrichment.jurisdiction or "",
        "title": title,
        "segment_count": len(enrichment.segments),
        "person_count": len(enrichment.persons),
        "location_count": len(enrichment.locations),
        "term_count": len(enrichment.terms),
        "external_doc_count": len(enrichment.external_documents),
        "date_count": len(enrichment.dates),
        "heading_count": len(enrichment.headings),
        "junk_span_count": len(enrichment.junk),
        "table_count": None,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
#
# The three whole-corpus E2E writers below predate the sharded layout and write
# whatever document identity their caller supplies into ``source_hash``. Their
# only caller, ``operations.write_batch_enrichment``, supplies a ``doc_id`` and
# has had no caller of its own since in-batch enrichment was removed in 0.5.10
# — which is why they are a fossil noted for retirement rather than a second
# supported shape. Everything a run actually writes goes through the sharded
# writers further down, which supply the source_hash the name states.


def write_entity_mentions(
    results: list[tuple[str, EnrichmentResult, list[object] | None]],
    output_path: Path,
) -> Path:
    """Write entity mentions to a Parquet file.

    Args:
        results: List of (identity, EnrichmentResult, chunks) tuples.
        output_path: Destination Parquet file path.

    Returns:
        The output path written.
    """
    all_rows: list[dict[str, Any]] = []
    for ident, enrichment, chunks in results:
        all_rows.extend(_entity_mentions_from_enrichment(ident, enrichment, chunks))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(all_rows, output_path, ENTITY_SCHEMA, role="enrichment_entities")
    logger.info("Wrote %d entity mentions to %s", len(all_rows), output_path)
    return output_path


def write_graph_edges(
    graphs: list[tuple[str, DocumentGraph]],
    output_path: Path,
) -> Path:
    """Write graph edges to a Parquet file.

    Args:
        graphs: List of (identity, DocumentGraph) tuples.
        output_path: Destination Parquet file path.

    Returns:
        The output path written.
    """
    all_rows: list[dict[str, Any]] = []
    for ident, graph in graphs:
        all_rows.extend(_graph_edges_to_rows(ident, graph))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(all_rows, output_path, GRAPH_EDGE_SCHEMA, role="graph_edges")
    logger.info("Wrote %d graph edges to %s", len(all_rows), output_path)
    return output_path


def write_enrichment_metadata(
    results: list[tuple[str, EnrichmentResult]],
    output_path: Path,
) -> Path:
    """Write enrichment metadata to a Parquet file.

    Args:
        results: List of (identity, EnrichmentResult) tuples.
        output_path: Destination Parquet file path.

    Returns:
        The output path written.
    """
    rows = [_enrichment_meta_row(ident, enrichment) for ident, enrichment in results]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, output_path, ENRICHMENT_META_SCHEMA, role="enrichment_meta")
    logger.info("Wrote %d enrichment metadata rows to %s", len(rows), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Per-batch sharded siblings (enrich stage, mirrors store.output chunk siblings)
# ---------------------------------------------------------------------------
#
# The sharded enrich stage (``analyse.enrich_stage``) writes one sibling
# parquet per extraction batch, joinable to ``*.elements.parquet`` /
# ``*.chunks.parquet`` on ``source_hash`` — the column every sidecar in the
# library names its document identity with, these three included since 0.6.0.
# They carried the same values under ``document_id`` before that; see
# ``LEGACY_HASH_COLUMN`` for the reader shim and its removal release.

ENRICHMENT_ENTITIES_SUFFIX = ".enrichment_entities.parquet"
ENRICHMENT_META_SUFFIX = ".enrichment_meta.parquet"
GRAPH_EDGES_SUFFIX = ".graph_edges.parquet"


def enrichment_entities_path_for(base_path: Path) -> Path:
    """Return ``<base>.enrichment_entities.parquet`` sibling for a shard base."""
    return base_path.parent / f"{base_path.stem}{ENRICHMENT_ENTITIES_SUFFIX}"


def graph_edges_path_for(base_path: Path) -> Path:
    """Return ``<base>.graph_edges.parquet`` sibling for a shard base."""
    return base_path.parent / f"{base_path.stem}{GRAPH_EDGES_SUFFIX}"


def enrichment_meta_path_for(base_path: Path) -> Path:
    """Return ``<base>.enrichment_meta.parquet`` sibling for a shard base."""
    return base_path.parent / f"{base_path.stem}{ENRICHMENT_META_SUFFIX}"


@dataclass(frozen=True)
class TableEnrichment:
    """One table's enrichment result, with the handle that locates the table."""

    source_hash: str
    ordinal: int           # position among the document's tables; namespaces entity ids
    elem_order: int | None
    sheet: str | None
    result: EnrichmentResult


def write_enrichment_entities_shard(
    results: list[tuple[str, EnrichmentResult]], base_path: Path,
    *, text_layer: str | None = None, tables: list[TableEnrichment] | None = None,
) -> Path:
    """Write a batch's entity mentions to ``<base>.enrichment_entities.parquet``.

    ``results`` is ``(source_hash, EnrichmentResult)`` over the document
    narrative, whose offsets index the ``text_layer`` narrative. ``tables`` are
    per-table results, whose offsets index that table's markdown: their entity
    ids are namespaced ``t<ordinal>:`` and they carry ``TABLE_LAYER``. Empty
    input produces an empty-but-schema-correct file so downstream globs are safe.
    """
    rows: list[dict[str, Any]] = []
    for source_hash, enrichment in results:
        rows.extend(_entity_mentions_from_enrichment(
            source_hash, enrichment, None, text_layer=text_layer))
    for t in tables or []:
        rows.extend(_entity_mentions_from_enrichment(
            t.source_hash, t.result, None, text_layer=TABLE_LAYER,
            elem_order=t.elem_order, sheet=t.sheet, id_prefix=f"t{t.ordinal}:"))
    target = enrichment_entities_path_for(base_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_enrichment_rows(rows, target, ENTITY_SCHEMA,
                          metadata=sidecar_footer(base_path, "enrich"))
    logger.info("Wrote enrichment entities shard %s: rows=%d", target.name, len(rows))
    return target


def write_enrichment_meta_shard(
    results: list[tuple[str, EnrichmentResult]], base_path: Path,
    *, table_counts: dict[str, int] | None = None,
) -> Path:
    """Write a batch's per-doc enrichment metadata to ``<base>.enrichment_meta.parquet``.

    ``table_counts`` is tables sent to the enricher per document, ``None`` when
    the table pass was off. A document with tables but no narrative result
    still gets a row (all counts zero), so the table coverage is not lost.
    """
    counts = table_counts or {}
    rows = [_enrichment_meta_row(src, enr) for src, enr in results]
    for r in rows:
        r["table_count"] = counts.get(r["source_hash"])
    seen = {r["source_hash"] for r in rows}
    for src, n in counts.items():
        if src not in seen:
            rows.append({f.name: 0 for f in ENRICHMENT_META_SCHEMA} | {
                "source_hash": src, "doc_type_enriched": "", "jurisdiction": "",
                "title": "", "table_count": n,
            })
    footer = sidecar_footer(base_path, "enrich") or {}
    if table_counts is not None:
        footer[TABLES_ENRICHED_KEY] = b"true"
    target = enrichment_meta_path_for(base_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_enrichment_rows(rows, target, ENRICHMENT_META_SCHEMA, metadata=footer)
    return target


def write_graph_edges_shard(
    graphs: list[tuple[str, DocumentGraph]], base_path: Path,
) -> Path:
    """Write a batch's graph edges to ``<base>.graph_edges.parquet``.

    ``graphs`` is ``(source_hash, DocumentGraph)``. Empty input produces an
    empty-but-schema-correct file so downstream globs are safe.
    """
    rows: list[dict[str, Any]] = []
    for source_hash, graph in graphs:
        rows.extend(_graph_edges_to_rows(source_hash, graph))
    target = graph_edges_path_for(base_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_enrichment_rows(rows, target, GRAPH_EDGE_SCHEMA,
                          metadata=sidecar_footer(base_path, "enrich"))
    logger.info("Wrote graph edges shard %s: rows=%d", target.name, len(rows))
    return target


def write_enrichment_entities_rows(rows: list[dict[str, Any]], base_path: Path) -> Path:
    """Write pre-built ENTITY_SCHEMA rows to ``<base>.enrichment_entities.parquet``.

    Lower-level than :func:`write_enrichment_entities_shard` (which derives rows
    from an ``EnrichmentResult``): the graph-edge refresh already holds the
    mention rows and only needs to rewrite them with ``chunk_index`` populated.
    """
    target = enrichment_entities_path_for(base_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_enrichment_rows(rows, target, ENTITY_SCHEMA,
                          metadata=sidecar_footer(base_path, "graph-refresh"))
    return target


def write_graph_edges_rows(rows: list[dict[str, Any]], base_path: Path) -> Path:
    """Write pre-built GRAPH_EDGE_SCHEMA rows to ``<base>.graph_edges.parquet``.

    Lower-level than :func:`write_graph_edges_shard` (which flattens a
    ``DocumentGraph``): the graph-edge refresh rewrites the existing edge set
    with refreshed mention→chunk edges appended.
    """
    target = graph_edges_path_for(base_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_enrichment_rows(rows, target, GRAPH_EDGE_SCHEMA,
                          metadata=sidecar_footer(base_path, "graph-refresh"))
    return target


def read_graph_edges(path: Path) -> pa.Table:
    """Read graph edges from a single sibling file or a shard-dir glob."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{GRAPH_EDGES_SUFFIX}"))
        if not shards:
            return pa.table(
                {f.name: pa.array([], type=f.type) for f in GRAPH_EDGE_SCHEMA},
                schema=GRAPH_EDGE_SCHEMA,
            )
        return pa.concat_tables([_read_enrichment_shard(s, GRAPH_EDGE_SCHEMA) for s in shards])
    target = p if p.name.endswith(GRAPH_EDGES_SUFFIX) else graph_edges_path_for(p)
    return _read_enrichment_shard(target, GRAPH_EDGE_SCHEMA)


def read_enrichment_entities(path: Path) -> pa.Table:
    """Read entity mentions from a single sibling file or a shard-dir glob."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{ENRICHMENT_ENTITIES_SUFFIX}"))
        if not shards:
            return pa.table(
                {f.name: pa.array([], type=f.type) for f in ENTITY_SCHEMA},
                schema=ENTITY_SCHEMA,
            )
        return pa.concat_tables([_read_enrichment_shard(s, ENTITY_SCHEMA) for s in shards])
    target = p if p.name.endswith(ENRICHMENT_ENTITIES_SUFFIX) else enrichment_entities_path_for(p)
    return _read_enrichment_shard(target, ENTITY_SCHEMA)


def _write_enrichment_rows(
    rows: list[dict[str, Any]],
    path: Path,
    schema: pa.Schema,
    *,
    metadata: dict[bytes, bytes] | None = None,
) -> None:
    _write_rows(rows, path, schema, metadata=metadata)


#: The name these three sidecars gave their document identity column before
#: 0.6.0. Values were byte-identical to the ``source_hash`` every other sidecar
#: joins on, so a shard written under the old name reads back through
#: :func:`_read_enrichment_shard` as if it had been written under the new one.
#: **Removed in 0.7.0** — from then a pre-0.6.0 shard is a missing-column error
#: and must be re-enriched.
LEGACY_HASH_COLUMN = "document_id"


def _read_enrichment_shard(path: Path, schema: pa.Schema) -> pa.Table:
    raw = pq.read_table(str(path))
    if "source_hash" not in raw.schema.names and LEGACY_HASH_COLUMN in raw.schema.names:
        raw = raw.rename_columns([
            "source_hash" if n == LEGACY_HASH_COLUMN else n for n in raw.schema.names
        ])
    backfill = _ENTITY_NULL_BACKFILL if schema is ENTITY_SCHEMA else (
        _META_NULL_BACKFILL if schema is ENRICHMENT_META_SCHEMA else ())
    for name in backfill:
        if name not in raw.schema.names:
            raw = raw.append_column(name, pa.nulls(raw.num_rows, schema.field(name).type))
    missing = [f.name for f in schema if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"enrichment shard {path} missing columns {missing}; schema bump without compat shim?"
        )
    return raw.select([f.name for f in schema]).cast(schema)
