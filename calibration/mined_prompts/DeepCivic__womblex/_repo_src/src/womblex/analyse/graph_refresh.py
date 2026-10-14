"""Offline graph-edge refresh: rebuild mention→chunk edges after chunking.

The corpus asset chunks *after* enriching (AI chunking reuses the persisted
ILGS Document), so the ``*.graph_edges.parquet`` written at enrich time cannot
carry mention→chunk edges — no chunks existed yet — and the
``*.enrichment_entities.parquet`` mentions all have ``chunk_index = -1``. This
stage closes that gap once chunks exist.

Both inputs carry character offsets into the same reassembled narrative
(entity mention spans; chunk ``start_char``/``end_char``), so the refresh is
**deterministic and API-free**: for every entity mention it finds the
overlapping narrative chunk(s) and

- rewrites the entities sidecar with ``chunk_index`` populated (first
  overlapping chunk, matching the enrich-time semantics), and
- rewrites the graph-edges sidecar, replacing any ``mentioned_in`` edges with
  freshly computed entity→chunk edges (all other edges — hierarchy,
  citations, cross-references — are preserved untouched).

Idempotent: recomputing from the same offsets yields the same result, so a
re-run (or a resume) is safe. Steps 3 and 5 of the product consume the
refreshed graph. Only entity mentions are (re)linked — segment/cross-reference
chunk edges need the full ``EnrichmentResult`` and are out of scope here.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from womblex.process.chunk_stage import _batch_bases
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_output import (
    graph_edges_path_for,
    read_enrichment_entities,
    read_graph_edges,
    write_enrichment_entities_rows,
    write_graph_edges_rows,
)
from womblex.store.evidence import TABLE_LAYER
from womblex.store.output import chunks_path_for, read_chunks, read_manifest

logger = logging.getLogger(__name__)

_MENTION_RELATION = "mentioned_in"


@dataclass
class GraphRefreshResult:
    batches_written: int
    docs_refreshed: int
    edges_added: int


def refresh_graph_edges(
    shard_dir: Path,
    *,
    checkpoint_mgr: CheckpointManager | None = None,
) -> GraphRefreshResult:
    """Refresh mention→chunk edges for every batch in ``shard_dir``.

    Needs both ``*.enrichment_entities.parquet`` and ``*.chunks.parquet``
    siblings. Batches missing either are skipped (logged). When
    ``checkpoint_mgr`` is provided, batches whose docs are all checkpointed
    are skipped on resume.
    """
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("refresh_graph_edges: no batches found in %s", shard_dir)
        return GraphRefreshResult(0, 0, 0)

    batches_written = 0
    docs_refreshed = 0
    edges_added = 0

    for base in bases:
        if checkpoint_mgr is not None and _all_docs_checkpointed(base, checkpoint_mgr):
            logger.info("refresh_graph_edges: skipping %s (all docs checkpointed)", base.stem)
            continue

        entities_table = read_enrichment_entities(base)
        if entities_table.num_rows == 0 or not chunks_path_for(base).exists():
            logger.info(
                "refresh_graph_edges: %s has no entities or no chunks sidecar — skipping",
                base.stem,
            )
            continue

        chunks_by_hash = _narrative_chunks(base)
        entity_rows = entities_table.to_pylist()
        refreshed_rows, mention_edges, docs = _relink_mentions(entity_rows, chunks_by_hash)

        write_enrichment_entities_rows(refreshed_rows, base)
        _rewrite_graph_edges(base, mention_edges)

        batches_written += 1
        docs_refreshed += docs
        edges_added += len(mention_edges)

        if checkpoint_mgr is not None:
            doc_ids = _doc_ids(base)
            if doc_ids:
                checkpoint_mgr.update(
                    doc_ids=doc_ids, succeeded=docs, failed=0,
                    batch_num=int(base.stem.replace("batch-", "") or 0),
                )

        logger.info(
            "refresh_graph_edges: %s refreshed %d docs, %d mention edges",
            base.stem, docs, len(mention_edges),
        )

    return GraphRefreshResult(
        batches_written=batches_written,
        docs_refreshed=docs_refreshed,
        edges_added=edges_added,
    )


# ---------------------------------------------------------------------------
# Overlap → refreshed rows + edges
# ---------------------------------------------------------------------------


def _relink_mentions(
    entity_rows: list[dict],
    chunks_by_hash: dict[str, list[tuple[int, int, int]]],
) -> tuple[list[dict], list[dict], int]:
    """Return (refreshed entity rows, new mention→chunk edge rows, doc count).

    Each entity row is one mention; ``chunk_index`` is set to its first
    overlapping narrative chunk (-1 when none / no chunks for the doc). One
    ``mentioned_in`` edge is emitted per (mention, overlapping chunk), matching
    :func:`womblex.analyse.graph.build_document_graph`.
    """
    refreshed: list[dict] = []
    edges: list[dict] = []
    docs_with_edges: set[str] = set()

    for row in entity_rows:
        if row.get("text_layer") == TABLE_LAYER:
            # Offsets index a table's own markdown, not the narrative chunks are cut from.
            refreshed.append(dict(row))
            continue
        source_hash = row["source_hash"]
        chunks = chunks_by_hash.get(source_hash, [])
        overlaps = _overlapping_chunks(row["mention_start"], row["mention_end"], chunks)

        new_row = dict(row)
        new_row["chunk_index"] = overlaps[0] if overlaps else -1
        refreshed.append(new_row)

        entity_node = f"{source_hash}:{row['entity_id']}"
        for ci in overlaps:
            chunk_node = f"{source_hash}:chunk:{ci}"
            edges.append({
                "source_hash": source_hash, "source_id": entity_node,
                "target_id": chunk_node, "relation": _MENTION_RELATION,
                "prop_key": "start", "prop_value": str(row["mention_start"]),
            })
            edges.append({
                "source_hash": source_hash, "source_id": entity_node,
                "target_id": chunk_node, "relation": _MENTION_RELATION,
                "prop_key": "end", "prop_value": str(row["mention_end"]),
            })
            docs_with_edges.add(source_hash)

    return refreshed, edges, len(docs_with_edges)


def _overlapping_chunks(
    start: int, end: int, chunks: list[tuple[int, int, int]],
) -> list[int]:
    """Chunk indices whose [start_char, end_char) overlaps [start, end).

    ``chunks`` is ``(chunk_index, start_char, end_char)`` pre-sorted by
    chunk_index, so the returned list is ordered and ``[0]`` is the first
    overlapping chunk.
    """
    return [ci for ci, s, e in chunks if s < end and start < e]


def _rewrite_graph_edges(base: Path, mention_edges: list[dict]) -> None:
    """Rewrite ``<base>.graph_edges.parquet``: keep non-mention edges, add fresh.

    All existing edges except ``mentioned_in`` are preserved (hierarchy,
    citations, cross-references); the previous ``mentioned_in`` set — empty at
    enrich time, or stale on a re-run — is dropped and replaced.
    """
    existing = read_graph_edges(base).to_pylist() if graph_edges_path_for(base).exists() else []
    kept = [e for e in existing if e["relation"] != _MENTION_RELATION]
    write_graph_edges_rows(kept + mention_edges, base)


# ---------------------------------------------------------------------------
# Sidecar readers
# ---------------------------------------------------------------------------


def _narrative_chunks(base: Path) -> dict[str, list[tuple[int, int, int]]]:
    """Per-source narrative chunks as ``(chunk_index, start_char, end_char)``.

    Only narrative chunks share the mention coordinate space (table chunks
    carry offsets in their own markdown), so only those are linked. Sorted by
    chunk_index for stable "first overlapping chunk" selection.
    """
    by_hash: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for row in read_chunks(base).to_pylist():
        if row["content_type"] != "narrative":
            continue
        by_hash[row["source_hash"]].append(
            (row["chunk_index"], row["start_char"], row["end_char"]),
        )
    for mentions in by_hash.values():
        mentions.sort(key=lambda t: t[0])
    return dict(by_hash)


def _doc_ids(base: Path) -> list[str]:
    try:
        return list(read_manifest(base).column("doc_id").to_pylist())
    except Exception:
        return []


def _all_docs_checkpointed(base: Path, mgr: CheckpointManager) -> bool:
    doc_ids = _doc_ids(base)
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["GraphRefreshResult", "refresh_graph_edges"]
