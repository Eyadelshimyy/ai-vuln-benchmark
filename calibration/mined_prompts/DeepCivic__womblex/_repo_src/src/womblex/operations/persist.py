"""Parquet output helpers for batch results (E2E composition path)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from womblex.operations.models import BatchResult
from womblex.store.run_stamp import RunStamp
from womblex.store.source_provenance import IngestProvenance

logger = logging.getLogger(__name__)


def write_batch_parquet(
    batch: BatchResult,
    output_path: Path,
    *,
    provenance: IngestProvenance | None = None,
    stamp: RunStamp | None = None,
) -> Path | None:
    """Write completed batch results to Parquet.

    ``provenance`` names the declared ingest root the documents were read
    from; it reaches the manifest columns and every shard's footer. ``stamp``
    names the run that produced them, in the same footer.
    """
    from womblex.store.output import write_results

    rows = []
    for r in batch.results:
        if r.status == "completed" and r.extraction:
            rows.append((r.doc_id, str(r.path), r.extraction))

    if not rows:
        logger.warning("No completed results to write to Parquet")
        return None

    return write_results(rows, output_path, provenance=provenance, stamp=stamp)


def write_batch_enrichment(batch: BatchResult, output_dir: Path) -> dict[str, Path | None]:
    """Write enrichment results and graph data to Parquet."""
    from womblex.store.enrichment_output import (
        write_enrichment_metadata,
        write_entity_mentions,
        write_graph_edges,
    )

    entity_rows: list[tuple[str, Any, list[object] | None]] = []
    graph_rows: list[tuple[str, Any]] = []
    meta_rows: list[tuple[str, Any]] = []

    for r in batch.results:
        if r.enrichment is None:
            continue
        entity_rows.append((r.doc_id, r.enrichment, r.chunks or None))  # type: ignore[arg-type]
        meta_rows.append((r.doc_id, r.enrichment))
        if r.graph is not None:
            graph_rows.append((r.doc_id, r.graph))

    paths: dict[str, Path | None] = {
        "entities": None,
        "graph_edges": None,
        "enrichment_meta": None,
    }

    if entity_rows:
        paths["entities"] = write_entity_mentions(entity_rows, output_dir / "entities.parquet")
    if graph_rows:
        paths["graph_edges"] = write_graph_edges(graph_rows, output_dir / "graph_edges.parquet")
    if meta_rows:
        paths["enrichment_meta"] = write_enrichment_metadata(
            meta_rows, output_dir / "enrichment_meta.parquet"
        )

    return paths
