"""Load enrichment graph data from Parquet for internal pipeline use.

Reads entity mentions and graph edges written by
``store/enrichment_output.py`` back into structures that pipeline
stages can consume — primarily for PII cleaning against previously
enriched data without re-running enrichment.

Not an end-user query API. This is an internal utility for the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from womblex.store.evidence import TABLE_LAYER

logger = logging.getLogger(__name__)


@dataclass
class EntityMention:
    """An entity mention loaded from Parquet."""

    source_hash: str
    entity_id: str
    entity_label: str
    name: str
    entity_type: str
    role: str
    mention_start: int
    mention_end: int
    chunk_index: int
    text_layer: str | None = None  # 'table_markdown' when the offsets index a table, not the narrative


@dataclass
class Edge:
    """A relationship edge loaded from Parquet."""

    source_hash: str
    source_id: str
    target_id: str
    relation: str
    properties: dict[str, str] = field(default_factory=dict)


def load_entity_mentions(path: Path) -> list[EntityMention]:
    """Load entity mentions from a Parquet file."""
    table = pq.read_table(str(path))
    layers = table.column("text_layer") if "text_layer" in table.schema.names else None
    return [
        EntityMention(
            source_hash=table.column("source_hash")[i].as_py(),
            entity_id=table.column("entity_id")[i].as_py(),
            entity_label=table.column("entity_label")[i].as_py(),
            name=table.column("name")[i].as_py(),
            entity_type=table.column("entity_type")[i].as_py(),
            role=table.column("role")[i].as_py() or "",
            mention_start=table.column("mention_start")[i].as_py(),
            mention_end=table.column("mention_end")[i].as_py(),
            chunk_index=table.column("chunk_index")[i].as_py(),
            text_layer=layers[i].as_py() if layers is not None else None,
        )
        for i in range(len(table))
    ]


def load_graph_edges(path: Path) -> list[Edge]:
    """Load graph edges from a Parquet file."""
    table = pq.read_table(str(path))
    edges: list[Edge] = []
    for i in range(len(table)):
        props: dict[str, str] = {}
        pk = table.column("prop_key")[i].as_py()
        pv = table.column("prop_value")[i].as_py()
        if pk:
            props[pk] = pv or ""
        edges.append(Edge(
            source_hash=table.column("source_hash")[i].as_py(),
            source_id=table.column("source_id")[i].as_py(),
            target_id=table.column("target_id")[i].as_py(),
            relation=table.column("relation")[i].as_py(),
            properties=props,
        ))
    return edges


def mentions_for_document(
    mentions: list[EntityMention], source_hash: str, *, label: str | None = None,
) -> list[EntityMention]:
    """Filter mentions for a document, optionally by entity label."""
    result = [m for m in mentions if m.source_hash == source_hash]
    if label:
        result = [m for m in result if m.entity_label == label]
    return result


def edges_for_document(
    edges: list[Edge], source_hash: str, *, relation: str | None = None,
) -> list[Edge]:
    """Filter edges for a document, optionally by relation type."""
    result = [e for e in edges if e.source_hash == source_hash]
    if relation:
        result = [e for e in result if e.relation == relation]
    return result


def pii_spans_from_mentions(
    mentions: list[EntityMention], source_hash: str, *, labels: set[str] | None = None,
) -> list[tuple[int, int, str]]:
    """Extract PII-relevant (start, end, label) spans of the narrative for masking.

    Table mentions index their own markdown, not the narrative, and are skipped.
    """
    if labels is None:
        labels = {"person", "location"}
    spans = [
        (m.mention_start, m.mention_end, m.entity_label)
        for m in mentions
        if m.source_hash == source_hash
        and m.text_layer != TABLE_LAYER
        and m.entity_label in labels
        and m.mention_start >= 0
        and m.mention_end > m.mention_start
    ]
    spans.sort(key=lambda s: s[0])
    return spans
