"""Graph construction from Isaacus enrichment results.

Builds a document graph with nodes (entities, segments, chunks) and
edges (relationships, containment, mentions).  The graph is a pure
Python data structure that can be serialised to Parquet for flat
queries or exported to JSON for loading into a graph database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from womblex.analyse.models import (
    EnrichmentResult,
    Span,
)
from womblex.process.chunker import TextChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    """A node in the document graph."""

    id: str
    label: str  # document | chunk | segment | person | location | term | external_document
    properties: dict[str, str | int | float | bool | None] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A directed edge in the document graph."""

    source: str  # node id
    target: str  # node id
    relation: str  # contains | parent_of | mentions | resides_at | crossreferences | cites | etc.
    properties: dict[str, str | int | float | bool | None] = field(default_factory=dict)


@dataclass
class DocumentGraph:
    """Complete graph for a single document."""

    document_id: str
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def get_nodes_by_label(self, label: str) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.label == label]

    def get_edges_by_relation(self, relation: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.relation == relation]

    def get_edges_from(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.target == node_id]


# ---------------------------------------------------------------------------
# Span-to-chunk mapping
# ---------------------------------------------------------------------------


def _find_chunks_for_span(span: Span, chunks: list[TextChunk]) -> list[int]:
    """Find chunk indices whose character range overlaps with a span."""
    result: list[int] = []
    for chunk in chunks:
        if chunk.start_char < span.end and span.start < chunk.end_char:
            result.append(chunk.chunk_index)
    return result


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _add_document_node(graph: DocumentGraph, enrichment: EnrichmentResult) -> None:
    """Add the root document node."""
    title_text = enrichment.title.decode(enrichment.text) if enrichment.title else None
    graph.add_node(GraphNode(
        id=graph.document_id,
        label="document",
        properties={
            "type": enrichment.type,
            "jurisdiction": enrichment.jurisdiction,
            "title": title_text,
        },
    ))


def _add_chunk_nodes(
    graph: DocumentGraph,
    chunks: list[TextChunk],
) -> None:
    """Add chunk nodes and link them to the document."""
    for chunk in chunks:
        chunk_id = f"{graph.document_id}:chunk:{chunk.chunk_index}"
        graph.add_node(GraphNode(
            id=chunk_id,
            label="chunk",
            properties={
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "content_type": chunk.content_type,
                "chunk_index": chunk.chunk_index,
            },
        ))
        graph.add_edge(GraphEdge(
            source=graph.document_id,
            target=chunk_id,
            relation="contains",
        ))


def _add_segment_nodes(
    graph: DocumentGraph,
    enrichment: EnrichmentResult,
    chunks: list[TextChunk],
) -> None:
    """Add segment nodes with hierarchy and link to overlapping chunks."""
    doc_id = graph.document_id

    for seg in enrichment.segments:
        seg_node_id = f"{doc_id}:{seg.id}"
        title_text = seg.title.decode(enrichment.text) if seg.title else None
        code_text = seg.code.decode(enrichment.text) if seg.code else None

        graph.add_node(GraphNode(
            id=seg_node_id,
            label="segment",
            properties={
                "kind": seg.kind,
                "type": seg.type,
                "category": seg.category,
                "level": seg.level,
                "title": title_text,
                "code": code_text,
                "start": seg.span.start,
                "end": seg.span.end,
            },
        ))

        # Parent-child hierarchy
        if seg.parent:
            parent_node_id = f"{doc_id}:{seg.parent}"
            graph.add_edge(GraphEdge(
                source=parent_node_id,
                target=seg_node_id,
                relation="parent_of",
            ))
        else:
            # Root segment → document
            graph.add_edge(GraphEdge(
                source=doc_id,
                target=seg_node_id,
                relation="contains",
            ))

        # Link segment to overlapping chunks
        chunk_indices = _find_chunks_for_span(seg.span, chunks)
        for ci in chunk_indices:
            chunk_node_id = f"{doc_id}:chunk:{ci}"
            graph.add_edge(GraphEdge(
                source=seg_node_id,
                target=chunk_node_id,
                relation="spans_chunk",
            ))


def _add_person_nodes(
    graph: DocumentGraph,
    enrichment: EnrichmentResult,
    chunks: list[TextChunk],
) -> None:
    """Add person entity nodes with hierarchy and mention links."""
    doc_id = graph.document_id

    for per in enrichment.persons:
        per_node_id = f"{doc_id}:{per.id}"
        name_text = per.name.decode(enrichment.text)

        graph.add_node(GraphNode(
            id=per_node_id,
            label="person",
            properties={
                "name": name_text,
                "type": per.type,
                "role": per.role,
            },
        ))

        # Parent entity relationship
        if per.parent:
            parent_node_id = f"{doc_id}:{per.parent}"
            graph.add_edge(GraphEdge(
                source=parent_node_id,
                target=per_node_id,
                relation="parent_of",
            ))

        # Residence link
        if per.residence:
            loc_node_id = f"{doc_id}:{per.residence}"
            graph.add_edge(GraphEdge(
                source=per_node_id,
                target=loc_node_id,
                relation="resides_at",
            ))

        # Mention links to chunks
        for mention in per.mentions:
            chunk_indices = _find_chunks_for_span(mention, chunks)
            for ci in chunk_indices:
                chunk_node_id = f"{doc_id}:chunk:{ci}"
                graph.add_edge(GraphEdge(
                    source=per_node_id,
                    target=chunk_node_id,
                    relation="mentioned_in",
                    properties={"start": mention.start, "end": mention.end},
                ))


def _add_location_nodes(
    graph: DocumentGraph,
    enrichment: EnrichmentResult,
    chunks: list[TextChunk],
) -> None:
    """Add location entity nodes with hierarchy and mention links."""
    doc_id = graph.document_id

    for loc in enrichment.locations:
        loc_node_id = f"{doc_id}:{loc.id}"
        name_text = loc.name.decode(enrichment.text)

        graph.add_node(GraphNode(
            id=loc_node_id,
            label="location",
            properties={
                "name": name_text,
                "type": loc.type,
            },
        ))

        # Parent location hierarchy
        if loc.parent:
            parent_node_id = f"{doc_id}:{loc.parent}"
            graph.add_edge(GraphEdge(
                source=parent_node_id,
                target=loc_node_id,
                relation="parent_of",
            ))

        # Mention links to chunks
        for mention in loc.mentions:
            chunk_indices = _find_chunks_for_span(mention, chunks)
            for ci in chunk_indices:
                chunk_node_id = f"{doc_id}:chunk:{ci}"
                graph.add_edge(GraphEdge(
                    source=loc_node_id,
                    target=chunk_node_id,
                    relation="mentioned_in",
                    properties={"start": mention.start, "end": mention.end},
                ))


def _add_term_nodes(
    graph: DocumentGraph,
    enrichment: EnrichmentResult,
    chunks: list[TextChunk],
) -> None:
    """Add defined term nodes with mention links."""
    doc_id = graph.document_id

    for term in enrichment.terms:
        term_node_id = f"{doc_id}:{term.id}"
        name_text = term.name.decode(enrichment.text)
        meaning_text = term.meaning.decode(enrichment.text)

        graph.add_node(GraphNode(
            id=term_node_id,
            label="term",
            properties={
                "name": name_text,
                "meaning": meaning_text,
            },
        ))

        # Mention links to chunks
        for mention in term.mentions:
            chunk_indices = _find_chunks_for_span(mention, chunks)
            for ci in chunk_indices:
                chunk_node_id = f"{doc_id}:chunk:{ci}"
                graph.add_edge(GraphEdge(
                    source=term_node_id,
                    target=chunk_node_id,
                    relation="mentioned_in",
                    properties={"start": mention.start, "end": mention.end},
                ))


def _add_external_document_nodes(
    graph: DocumentGraph,
    enrichment: EnrichmentResult,
    chunks: list[TextChunk],
) -> None:
    """Add external document nodes with citation links."""
    doc_id = graph.document_id

    for exd in enrichment.external_documents:
        exd_node_id = f"{doc_id}:{exd.id}"
        name_text = exd.name.decode(enrichment.text)

        graph.add_node(GraphNode(
            id=exd_node_id,
            label="external_document",
            properties={
                "name": name_text,
                "type": exd.type,
                "jurisdiction": exd.jurisdiction,
                "reception": exd.reception,
            },
        ))

        # Citation links — document cites external document
        graph.add_edge(GraphEdge(
            source=doc_id,
            target=exd_node_id,
            relation="cites",
            properties={"reception": exd.reception},
        ))

        # Mention links to chunks
        for mention in exd.mentions:
            chunk_indices = _find_chunks_for_span(mention, chunks)
            for ci in chunk_indices:
                chunk_node_id = f"{doc_id}:chunk:{ci}"
                graph.add_edge(GraphEdge(
                    source=exd_node_id,
                    target=chunk_node_id,
                    relation="mentioned_in",
                    properties={"start": mention.start, "end": mention.end},
                ))


def _add_crossreference_edges(
    graph: DocumentGraph,
    enrichment: EnrichmentResult,
    chunks: list[TextChunk],
) -> None:
    """Add cross-reference edges between segments."""
    doc_id = graph.document_id

    for xref in enrichment.crossreferences:
        start_seg_id = f"{doc_id}:{xref.start}"
        end_seg_id = f"{doc_id}:{xref.end}"

        graph.add_edge(GraphEdge(
            source=start_seg_id,
            target=end_seg_id,
            relation="crossreferences",
            properties={
                "span_start": xref.span.start,
                "span_end": xref.span.end,
            },
        ))

        # Link crossreference mention to chunks
        chunk_indices = _find_chunks_for_span(xref.span, chunks)
        for ci in chunk_indices:
            chunk_node_id = f"{doc_id}:chunk:{ci}"
            graph.add_edge(GraphEdge(
                source=start_seg_id,
                target=chunk_node_id,
                relation="crossref_in",
                properties={"start": xref.span.start, "end": xref.span.end},
            ))


def _add_contact_info_edges(
    graph: DocumentGraph,
    enrichment: EnrichmentResult,
) -> None:
    """Link contact information (email, website, phone, ID) to persons."""
    doc_id = graph.document_id

    for email in enrichment.emails:
        per_node_id = f"{doc_id}:{email.person}"
        graph.add_edge(GraphEdge(
            source=per_node_id,
            target=per_node_id,
            relation="has_email",
            properties={"address": email.address},
        ))

    for website in enrichment.websites:
        per_node_id = f"{doc_id}:{website.person}"
        graph.add_edge(GraphEdge(
            source=per_node_id,
            target=per_node_id,
            relation="has_website",
            properties={"url": website.url},
        ))

    for phone in enrichment.phone_numbers:
        per_node_id = f"{doc_id}:{phone.person}"
        graph.add_edge(GraphEdge(
            source=per_node_id,
            target=per_node_id,
            relation="has_phone",
            properties={"number": phone.number},
        ))

    for idn in enrichment.id_numbers:
        per_node_id = f"{doc_id}:{idn.person}"
        graph.add_edge(GraphEdge(
            source=per_node_id,
            target=per_node_id,
            relation="has_id_number",
            properties={"number": idn.number},
        ))


def _add_date_edges(
    graph: DocumentGraph,
    enrichment: EnrichmentResult,
) -> None:
    """Link dates to the document and optionally to persons."""
    doc_id = graph.document_id

    for date in enrichment.dates:
        target = doc_id
        if date.person:
            target = f"{doc_id}:{date.person}"

        graph.add_edge(GraphEdge(
            source=doc_id,
            target=target,
            relation=f"date_{date.type}",
            properties={"value": date.value},
        ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_document_graph(
    document_id: str,
    enrichment: EnrichmentResult,
    chunks: list[TextChunk] | None = None,
) -> DocumentGraph:
    """Build a complete document graph from enrichment results and chunks.

    The graph contains nodes for the document, its chunks, segments,
    persons, locations, terms, and external documents, connected by
    edges representing containment, hierarchy, mentions, citations,
    and cross-references.

    Args:
        document_id: Unique identifier for the document.
        enrichment: Enrichment result from the Isaacus API.
        chunks: Optional list of text chunks with character offsets.
            When provided, entity mentions are linked to specific chunks.

    Returns:
        A DocumentGraph with nodes and edges.
    """
    graph = DocumentGraph(document_id=document_id)
    chunk_list = chunks or []

    _add_document_node(graph, enrichment)
    _add_chunk_nodes(graph, chunk_list)
    _add_segment_nodes(graph, enrichment, chunk_list)
    _add_person_nodes(graph, enrichment, chunk_list)
    _add_location_nodes(graph, enrichment, chunk_list)
    _add_term_nodes(graph, enrichment, chunk_list)
    _add_external_document_nodes(graph, enrichment, chunk_list)
    _add_crossreference_edges(graph, enrichment, chunk_list)
    _add_contact_info_edges(graph, enrichment)
    _add_date_edges(graph, enrichment)

    logger.info(
        "Built graph for %s: %d nodes, %d edges",
        document_id, len(graph.nodes), len(graph.edges),
    )
    return graph
