"""Tests for graph construction from enrichment results.

The isaacus SDK is a core dependency (always installed).
"""

from __future__ import annotations

import pytest

# Skip only on a broken install where the isaacus SDK cannot be imported
pytest.importorskip("isaacus", reason="isaacus SDK not importable (broken install?)")

from womblex.analyse.graph import (
    DocumentGraph,
    GraphEdge,
    GraphNode,
    _find_chunks_for_span,
    build_document_graph,
)
from womblex.analyse.models import (
    CrossReference,
    DateInfo,
    Email,
    EnrichmentResult,
    ExternalDocument,
    Location,
    Person,
    Segment,
    Span,
    Term,
    Website,
)
from womblex.process.chunker import TextChunk

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_text() -> str:
    return "The quick brown fox jumps over the lazy dog. This is a test document about government compliance."


def _make_chunks(text: str) -> list[TextChunk]:
    """Split text into two chunks at the sentence boundary."""
    split = text.index(". ") + 2
    return [
        TextChunk(text=text[:split], start_char=0, end_char=split, chunk_index=0),
        TextChunk(text=text[split:], start_char=split, end_char=len(text), chunk_index=1),
    ]


def _make_enrichment(text: str) -> EnrichmentResult:
    """Build a minimal EnrichmentResult for testing."""
    return EnrichmentResult(
        text=text,
        type="other",
        jurisdiction="AU",
        title=Span(start=0, end=3),  # "The"
        segments=[
            Segment(
                id="seg:0",
                kind="container",
                type="section",
                category="main",
                span=Span(start=0, end=len(text)),
                children=["seg:1"],
                level=0,
            ),
            Segment(
                id="seg:1",
                kind="unit",
                type="paragraph",
                category="main",
                span=Span(start=0, end=44),  # first sentence
                parent="seg:0",
                level=1,
            ),
        ],
        crossreferences=[
            CrossReference(
                start="seg:1",
                end="seg:1",
                span=Span(start=4, end=9),  # "quick"
            ),
        ],
        locations=[
            Location(
                id="loc:0",
                name=Span(start=46, end=50),  # "This"
                type="country",
                mentions=[Span(start=46, end=50)],
            ),
        ],
        persons=[
            Person(
                id="per:0",
                name=Span(start=4, end=9),  # "quick"
                type="corporate",
                role="other",
                mentions=[Span(start=4, end=9)],
                residence="loc:0",
            ),
        ],
        emails=[
            Email(address="test@example.com", person="per:0", mentions=[Span(start=0, end=3)]),
        ],
        websites=[
            Website(url="https://example.com/", person="per:0", mentions=[Span(start=0, end=3)]),
        ],
        terms=[
            Term(
                id="term:0",
                name=Span(start=56, end=60),  # "test"
                meaning=Span(start=56, end=64),  # "test doc"
                mentions=[Span(start=56, end=60)],
            ),
        ],
        external_documents=[
            ExternalDocument(
                id="exd:0",
                name=Span(start=71, end=81),  # "government"
                type="statute",
                reception="neutral",
                jurisdiction="AU-FED",
                mentions=[Span(start=71, end=81)],
            ),
        ],
        dates=[
            DateInfo(value="2024-01-15", type="creation", mentions=[Span(start=0, end=3)]),
        ],
    )


# ---------------------------------------------------------------------------
# Tests: span-to-chunk mapping
# ---------------------------------------------------------------------------


class TestSpanToChunkMapping:
    def test_span_in_first_chunk(self) -> None:
        text = _make_text()
        chunks = _make_chunks(text)
        # Span within first chunk
        result = _find_chunks_for_span(Span(start=0, end=10), chunks)
        assert result == [0]

    def test_span_in_second_chunk(self) -> None:
        text = _make_text()
        chunks = _make_chunks(text)
        # Span well into second chunk
        result = _find_chunks_for_span(Span(start=56, end=60), chunks)
        assert result == [1]

    def test_span_crossing_chunks(self) -> None:
        text = _make_text()
        chunks = _make_chunks(text)
        split = text.index(". ") + 2
        # Span that crosses the boundary
        result = _find_chunks_for_span(Span(start=split - 5, end=split + 5), chunks)
        assert 0 in result
        assert 1 in result

    def test_span_no_match(self) -> None:
        text = _make_text()
        chunks = _make_chunks(text)
        # Span beyond all chunks
        result = _find_chunks_for_span(Span(start=9999, end=10000), chunks)
        assert result == []

    def test_empty_chunks(self) -> None:
        result = _find_chunks_for_span(Span(start=0, end=10), [])
        assert result == []


# ---------------------------------------------------------------------------
# Tests: graph primitives
# ---------------------------------------------------------------------------


class TestGraphPrimitives:
    def test_add_node(self) -> None:
        graph = DocumentGraph(document_id="doc1")
        node = GraphNode(id="n1", label="test", properties={"key": "value"})
        graph.add_node(node)
        assert "n1" in graph.nodes
        assert graph.nodes["n1"].properties["key"] == "value"

    def test_add_edge(self) -> None:
        graph = DocumentGraph(document_id="doc1")
        edge = GraphEdge(source="a", target="b", relation="test")
        graph.add_edge(edge)
        assert len(graph.edges) == 1
        assert graph.edges[0].relation == "test"

    def test_get_nodes_by_label(self) -> None:
        graph = DocumentGraph(document_id="doc1")
        graph.add_node(GraphNode(id="n1", label="person"))
        graph.add_node(GraphNode(id="n2", label="location"))
        graph.add_node(GraphNode(id="n3", label="person"))

        persons = graph.get_nodes_by_label("person")
        assert len(persons) == 2

    def test_get_edges_by_relation(self) -> None:
        graph = DocumentGraph(document_id="doc1")
        graph.add_edge(GraphEdge(source="a", target="b", relation="contains"))
        graph.add_edge(GraphEdge(source="a", target="c", relation="mentions"))
        graph.add_edge(GraphEdge(source="b", target="c", relation="contains"))

        contains = graph.get_edges_by_relation("contains")
        assert len(contains) == 2

    def test_get_edges_from(self) -> None:
        graph = DocumentGraph(document_id="doc1")
        graph.add_edge(GraphEdge(source="a", target="b", relation="r1"))
        graph.add_edge(GraphEdge(source="a", target="c", relation="r2"))
        graph.add_edge(GraphEdge(source="b", target="c", relation="r3"))

        from_a = graph.get_edges_from("a")
        assert len(from_a) == 2

    def test_get_edges_to(self) -> None:
        graph = DocumentGraph(document_id="doc1")
        graph.add_edge(GraphEdge(source="a", target="c", relation="r1"))
        graph.add_edge(GraphEdge(source="b", target="c", relation="r2"))

        to_c = graph.get_edges_to("c")
        assert len(to_c) == 2


# ---------------------------------------------------------------------------
# Tests: full graph construction
# ---------------------------------------------------------------------------


class TestBuildDocumentGraph:
    def test_basic_graph(self) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)

        assert graph.document_id == "doc1"
        assert "doc1" in graph.nodes
        assert graph.nodes["doc1"].label == "document"
        assert graph.nodes["doc1"].properties["type"] == "other"
        assert graph.nodes["doc1"].properties["jurisdiction"] == "AU"

    def test_graph_with_chunks(self) -> None:
        text = _make_text()
        chunks = _make_chunks(text)
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment, chunks=chunks)

        # Should have chunk nodes
        chunk_nodes = graph.get_nodes_by_label("chunk")
        assert len(chunk_nodes) == 2

        # Chunks should be linked to document
        contains_edges = [
            e for e in graph.get_edges_by_relation("contains")
            if e.source == "doc1" and "chunk" in e.target
        ]
        assert len(contains_edges) == 2

    def test_segment_hierarchy(self) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)

        # Root segment linked to document
        root_edges = [
            e for e in graph.edges
            if e.source == "doc1" and e.target == "doc1:seg:0" and e.relation == "contains"
        ]
        assert len(root_edges) == 1

        # Child segment linked to parent
        parent_edges = [
            e for e in graph.edges
            if e.source == "doc1:seg:0" and e.target == "doc1:seg:1" and e.relation == "parent_of"
        ]
        assert len(parent_edges) == 1

    def test_person_nodes(self) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)

        person_nodes = graph.get_nodes_by_label("person")
        assert len(person_nodes) == 1
        assert person_nodes[0].properties["type"] == "corporate"

    def test_person_residence_link(self) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)

        resides_edges = graph.get_edges_by_relation("resides_at")
        assert len(resides_edges) == 1
        assert resides_edges[0].source == "doc1:per:0"
        assert resides_edges[0].target == "doc1:loc:0"

    def test_location_nodes(self) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)

        loc_nodes = graph.get_nodes_by_label("location")
        assert len(loc_nodes) == 1
        assert loc_nodes[0].properties["type"] == "country"

    def test_term_nodes(self) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)

        term_nodes = graph.get_nodes_by_label("term")
        assert len(term_nodes) == 1

    def test_external_document_nodes(self) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)

        exd_nodes = graph.get_nodes_by_label("external_document")
        assert len(exd_nodes) == 1
        assert exd_nodes[0].properties["type"] == "statute"

        # Should have a cites edge
        cites_edges = graph.get_edges_by_relation("cites")
        assert len(cites_edges) == 1

    def test_mention_to_chunk_links(self) -> None:
        text = _make_text()
        chunks = _make_chunks(text)
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment, chunks=chunks)

        mention_edges = graph.get_edges_by_relation("mentioned_in")
        assert len(mention_edges) > 0

        # Person mention is in chunk 0 (start=4, end=9)
        per_mention_edges = [
            e for e in mention_edges if e.source == "doc1:per:0"
        ]
        assert len(per_mention_edges) >= 1
        assert any("chunk:0" in e.target for e in per_mention_edges)

    def test_contact_info_edges(self) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)

        email_edges = graph.get_edges_by_relation("has_email")
        assert len(email_edges) == 1
        assert email_edges[0].properties["address"] == "test@example.com"

        website_edges = graph.get_edges_by_relation("has_website")
        assert len(website_edges) == 1

    def test_date_edges(self) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)

        date_edges = graph.get_edges_by_relation("date_creation")
        assert len(date_edges) == 1
        assert date_edges[0].properties["value"] == "2024-01-15"

    def test_crossreference_edges(self) -> None:
        text = _make_text()
        chunks = _make_chunks(text)
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment, chunks=chunks)

        xref_edges = graph.get_edges_by_relation("crossreferences")
        assert len(xref_edges) == 1

    def test_graph_without_chunks(self) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)

        # Should still work, just no chunk nodes or mention-to-chunk links
        chunk_nodes = graph.get_nodes_by_label("chunk")
        assert len(chunk_nodes) == 0

        # Entities still exist
        person_nodes = graph.get_nodes_by_label("person")
        assert len(person_nodes) == 1

    def test_node_count(self) -> None:
        text = _make_text()
        chunks = _make_chunks(text)
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment, chunks=chunks)

        # 1 document + 2 chunks + 2 segments + 1 person + 1 location +
        # 1 term + 1 external_document = 9 nodes
        assert len(graph.nodes) == 9
