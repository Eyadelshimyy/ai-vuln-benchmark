"""Tests for analyse/query.py — loading enrichment data from Parquet.

Uses the same enrichment fixtures as test_enrichment_output.py.
The isaacus SDK is a core dependency (always installed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("isaacus", reason="isaacus SDK not importable (broken install?)")

from womblex.analyse.graph import build_document_graph
from womblex.analyse.models import (
    EnrichmentResult,
    Location,
    Person,
    Segment,
    Span,
    Term,
)
from womblex.analyse.query import (
    edges_for_document,
    load_entity_mentions,
    load_graph_edges,
    mentions_for_document,
    pii_spans_from_mentions,
)
from womblex.process.chunker import TextChunk
from womblex.store.enrichment_output import write_entity_mentions, write_graph_edges

# ---------------------------------------------------------------------------
# Shared fixture builders (same data as test_enrichment_output.py)
# ---------------------------------------------------------------------------


def _make_text() -> str:
    return "The Department of Home Affairs issued a compliance notice."


def _make_chunks(text: str) -> list[TextChunk]:
    return [TextChunk(text=text, start_char=0, end_char=len(text), chunk_index=0)]


def _make_enrichment(text: str) -> EnrichmentResult:
    return EnrichmentResult(
        text=text,
        type="other",
        jurisdiction="AU",
        title=Span(start=0, end=3),
        segments=[
            Segment(
                id="seg:0", kind="unit", type="paragraph", category="main",
                span=Span(start=0, end=len(text)), level=0,
            ),
        ],
        locations=[
            Location(
                id="loc:0", name=Span(start=4, end=14), type="city",
                mentions=[Span(start=4, end=14)],
            ),
        ],
        persons=[
            Person(
                id="per:0", name=Span(start=4, end=31), type="politic",
                role="other", mentions=[Span(start=4, end=31)],
            ),
        ],
        terms=[
            Term(
                id="term:0", name=Span(start=40, end=50),
                meaning=Span(start=40, end=57), mentions=[Span(start=40, end=50)],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Parquet fixtures (written once per test via enrichment output writers)
# ---------------------------------------------------------------------------


@pytest.fixture()
def entity_parquet(tmp_path: Path) -> Path:
    text = _make_text()
    enrichment = _make_enrichment(text)
    chunks = _make_chunks(text)
    path = tmp_path / "entities.parquet"
    write_entity_mentions([("doc1", enrichment, chunks)], path)  # type: ignore[arg-type]
    return path


@pytest.fixture()
def edge_parquet(tmp_path: Path) -> Path:
    text = _make_text()
    enrichment = _make_enrichment(text)
    graph = build_document_graph("doc1", enrichment)
    path = tmp_path / "edges.parquet"
    write_graph_edges([("doc1", graph)], path)
    return path


# ---------------------------------------------------------------------------
# Tests: load entity mentions
# ---------------------------------------------------------------------------


class TestLoadEntityMentions:
    def test_roundtrip(self, entity_parquet: Path) -> None:
        mentions = load_entity_mentions(entity_parquet)
        assert len(mentions) == 3  # 1 person + 1 location + 1 term

    def test_person_fields(self, entity_parquet: Path) -> None:
        mentions = load_entity_mentions(entity_parquet)
        persons = [m for m in mentions if m.entity_label == "person"]
        assert len(persons) == 1
        assert persons[0].source_hash == "doc1"
        assert persons[0].entity_type == "politic"
        assert persons[0].mention_start == 4
        assert persons[0].mention_end == 31

    def test_location_fields(self, entity_parquet: Path) -> None:
        mentions = load_entity_mentions(entity_parquet)
        locs = [m for m in mentions if m.entity_label == "location"]
        assert len(locs) == 1
        assert locs[0].entity_type == "city"


# ---------------------------------------------------------------------------
# Tests: load graph edges
# ---------------------------------------------------------------------------


class TestLoadGraphEdges:
    def test_roundtrip(self, edge_parquet: Path) -> None:
        edges = load_graph_edges(edge_parquet)
        assert len(edges) > 0

    def test_contains_relation(self, edge_parquet: Path) -> None:
        edges = load_graph_edges(edge_parquet)
        contains = [e for e in edges if e.relation == "contains"]
        assert len(contains) >= 1
        assert all(e.source_hash == "doc1" for e in contains)


# ---------------------------------------------------------------------------
# Tests: filtering
# ---------------------------------------------------------------------------


class TestMentionsForDocument:
    def test_filter_by_doc(self, entity_parquet: Path) -> None:
        mentions = load_entity_mentions(entity_parquet)
        doc1 = mentions_for_document(mentions, "doc1")
        assert len(doc1) == 3
        missing = mentions_for_document(mentions, "nonexistent")
        assert len(missing) == 0

    def test_filter_by_label(self, entity_parquet: Path) -> None:
        mentions = load_entity_mentions(entity_parquet)
        persons = mentions_for_document(mentions, "doc1", label="person")
        assert len(persons) == 1
        terms = mentions_for_document(mentions, "doc1", label="term")
        assert len(terms) == 1


class TestEdgesForDocument:
    def test_filter_by_doc(self, edge_parquet: Path) -> None:
        edges = load_graph_edges(edge_parquet)
        doc1 = edges_for_document(edges, "doc1")
        assert len(doc1) > 0
        missing = edges_for_document(edges, "nonexistent")
        assert len(missing) == 0

    def test_filter_by_relation(self, edge_parquet: Path) -> None:
        edges = load_graph_edges(edge_parquet)
        contains = edges_for_document(edges, "doc1", relation="contains")
        assert len(contains) >= 1


# ---------------------------------------------------------------------------
# Tests: PII span extraction
# ---------------------------------------------------------------------------


class TestPIISpansFromMentions:
    def test_extracts_person_and_location(self, entity_parquet: Path) -> None:
        mentions = load_entity_mentions(entity_parquet)
        spans = pii_spans_from_mentions(mentions, "doc1")
        # Should have person (4-31) and location (4-14)
        assert len(spans) == 2
        assert all(s[0] >= 0 and s[1] > s[0] for s in spans)

    def test_sorted_by_start(self, entity_parquet: Path) -> None:
        mentions = load_entity_mentions(entity_parquet)
        spans = pii_spans_from_mentions(mentions, "doc1")
        starts = [s[0] for s in spans]
        assert starts == sorted(starts)

    def test_custom_labels(self, entity_parquet: Path) -> None:
        mentions = load_entity_mentions(entity_parquet)
        person_only = pii_spans_from_mentions(mentions, "doc1", labels={"person"})
        assert len(person_only) == 1
        assert person_only[0][2] == "person"

    def test_empty_for_missing_doc(self, entity_parquet: Path) -> None:
        mentions = load_entity_mentions(entity_parquet)
        spans = pii_spans_from_mentions(mentions, "nonexistent")
        assert spans == []
