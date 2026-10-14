"""Tests for enrichment Parquet output.

The isaacus SDK is a core dependency (always installed).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Skip only on a broken install where the isaacus SDK cannot be imported
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
from womblex.process.chunker import TextChunk
from womblex.store.enrichment_output import (
    ENRICHMENT_META_SCHEMA,
    ENTITY_SCHEMA,
    GRAPH_EDGE_SCHEMA,
    LEGACY_HASH_COLUMN,
    enrichment_entities_path_for,
    graph_edges_path_for,
    read_enrichment_entities,
    read_graph_edges,
    write_enrichment_entities_shard,
    write_enrichment_metadata,
    write_entity_mentions,
    write_graph_edges,
    write_graph_edges_shard,
)

# ---------------------------------------------------------------------------
# Fixtures
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
                id="seg:0",
                kind="unit",
                type="paragraph",
                category="main",
                span=Span(start=0, end=len(text)),
                level=0,
            ),
        ],
        locations=[
            Location(
                id="loc:0",
                name=Span(start=4, end=14),  # "Department"
                type="city",
                mentions=[Span(start=4, end=14)],
            ),
        ],
        persons=[
            Person(
                id="per:0",
                name=Span(start=4, end=31),  # "Department of Home Affairs"
                type="politic",
                role="other",
                mentions=[Span(start=4, end=31)],
            ),
        ],
        terms=[
            Term(
                id="term:0",
                name=Span(start=40, end=50),  # "compliance"
                meaning=Span(start=40, end=57),  # "compliance notice"
                mentions=[Span(start=40, end=50)],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests: entity mentions
# ---------------------------------------------------------------------------


class TestWriteEntityMentions:
    def test_write_entities(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        chunks = _make_chunks(text)
        output = tmp_path / "entities.parquet"

        write_entity_mentions([(
            "doc1", enrichment, chunks,  # type: ignore[arg-type]
        )], output)

        assert output.exists()
        table = pq.read_table(str(output))
        assert table.num_rows == 3  # 1 person + 1 location + 1 term mention

        # Check schema
        for field in ENTITY_SCHEMA:
            assert field.name in table.column_names

    def test_write_entities_empty(self, tmp_path: Path) -> None:
        output = tmp_path / "entities_empty.parquet"
        write_entity_mentions([], output)
        assert output.exists()
        table = pq.read_table(str(output))
        assert table.num_rows == 0

    def test_entity_content(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        output = tmp_path / "entities.parquet"

        write_entity_mentions([(
            "doc1", enrichment, None,
        )], output)

        table = pq.read_table(str(output))
        doc_ids = table.column("source_hash").to_pylist()
        assert all(d == "doc1" for d in doc_ids)

        labels = table.column("entity_label").to_pylist()
        assert "person" in labels
        assert "location" in labels
        assert "term" in labels

    def test_chunk_mapping(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        chunks = _make_chunks(text)
        output = tmp_path / "entities.parquet"

        write_entity_mentions([(
            "doc1", enrichment, chunks,  # type: ignore[arg-type]
        )], output)

        table = pq.read_table(str(output))
        chunk_indices = table.column("chunk_index").to_pylist()
        # All mentions overlap with chunk 0
        assert all(ci == 0 for ci in chunk_indices)


# ---------------------------------------------------------------------------
# Tests: graph edges
# ---------------------------------------------------------------------------


class TestWriteGraphEdges:
    def test_write_edges(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)
        output = tmp_path / "edges.parquet"

        write_graph_edges([("doc1", graph)], output)

        assert output.exists()
        table = pq.read_table(str(output))
        assert table.num_rows > 0

        for field in GRAPH_EDGE_SCHEMA:
            assert field.name in table.column_names

    def test_write_edges_empty(self, tmp_path: Path) -> None:
        output = tmp_path / "edges_empty.parquet"
        write_graph_edges([], output)
        assert output.exists()
        table = pq.read_table(str(output))
        assert table.num_rows == 0

    def test_edge_content(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)
        output = tmp_path / "edges.parquet"

        write_graph_edges([("doc1", graph)], output)

        table = pq.read_table(str(output))
        relations = table.column("relation").to_pylist()
        assert "contains" in relations


class TestGraphEdgesShard:
    """Per-batch ``*.graph_edges.parquet`` sibling (the shippable graph)."""

    def test_shard_round_trips_via_base_and_dir(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        chunks = _make_chunks(text)
        graph = build_document_graph("hash1", enrichment, chunks)
        base = tmp_path / "batch-0001.parquet"

        target = write_graph_edges_shard([("hash1", graph)], base)
        assert target == graph_edges_path_for(base)
        assert target.exists()

        by_base = read_graph_edges(base)
        by_dir = read_graph_edges(tmp_path)
        assert by_base.num_rows == by_dir.num_rows > 0
        assert set(by_base.column("source_hash").to_pylist()) == {"hash1"}
        relations = set(by_base.column("relation").to_pylist())
        # containment + chunk-mention edges both present when chunks are passed
        assert "contains" in relations
        assert "mentioned_in" in relations

    def test_empty_shard_and_empty_dir_are_schema_correct(self, tmp_path: Path) -> None:
        base = tmp_path / "batch-0001.parquet"
        write_graph_edges_shard([], base)
        table = read_graph_edges(base)
        assert table.num_rows == 0
        assert table.schema.names == GRAPH_EDGE_SCHEMA.names

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        table = read_graph_edges(empty_dir)
        assert table.num_rows == 0
        assert table.schema.names == GRAPH_EDGE_SCHEMA.names


# ---------------------------------------------------------------------------
# Tests: enrichment metadata
# ---------------------------------------------------------------------------


class TestWriteEnrichmentMetadata:
    def test_write_metadata(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        output = tmp_path / "meta.parquet"

        write_enrichment_metadata([("doc1", enrichment)], output)

        assert output.exists()
        table = pq.read_table(str(output))
        assert table.num_rows == 1

        for field in ENRICHMENT_META_SCHEMA:
            assert field.name in table.column_names

    def test_metadata_values(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        output = tmp_path / "meta.parquet"

        write_enrichment_metadata([("doc1", enrichment)], output)

        table = pq.read_table(str(output))
        row = table.to_pydict()
        assert row["source_hash"][0] == "doc1"
        assert row["doc_type_enriched"][0] == "other"
        assert row["jurisdiction"][0] == "AU"
        assert row["segment_count"][0] == 1
        assert row["person_count"][0] == 1
        assert row["location_count"][0] == 1
        assert row["term_count"][0] == 1

    def test_write_metadata_empty(self, tmp_path: Path) -> None:
        output = tmp_path / "meta_empty.parquet"
        write_enrichment_metadata([], output)
        assert output.exists()
        table = pq.read_table(str(output))
        assert table.num_rows == 0


class TestJoinKeyName:
    """One document-identity column name across every sidecar in the library.

    These three sidecars named theirs ``document_id`` until 0.6.0 with values
    byte-identical to the ``source_hash`` the others join on, so every consumer
    carried a note and a rename. The enumeration case below is what stops the
    next sidecar reintroducing a second name; it lives here because this module
    held the one that diverged.
    """

    # The register ingests bypass the NLP pipeline, hash no source document and
    # have no source_hash to carry — the one schema outside the convention.
    EXEMPT: ClassVar[set[str]] = {"REGISTER_MANIFEST_SCHEMA"}

    def _store_schemas(self) -> dict[str, pa.Schema]:
        import importlib
        import pkgutil

        import womblex.store as store_pkg

        found: dict[str, pa.Schema] = {}
        for mod_info in pkgutil.iter_modules(store_pkg.__path__):
            mod = importlib.import_module(f"womblex.store.{mod_info.name}")
            for name, obj in vars(mod).items():
                if isinstance(obj, pa.Schema) and name.isupper():
                    found[f"{mod_info.name}.{name}"] = obj
        return found

    def test_every_sidecar_joins_on_source_hash(self) -> None:
        schemas = self._store_schemas()
        assert schemas, "no store schemas found — did the enumeration break?"
        offenders = {
            name: schema.names
            for name, schema in schemas.items()
            if name.split(".")[-1] not in self.EXEMPT and "source_hash" not in schema.names
        }
        assert not offenders, f"schemas with no source_hash column: {offenders}"

    def test_no_sidecar_reintroduces_the_legacy_name(self) -> None:
        offenders = [
            name for name, schema in self._store_schemas().items()
            if LEGACY_HASH_COLUMN in schema.names
        ]
        assert not offenders, (
            f"{LEGACY_HASH_COLUMN} is the pre-0.6.0 name for source_hash; "
            f"found on {offenders}"
        )


class TestLegacyColumnShim:
    """A shard written before 0.6.0 reads back as one written after it.

    Removed in 0.7.0 along with ``LEGACY_HASH_COLUMN`` — from then a pre-rename
    shard is a missing-column error and must be re-enriched.
    """

    def _legacy(self, schema: pa.Schema) -> pa.Schema:
        return pa.schema([
            pa.field(LEGACY_HASH_COLUMN, f.type) if f.name == "source_hash" else f
            for f in schema
        ])

    def test_entities_shard_written_under_the_old_name(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        base = tmp_path / "batch-0001.parquet"
        write_enrichment_entities_shard([("hash1", enrichment)], base)
        current = read_enrichment_entities(base).to_pylist()

        legacy_table = pq.read_table(str(enrichment_entities_path_for(base)))
        legacy_table = legacy_table.rename_columns([
            LEGACY_HASH_COLUMN if n == "source_hash" else n for n in legacy_table.schema.names
        ]).cast(self._legacy(ENTITY_SCHEMA))
        pq.write_table(legacy_table, str(enrichment_entities_path_for(base)))

        assert read_enrichment_entities(base).to_pylist() == current

    def test_graph_edges_shard_written_under_the_old_name(self, tmp_path: Path) -> None:
        text = _make_text()
        graph = build_document_graph("hash1", _make_enrichment(text), [])
        base = tmp_path / "batch-0001.parquet"
        write_graph_edges_shard([("hash1", graph)], base)
        current = read_graph_edges(base).to_pylist()

        legacy_table = pq.read_table(str(graph_edges_path_for(base)))
        legacy_table = legacy_table.rename_columns([
            LEGACY_HASH_COLUMN if n == "source_hash" else n for n in legacy_table.schema.names
        ]).cast(self._legacy(GRAPH_EDGE_SCHEMA))
        pq.write_table(legacy_table, str(graph_edges_path_for(base)))

        assert read_graph_edges(base).to_pylist() == current
