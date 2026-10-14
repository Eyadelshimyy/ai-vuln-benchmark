"""Tests for womblex.store.output.

Exercises the writer / reader / integrity check round-trip on the synthetic
budget-statement DOCX (``tests/_synthetic.py``).
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests._synthetic import BUDGET_DOCX as _BUDGET_DOCX
from womblex.ingest.strategies_file import DocxExtractor
from womblex.store.output import (
    CHUNKS_SCHEMA,
    CHUNKS_SUFFIX,
    ELEMENT_SCHEMA,
    FORM_FIELDS_SCHEMA,
    MANIFEST_SCHEMA,
    TABLE_CELLS_SCHEMA,
    ShardVerificationError,
    _shard_paths,
    chunks_path_for,
    read_chunks,
    read_elements,
    read_form_fields,
    read_manifest,
    read_table_cells,
    verify_chunks_persistence,
    verify_shard_persistence,
    write_chunks,
    write_results,
)


@pytest.fixture(scope="module")
def budget_statement_extraction():
    return DocxExtractor().extract_path(_BUDGET_DOCX)


@pytest.fixture
def written_shard(tmp_path, budget_statement_extraction):
    shard = tmp_path / "batch-0001.parquet"
    write_results(
        [("budget", str(_BUDGET_DOCX), budget_statement_extraction)],
        shard,
        collection_id="test",
    )
    return shard


class TestShardLayout:
    def test_four_sibling_files_written(self, written_shard):
        paths = _shard_paths(written_shard)
        for role, p in paths.items():
            assert p.exists(), f"{role} not written: {p}"
            assert p.stat().st_size > 0, f"{role} is empty: {p}"

    def test_schemas_round_trip(self, written_shard):
        # Each file matches the canonical schema (read_* enforces).
        assert read_elements(written_shard).schema.equals(ELEMENT_SCHEMA)
        assert read_table_cells(written_shard).schema.equals(TABLE_CELLS_SCHEMA)
        assert read_form_fields(written_shard).schema.equals(FORM_FIELDS_SCHEMA)
        assert read_manifest(written_shard).schema.equals(MANIFEST_SCHEMA)


class TestVerbatimRoundTrip:
    def test_first_table_headers_match_source(self, written_shard, budget_statement_extraction):
        # First kind='table' element's row 0 (header row) reads back verbatim
        # against the legacy derived TableData.headers view.
        e_table = read_elements(written_shard)
        kinds = e_table.column("kind").to_pylist()
        first_table_pos = kinds.index("table")
        parent_order = e_table.column("elem_order")[first_table_pos].as_py()
        src_hash = e_table.column("source_hash")[first_table_pos].as_py()

        tc = read_table_cells(written_shard)
        header_cells = [
            (r["col"], r["value"])
            for r in tc.to_pylist()
            if r["source_hash"] == src_hash
            and r["parent_elem_order"] == parent_order
            and r["row"] == 0
        ]
        header_cells.sort()
        cell_headers = [v for _, v in header_cells]
        assert cell_headers == budget_statement_extraction.tables[0].headers


class TestIntegrity:
    def test_passes_on_real_fixture(self, written_shard):
        cumulative = verify_shard_persistence(
            written_shard, expected_docs=1, prev_total_size=0,
        )
        assert cumulative > 0

    def test_detects_missing_shard(self, written_shard, tmp_path):
        paths = _shard_paths(written_shard)
        paths["elements"].unlink()
        with pytest.raises(ShardVerificationError, match="shard missing"):
            verify_shard_persistence(written_shard, 1, 0)

    def test_detects_manifest_row_mismatch(self, written_shard):
        with pytest.raises(ShardVerificationError, match="manifest row count mismatch"):
            verify_shard_persistence(written_shard, expected_docs=99, prev_total_size=0)


class TestManifestColumns:
    def test_manifest_records_source_metadata(self, written_shard):
        m = read_manifest(written_shard)
        assert m.num_rows == 1
        row = m.to_pylist()[0]
        assert row["doc_id"] == "budget"
        assert row["filename"] == _BUDGET_DOCX.name
        assert row["ext"] == ".docx"
        assert row["extraction_method"] == "docx"
        assert row["elements_count"] > 0
        assert row["table_cells_count"] > 0
        assert row["status"] == "completed"
        assert row["error"] == ""
        assert row["collection_id"] == "test"
        assert len(row["source_hash"]) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# Chunks sidecar (B1)
# ---------------------------------------------------------------------------


_SAMPLE_CHUNK_ROWS = [
    {
        "source_hash": "h" * 64,
        "chunk_index": 0,
        "text": "first chunk",
        "start_char": 0,
        "end_char": 11,
        "content_type": "narrative",
        "has_redaction": False,
        "page_start": 1,
        "page_end": 1,
        "elem_order": None,
        "token_count": 2,
    },
    {
        "source_hash": "h" * 64,
        "chunk_index": 1,
        "text": "second <REDACTED>",
        "start_char": 11,
        "end_char": 28,
        "content_type": "narrative",
        "has_redaction": True,
        "page_start": 1,
        "page_end": 2,
        "elem_order": None,
        "token_count": 2,
    },
    {
        "source_hash": "h" * 64,
        "chunk_index": 2,
        "text": "| H |\n| --- |\n| v |",
        "start_char": 0,
        "end_char": 19,
        "content_type": "table",
        "has_redaction": False,
        "page_start": 3,
        "page_end": 3,
        "elem_order": 4,
        "token_count": 9,
    },
]


class TestChunksSchemaRoundTrip:
    def test_write_then_read(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        write_chunks(_SAMPLE_CHUNK_ROWS, base)
        target = chunks_path_for(base)
        assert target.exists() and target.stat().st_size > 0
        out = read_chunks(base)
        assert out.schema.equals(CHUNKS_SCHEMA)
        assert out.num_rows == 3
        row = out.to_pylist()[1]
        assert row["text"] == "second <REDACTED>"
        assert row["has_redaction"] is True
        assert row["page_end"] == 2

    def test_elem_order_anchors_table_chunks_only(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        write_chunks(_SAMPLE_CHUNK_ROWS, base)
        rows = read_chunks(base).to_pylist()
        narrative = [r for r in rows if r["content_type"] == "narrative"]
        table = [r for r in rows if r["content_type"] == "table"]
        assert all(r["elem_order"] is None for r in narrative)
        assert [r["elem_order"] for r in table] == [4]

    def test_empty_write_yields_schema_correct_file(self, tmp_path):
        base = tmp_path / "batch-0002.parquet"
        write_chunks([], base)
        out = read_chunks(base)
        assert out.schema.equals(CHUNKS_SCHEMA)
        assert out.num_rows == 0

    def test_directory_read_concatenates(self, tmp_path):
        write_chunks(_SAMPLE_CHUNK_ROWS, tmp_path / "batch-0001.parquet")
        write_chunks(_SAMPLE_CHUNK_ROWS[:1], tmp_path / "batch-0002.parquet")
        out = read_chunks(tmp_path)
        assert out.num_rows == 4

    def test_nullable_pages(self, tmp_path):
        # DOCX / spreadsheet sources omit page semantics.
        base = tmp_path / "batch-0001.parquet"
        write_chunks([{
            **_SAMPLE_CHUNK_ROWS[0],
            "page_start": None,
            "page_end": None,
        }], base)
        out = read_chunks(base).to_pylist()
        assert out[0]["page_start"] is None
        assert out[0]["page_end"] is None

    def test_pre_elem_order_shard_backfills(self, tmp_path):
        # Shards written before the elem_order column must stay readable —
        # the compat shim back-fills nulls instead of failing the read.
        legacy_schema = pa.schema([
            f for f in CHUNKS_SCHEMA if f.name != "elem_order"
        ])
        legacy_rows = [
            {k: v for k, v in r.items() if k != "elem_order"}
            for r in _SAMPLE_CHUNK_ROWS
        ]
        target = chunks_path_for(tmp_path / "batch-0001.parquet")
        pq.write_table(
            pa.Table.from_pylist(legacy_rows, schema=legacy_schema), str(target),
        )
        out = read_chunks(target)
        assert out.schema.equals(CHUNKS_SCHEMA)
        assert out.num_rows == len(_SAMPLE_CHUNK_ROWS)
        assert all(r["elem_order"] is None for r in out.to_pylist())

    def test_pre_token_count_shard_backfills(self, tmp_path):
        # Same compat convention for the token count: a run completed before
        # the column exists reads back with nulls rather than failing.
        legacy_schema = pa.schema([
            f for f in CHUNKS_SCHEMA if f.name != "token_count"
        ])
        legacy_rows = [
            {k: v for k, v in r.items() if k != "token_count"}
            for r in _SAMPLE_CHUNK_ROWS
        ]
        target = chunks_path_for(tmp_path / "batch-0001.parquet")
        pq.write_table(
            pa.Table.from_pylist(legacy_rows, schema=legacy_schema), str(target),
        )
        out = read_chunks(target)
        assert out.schema.equals(CHUNKS_SCHEMA)
        assert all(r["token_count"] is None for r in out.to_pylist())

    def test_token_count_round_trips(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        write_chunks(_SAMPLE_CHUNK_ROWS, base)
        rows = read_chunks(base).to_pylist()
        assert [r["token_count"] for r in rows] == [2, 2, 9]

    def test_genuinely_missing_column_still_raises(self, tmp_path):
        # The shim is scoped to elem_order; other gaps remain hard errors.
        broken_schema = pa.schema([
            f for f in CHUNKS_SCHEMA if f.name != "content_type"
        ])
        rows = [
            {k: v for k, v in r.items() if k != "content_type"}
            for r in _SAMPLE_CHUNK_ROWS
        ]
        target = chunks_path_for(tmp_path / "batch-0001.parquet")
        pq.write_table(pa.Table.from_pylist(rows, schema=broken_schema), str(target))
        with pytest.raises(ValueError, match="content_type"):
            read_chunks(target)


class TestChunksIntegrity:
    def test_passes_on_valid_shard(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        write_chunks(_SAMPLE_CHUNK_ROWS, base)
        size = verify_chunks_persistence(
            base, expected_source_hashes={"h" * 64}, prev_total_size=0,
        )
        assert size > 0

    def test_missing_file(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        with pytest.raises(ShardVerificationError, match="missing"):
            verify_chunks_persistence(base, {"h" * 64}, 0)

    def test_unknown_source_hash(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        write_chunks(_SAMPLE_CHUNK_ROWS, base)
        with pytest.raises(ShardVerificationError, match="unknown source_hash"):
            verify_chunks_persistence(base, {"different"}, 0)

    def test_unexpected_rows_when_no_hashes_expected(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        write_chunks(_SAMPLE_CHUNK_ROWS, base)
        with pytest.raises(ShardVerificationError, match="no source_hashes were expected"):
            verify_chunks_persistence(base, set(), 0)

    def test_empty_chunks_with_empty_expected_passes(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        write_chunks([], base)
        size = verify_chunks_persistence(base, set(), 0)
        assert size > 0  # zero-row parquet still has a header

    def test_detects_directory_shrink(self, tmp_path):
        base = tmp_path / "batch-0001.parquet"
        write_chunks(_SAMPLE_CHUNK_ROWS, base)
        big_size = chunks_path_for(base).stat().st_size + 1000
        with pytest.raises(ShardVerificationError, match="shrank"):
            verify_chunks_persistence(base, {"h" * 64}, big_size)


def test_chunks_path_for_uses_stem(tmp_path):
    base = tmp_path / "batch-0042.parquet"
    target = chunks_path_for(base)
    assert target.name == f"batch-0042{CHUNKS_SUFFIX}"
    assert target.parent == tmp_path


def test_content_digest_is_stable_across_extractions(tmp_path):
    """Same source, same code: equal digest and rows; bytes and timestamps may differ."""
    manifests, elements = [], []
    for name in ("a", "b"):
        result = DocxExtractor().extract_path(_BUDGET_DOCX)
        shard = tmp_path / name / "batch-0001.parquet"
        write_results([("budget", str(_BUDGET_DOCX), result)], shard, collection_id="t")
        manifests.append(read_manifest(shard).to_pylist()[0])
        elements.append(read_elements(shard).to_pylist())
    assert manifests[0]["content_digest"]
    assert manifests[0]["content_digest"] == manifests[1]["content_digest"]
    assert elements[0] == elements[1]


def test_content_digest_changes_with_content():
    from womblex.ingest.elements import Element
    from womblex.store.content_digest import content_digest

    a = Element(kind="paragraph", order=0, extractor="t", text="alpha")
    b = Element(kind="paragraph", order=0, extractor="t", text="alpha!")
    assert content_digest([a]) != content_digest([b])
    assert content_digest([a]) == content_digest([Element(kind="paragraph", order=0, extractor="t", text="alpha")])


def test_content_digest_covers_sheet_cells_and_meta():
    from womblex.ingest.elements import Element
    from womblex.store.content_digest import content_digest

    def cell(value: str) -> Element:
        return Element(kind="sheet_cell", order=0, extractor="xlsx", sheet="S", row=1, col=0, value=value)

    assert content_digest([cell("100")]) != content_digest([cell("999")])
    m1 = Element(kind="sheet_meta", order=0, extractor="xlsx", sheet="S", meta={"preamble": "a"})
    m2 = Element(kind="sheet_meta", order=0, extractor="xlsx", sheet="S", meta={"preamble": "b"})
    assert content_digest([m1]) != content_digest([m2])
    t1 = Element(kind="table", order=0, extractor="t", cells=[], header_rows=[0])
    t2 = Element(kind="table", order=0, extractor="t", cells=[], header_rows=[1])
    assert content_digest([t1]) != content_digest([t2])


def test_manifest_without_content_digest_reads_back_null(tmp_path):
    shard = tmp_path / "batch-0001._manifest.parquet"
    legacy = pa.schema([f for f in MANIFEST_SCHEMA if f.name != "content_digest"])
    row = {f.name: ("x" if pa.types.is_string(f.type) else 0) for f in legacy}
    pq.write_table(pa.Table.from_pylist([row], schema=legacy), str(shard))
    from womblex.store.output import _read_shard

    assert _read_shard(shard, "manifest").to_pylist()[0]["content_digest"] is None
