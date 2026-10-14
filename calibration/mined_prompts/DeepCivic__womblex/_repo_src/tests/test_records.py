"""Tests for the pre-extracted records ingest (``ingest/records.py``).

Offline, no Isaacus. Confirms records become the same element-shard layout
the file extractors produce (so enrich/chunk/embed run over them unchanged),
the content-addressed source_hash, provenance sidecar + consolidated corpus
manifest, and that the reassembled narrative round-trips the block split.
"""

from __future__ import annotations

from womblex.ingest.records import (
    RecordFieldMapping,
    ingest_records,
    records_source_hash,
    split_text_blocks,
)
from womblex.process.chunker import reassemble_narrative
from womblex.store.output import read_elements, read_manifest
from womblex.store.provenance_output import read_provenance, write_corpus_manifest

_MAPPING = RecordFieldMapping(
    id_field="version_id",
    text_field="text",
    provenance_fields=["jurisdiction", "type", "citation"],
    collection_id="oalc-test",
)


def _records() -> list[dict]:
    return [
        {
            "version_id": "nsw:2020/case-1",
            "text": "First paragraph of the judgment.\n\nSecond paragraph here.",
            "jurisdiction": "new_south_wales",
            "type": "decision",
            "citation": "Foo v Bar [2020] NSWLEC 1",
        },
        {
            "version_id": "nsw:2020/case-2",
            "text": "Single block only.",
            "jurisdiction": "new_south_wales",
            "type": "decision",
            "citation": "Baz v Qux [2020] NSWSC 2",
        },
        {
            "version_id": "nsw:2020/empty",
            "text": "   \n\n  ",
            "jurisdiction": "new_south_wales",
            "type": "decision",
            "citation": "Empty [2020] NSWSC 3",
        },
    ]


class TestSourceHash:
    def test_deterministic_and_content_addressed(self):
        h1 = records_source_hash("id-1", "hello")
        h2 = records_source_hash("id-1", "hello")
        h3 = records_source_hash("id-1", "hello world")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64  # sha256 hex


class TestBlockSplit:
    def test_splits_on_blank_lines(self):
        assert split_text_blocks("a\n\nb\n\nc") == ["a", "b", "c"]

    def test_collapses_long_newline_runs_and_drops_empties(self):
        assert split_text_blocks("a\n\n\n\nb") == ["a", "b"]
        assert split_text_blocks("\n\na\n\n") == ["a"]

    def test_empty_text_yields_no_blocks(self):
        assert split_text_blocks("") == []
        assert split_text_blocks("   \n\n  ") == []


class TestIngestRecords:
    def test_writes_shard_layout(self, tmp_path):
        out = tmp_path / "documents"
        result = ingest_records(_records(), out, _MAPPING, batch_size=500)
        assert result.batches_written == 1
        assert result.docs_ingested == 3
        assert result.empty_docs == 1

        base = out / "batch-0001.parquet"
        for suffix in ("elements", "_manifest", "provenance"):
            assert (out / f"batch-0001.{suffix}.parquet").exists()

        manifest = read_manifest(base).to_pylist()
        assert len(manifest) == 3
        # doc_id is the record id, so checkpoints key on version_id.
        assert {m["doc_id"] for m in manifest} == {
            "nsw:2020/case-1", "nsw:2020/case-2", "nsw:2020/empty",
        }
        empty = next(m for m in manifest if m["doc_id"] == "nsw:2020/empty")
        assert empty["status"] == "empty" and empty["elements_count"] == 0

    def test_narrative_round_trips_block_split(self, tmp_path):
        out = tmp_path / "documents"
        ingest_records(_records()[:1], out, _MAPPING)
        base = out / "batch-0001.parquet"
        elem_rows = read_elements(base).to_pylist()
        from womblex.ingest.elements import Element

        elems = [
            Element(order=r["elem_order"], kind=r["kind"], extractor=r["extractor"], text=r["text"])
            for r in sorted(elem_rows, key=lambda r: r["elem_order"])
        ]
        narrative, _ = reassemble_narrative(elems)
        # \n\n-delimited source → reassembled narrative is byte-identical.
        assert narrative == "First paragraph of the judgment.\n\nSecond paragraph here."

    def test_source_hash_joins_elements_and_provenance(self, tmp_path):
        out = tmp_path / "documents"
        ingest_records(_records(), out, _MAPPING)
        base = out / "batch-0001.parquet"
        elem_hashes = set(read_elements(base).column("source_hash").to_pylist())
        prov = read_provenance(base).to_pylist()
        prov_by_hash = {r["source_hash"]: r for r in prov}
        # every element hash resolves to a provenance row carrying the metadata
        assert elem_hashes <= set(prov_by_hash)
        row = prov_by_hash[records_source_hash("nsw:2020/case-1", _records()[0]["text"])]
        assert row["jurisdiction"] == "new_south_wales"
        assert row["citation"] == "Foo v Bar [2020] NSWLEC 1"

    def test_batching_and_start_batch(self, tmp_path):
        out = tmp_path / "documents"
        r1 = ingest_records(_records(), out, _MAPPING, batch_size=2)
        assert r1.batches_written == 2
        assert (out / "batch-0001.elements.parquet").exists()
        assert (out / "batch-0002.elements.parquet").exists()
        # extend the ingest past existing shards without overwriting
        r2 = ingest_records(_records()[:1], out, _MAPPING, batch_size=2, start_batch=r1.batches_written + 1)
        assert (out / "batch-0003.elements.parquet").exists()
        assert r2.docs_ingested == 1

    def test_corpus_manifest_consolidation(self, tmp_path):
        out = tmp_path / "documents"
        ingest_records(_records(), out, _MAPPING, batch_size=2)
        manifest_path = write_corpus_manifest(out)
        assert manifest_path == out.parent / "manifest.parquet"
        import pyarrow.parquet as pq

        table = pq.read_table(str(manifest_path))
        assert table.num_rows == 3
        assert "citation" in table.schema.names
        assert "source_hash" in table.schema.names
