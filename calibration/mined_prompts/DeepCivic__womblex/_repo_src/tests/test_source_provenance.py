"""Source provenance: the ingest root and the root-relative source path.

One case per acceptance criterion — root qualification, the manifest columns,
the namespaced footer, resolution after the corpus moves, the read-side
back-fill for shards written before the columns existed, and the stated
non-goal that masking never rewrites either.

Extraction round-trips use the synthetic budget-statement DOCX; the schema-level
cases build rows directly, as ``test_pii_stage`` does.
"""

from __future__ import annotations

import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests._synthetic import BUDGET_DOCX as _BUDGET_DOCX
from womblex.ingest.strategies_file import DocxExtractor
from womblex.store.output import MANIFEST_SCHEMA, _shard_paths, read_manifest, write_results
from womblex.store.source_provenance import (
    COLLECTION_ID_KEY,
    INGEST_ROOT_KEY,
    SOURCE_RELPATH_KEY,
    IngestProvenance,
    qualify_root,
    read_footer_provenance,
    relpath_under,
)

_REL = f"dfat/2025-26/{_BUDGET_DOCX.name}"


@pytest.fixture(scope="module")
def extraction():
    return DocxExtractor().extract_path(_BUDGET_DOCX)


@pytest.fixture
def corpus(tmp_path, extraction):
    """A nested corpus root holding the fixture at a known relative path."""
    root = tmp_path / "corpus"
    target = root / _REL
    target.parent.mkdir(parents=True)
    shutil.copy(_BUDGET_DOCX, target)
    return root, target


@pytest.fixture
def stamped(corpus, extraction, tmp_path):
    """A shard written with provenance declared against the corpus root."""
    root, target = corpus
    shard = tmp_path / "run" / "documents" / "batch-0001.parquet"
    write_results(
        [("budget", str(target), extraction)],
        shard,
        provenance=IngestProvenance.declare(root, "womblex-collection"),
    )
    return root, shard


class TestRootQualification:
    def test_local_root_becomes_an_absolute_file_uri(self, tmp_path):
        assert qualify_root(tmp_path) == f"file://{tmp_path}"

    def test_relative_root_is_resolved_not_left_relative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        assert qualify_root("./data") == f"file://{tmp_path / 'data'}"

    def test_object_store_root_keeps_its_scheme(self):
        assert qualify_root("s3://bucket/inbox/") == "s3://bucket/inbox"

    def test_an_undeclared_root_raises_rather_than_defaulting_to_cwd(self):
        with pytest.raises(ValueError, match="empty"):
            qualify_root("")

    def test_a_mistyped_scheme_is_refused(self):
        with pytest.raises(ValueError):
            qualify_root("s3:/bucket")


class TestRootPlusRelpath:
    def test_join_names_the_file_that_was_read(self, corpus):
        root, target = corpus
        rel = relpath_under(qualify_root(root), target)
        assert rel == _REL
        assert (root / rel).read_bytes() == target.read_bytes()

    def test_a_document_outside_the_root_is_refused_not_guessed(self, tmp_path, corpus):
        root, _ = corpus
        stray = tmp_path / "stray.docx"
        stray.write_bytes(b"x")
        with pytest.raises(ValueError, match="not under the declared ingest root"):
            relpath_under(qualify_root(root), stray)

    def test_all_three_manifest_columns_are_populated(self, stamped):
        root, shard = stamped
        row = read_manifest(shard).to_pylist()[0]
        assert (row["ingest_root"], row["source_relpath"]) == (f"file://{root}", _REL)
        assert row["collection_id"] == "womblex-collection"

    def test_the_corpus_can_move_and_the_relpath_still_resolves(self, stamped, tmp_path):
        root, shard = stamped
        rel = read_manifest(shard).to_pylist()[0]["source_relpath"]
        moved = tmp_path / "elsewhere"
        shutil.move(str(root), str(moved))
        assert (moved / rel).is_file()

    def test_without_a_declared_root_the_columns_are_empty_not_invented(
        self, extraction, tmp_path,
    ):
        shard = tmp_path / "out" / "batch-0001.parquet"
        write_results([("budget", str(_BUDGET_DOCX), extraction)], shard)
        row = read_manifest(shard).to_pylist()[0]
        assert (row["ingest_root"], row["source_relpath"]) == ("", "")


class TestFooterMetadata:
    def test_every_extraction_parquet_carries_the_namespaced_pair(self, stamped):
        root, shard = stamped
        for role, path in _shard_paths(shard).items():
            keys = pq.read_metadata(str(path)).metadata
            assert {INGEST_ROOT_KEY, SOURCE_RELPATH_KEY, COLLECTION_ID_KEY} <= {
                k.decode() for k in keys
            }, role
            prov = read_footer_provenance(keys)
            assert prov["ingest_root"] == f"file://{root}", role
            assert prov["source_relpath"] == [_REL], role

    def test_a_reader_ignoring_the_footer_reads_the_file_unchanged(self, stamped):
        _, shard = stamped
        assert read_manifest(shard).schema.equals(MANIFEST_SCHEMA)
        assert pq.read_table(str(_shard_paths(shard)["elements"])).num_rows > 0

    def test_a_multi_document_shard_lists_every_relpath(self, corpus, extraction, tmp_path):
        root, target = corpus
        second = root / "dfat" / "other.docx"
        shutil.copy(target, second)
        shard = tmp_path / "out" / "batch-0001.parquet"
        write_results(
            [("a", str(target), extraction), ("b", str(second), extraction)],
            shard,
            provenance=IngestProvenance.declare(root, "womblex-collection"),
        )
        prov = read_footer_provenance(
            pq.read_metadata(str(_shard_paths(shard)["manifest"])).metadata
        )
        assert prov["source_relpath"] == [_REL, "dfat/other.docx"]

    def test_the_consolidated_run_manifest_carries_columns_and_footer(self, stamped):
        from womblex.store.run_manifest import write_run_manifest

        root, shard = stamped
        manifest = write_run_manifest(shard.parent)
        assert pq.read_table(str(manifest)).to_pylist()[0]["ingest_root"] == f"file://{root}"
        prov = read_footer_provenance(pq.read_metadata(str(manifest)).metadata)
        assert prov["ingest_root"] == f"file://{root}"


class TestObjectStoreRoot:
    def test_an_explicit_key_mapping_carries_an_s3_root(self, extraction, tmp_path):
        """The distributed shape: a staged scratch path, an s3 root, explicit keys."""
        staged = tmp_path / "inputs" / "budget.docx"
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"staged")
        shard = tmp_path / "out" / "batch-0001.parquet"
        write_results(
            [("budget", str(staged), extraction)],
            shard,
            provenance=IngestProvenance.declare(
                "s3://bucket/inbox", "womblex-collection", relpaths={staged: _REL},
            ),
        )
        row = read_manifest(shard).to_pylist()[0]
        assert (row["ingest_root"], row["source_relpath"]) == ("s3://bucket/inbox", _REL)


class TestBackFill:
    def test_a_shard_written_before_the_columns_reads_back_with_them_empty(self, tmp_path):
        legacy = pa.schema([
            f for f in MANIFEST_SCHEMA if f.name not in ("ingest_root", "source_relpath")
        ])
        row = {f.name: None for f in legacy}
        row.update({"source_hash": "h", "doc_id": "d", "filename": "d.pdf", "status": "completed"})
        pq.write_table(
            pa.Table.from_pylist([row], schema=legacy),
            str(tmp_path / "batch-0001._manifest.parquet"),
        )
        out = read_manifest(tmp_path).to_pylist()[0]
        assert (out["ingest_root"], out["source_relpath"]) == ("", "")


class TestMaskingLeavesProvenanceAlone:
    """Stated non-goal: the manifest is not a masking surface.

    FOI source paths carry personal names, so masking them would be tempting.
    The `pii` stage must not: it writes its own sidecars and leaves the
    manifest's columns and footer byte-identical.
    """

    def test_the_pii_stage_leaves_manifest_bytes_untouched(self, stamped):
        from womblex.config import PIIConfig
        from womblex.pii.pii_stage import pii_shards
        from womblex.store.enrichment_output import ENTITY_SCHEMA, enrichment_entities_path_for
        from womblex.store.output import write_chunks
        from womblex.store.pii_output import read_clean_text

        root, shard = stamped
        source_hash = read_manifest(shard).to_pylist()[0]["source_hash"]
        write_chunks(
            [{
                "source_hash": source_hash, "chunk_index": 0,
                "text": "The officer Jane Doe signed the notice.",
                "start_char": 0, "end_char": 38, "content_type": "narrative",
                "has_redaction": False, "page_start": 1, "page_end": 1, "elem_order": None,
            }],
            shard,
        )
        pq.write_table(
            pa.Table.from_pylist(
                [{
                    "source_hash": source_hash, "entity_id": "e1", "entity_label": "person",
                    "name": "Jane Doe", "entity_type": "natural", "role": "other",
                    "mention_start": 12, "mention_end": 20, "chunk_index": -1,
                }],
                schema=ENTITY_SCHEMA,
            ),
            str(enrichment_entities_path_for(shard)),
        )

        manifest_path = _shard_paths(shard)["manifest"]
        before = manifest_path.read_bytes()
        pii_shards(shard.parent, PIIConfig(use_regex_backstop=False))

        assert manifest_path.read_bytes() == before
        prov = read_footer_provenance(pq.read_metadata(str(manifest_path)).metadata)
        assert prov["ingest_root"] == f"file://{root}"
        # The stage must actually have masked something, or "left the manifest
        # alone" is satisfied by a run that did nothing — which is what a
        # mis-keyed entity row quietly turns this into.
        masked = read_clean_text(shard).to_pylist()
        assert masked and "<PERSON_1>" in masked[0]["text"], masked
