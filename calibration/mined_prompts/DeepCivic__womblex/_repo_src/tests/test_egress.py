"""Tests for the egress bundle builder (store/egress.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from womblex.store.contract import CONTRACT_VERSION
from womblex.store.egress import CORPUS_DIRNAME, EGRESS_MANIFEST_FILENAME, build_bundle
from womblex.store.egress_output import SOURCE_INDEX_FILENAME, SOURCES_DIRNAME, raw_key_for
from womblex.store.output import ELEMENT_SCHEMA, MANIFEST_SCHEMA, _source_hash, _write_rows
from womblex.store.remote import RemoteStore
from womblex.store.run_manifest import write_run_manifest
from womblex.store.run_stamp import RunStamp
from womblex.store.source_provenance import qualify_root

_STAMP = RunStamp("run-A", "1.2.3", "c" * 40, "sha256:abc", "extract", preset="corpus")


def _manifest_row(source_hash: str, doc_id: str, filename: str, ext: str, *,
                   ingest_root: str, relpath: str, extraction_method: str = "native") -> dict:
    return {
        "source_hash": source_hash, "collection_id": "", "doc_id": doc_id,
        "ingest_root": ingest_root, "source_relpath": relpath, "filename": filename,
        "ext": ext, "extraction_method": extraction_method, "elements_count": 1,
        "table_cells_count": 0, "form_fields_count": 0, "status": "ok", "error": "",
        "extracted_at_iso": "2026-01-01T00:00:00Z", "parser_version": "1.0",
    }


def _build_run(tmp_path: Path) -> tuple[Path, Path]:
    """A finished local run plus its corpus directory, with one resolvable
    document, one duplicate-hash document, one unresolvable (missing file)
    document, and one records-hashed (unsupported basis) document."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "report.pdf").write_bytes(b"%PDF-1.4 report bytes")
    report_hash = _source_hash(str(corpus_dir / "report.pdf"))
    root = qualify_root(corpus_dir)

    run_root = tmp_path / "run"
    rows = [
        _manifest_row(report_hash, "doc-1", "report.pdf", ".pdf",
                      ingest_root=root, relpath="report.pdf"),
        # Deliberately mismatched ext from doc-1's — same bytes (same hash),
        # differently-recorded extension. The uploaded raw_key must come from
        # the actual resolved file (.pdf), not this row's claimed ext.
        _manifest_row(report_hash, "doc-1-dup", "report-copy.dat", ".dat",
                      ingest_root=root, relpath="report.pdf"),
        _manifest_row("missing" * 8, "doc-2", "gone.pdf", ".pdf",
                      ingest_root=root, relpath="gone.pdf"),
        _manifest_row("recid" * 8, "doc-3", "record-42.txt", ".txt",
                      ingest_root=root, relpath="record-42.txt", extraction_method="records"),
    ]
    return _write_run(run_root, rows), corpus_dir


def _write_run(run_root: Path, rows: list[dict]) -> Path:
    shard_dir = run_root / "documents"
    shard_dir.mkdir(parents=True, exist_ok=True)
    footer = _STAMP.footer_metadata()
    _write_rows([], shard_dir / "batch-0001.elements.parquet", ELEMENT_SCHEMA, metadata=footer)
    _write_rows(rows, shard_dir / "batch-0001._manifest.parquet", MANIFEST_SCHEMA, metadata=footer)
    write_run_manifest(shard_dir)
    return run_root


def test_missing_manifest_raises(tmp_path: Path):
    run_root = tmp_path / "run"
    (run_root / "documents").mkdir(parents=True)
    store = RemoteStore.from_uri(str(tmp_path / "dest"))

    with pytest.raises(FileNotFoundError, match="finalise"):
        build_bundle(run_root, store, run_id="run-A")


def test_corpus_is_mirrored_flat_under_the_bundle(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    build_bundle(run_root, store, run_id="run-A")

    bundle_corpus = dest / "run-A" / CORPUS_DIRNAME
    assert (bundle_corpus / "manifest.parquet").is_file()
    assert (bundle_corpus / "batch-0001.elements.parquet").is_file()
    assert (bundle_corpus / "batch-0001._manifest.parquet").is_file()


def test_resolved_source_is_copied_and_indexed(tmp_path: Path):
    run_root, corpus_dir = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    result = build_bundle(run_root, store, run_id="run-A")

    report_hash = _source_hash(str(corpus_dir / "report.pdf"))
    raw_key = raw_key_for(report_hash, ".pdf")
    assert (dest / "run-A" / raw_key).read_bytes() == (corpus_dir / "report.pdf").read_bytes()

    index = pq.read_table(str(dest / "run-A" / SOURCE_INDEX_FILENAME)).to_pylist()
    by_doc = {r["doc_id"]: r for r in index}
    assert by_doc["doc-1"]["status"] == "resolved"
    assert by_doc["doc-1"]["raw_key"] == raw_key
    assert by_doc["doc-2"]["status"] == "not_found"
    assert by_doc["doc-2"]["raw_key"] is None
    assert by_doc["doc-3"]["status"] == "unsupported_basis"
    assert result.documents == 4


def test_duplicate_hash_is_uploaded_once_but_indexed_per_document(tmp_path: Path):
    run_root, corpus_dir = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    result = build_bundle(run_root, store, run_id="run-A")

    index = pq.read_table(str(dest / "run-A" / SOURCE_INDEX_FILENAME)).to_pylist()
    by_doc = {r["doc_id"]: r for r in index}
    expected_key = raw_key_for(_source_hash(str(corpus_dir / "report.pdf")), ".pdf")
    # doc-1-dup's manifest row claims ext=".dat" (a stale/wrong record) but
    # shares doc-1's source_hash — its raw_key must still be the key that was
    # actually uploaded (derived from the resolved file), not a ".dat" key
    # nothing was ever written to.
    assert by_doc["doc-1"]["raw_key"] == expected_key
    assert by_doc["doc-1-dup"]["raw_key"] == expected_key
    assert (dest / "run-A" / expected_key).is_file()
    assert result.sources_copied == 1  # one upload, not two, for the shared hash


def test_source_index_is_stamped_with_the_exported_run_at_stage_egress(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    build_bundle(run_root, store, run_id="run-A")

    meta = pq.read_schema(str(dest / "run-A" / SOURCE_INDEX_FILENAME)).metadata
    assert meta[b"womblex.run_id"] == b"run-A"
    assert meta[b"womblex.stage"] == b"egress"


def test_egress_manifest_reports_the_export(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    result = build_bundle(run_root, store, run_id="run-A")

    descriptor = json.loads((dest / "run-A" / EGRESS_MANIFEST_FILENAME).read_text())
    assert descriptor["run_id"] == "run-A"
    assert descriptor["contract_version"] == CONTRACT_VERSION
    assert descriptor["documents"] == 4
    assert descriptor["sources_copied"] == result.sources_copied == 1
    assert descriptor["sources_by_status"]["not_found"] == 1
    assert descriptor["sources_by_status"]["unsupported_basis"] == 1


def test_corpus_only_export_skips_sources_entirely(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    result = build_bundle(run_root, store, run_id="run-A", include_sources=False)

    assert not (dest / "run-A" / "sources").exists()
    assert not (dest / "run-A" / SOURCE_INDEX_FILENAME).exists()
    assert (dest / "run-A" / EGRESS_MANIFEST_FILENAME).exists()
    assert result.sources_copied == 0
    assert result.sources_by_status == {}


def test_bundle_prefix_overrides_the_run_id_as_the_export_folder(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    build_bundle(run_root, store, run_id="run-A", bundle_prefix="handoff/2026-09-26")

    assert (dest / "handoff" / "2026-09-26" / CORPUS_DIRNAME / "manifest.parquet").is_file()
    assert not (dest / "run-A").exists()


@pytest.mark.parametrize("run_id", ["../escaped", "..", "a/../../b", "run\x00id"])
def test_a_run_id_that_would_escape_the_store_root_is_refused(tmp_path: Path, run_id: str):
    run_root, _ = _build_run(tmp_path)
    store = RemoteStore.from_uri(str(tmp_path / "dest"))

    with pytest.raises(ValueError, match="unsafe bundle destination"):
        build_bundle(run_root, store, run_id=run_id)

    # Nothing was written outside (or even inside) the intended destination.
    assert not (tmp_path / "escaped").exists()


def test_a_bundle_prefix_segment_that_would_escape_is_refused(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    store = RemoteStore.from_uri(str(tmp_path / "dest"))

    with pytest.raises(ValueError, match="unsafe bundle destination"):
        build_bundle(run_root, store, run_id="run-A", bundle_prefix="handoff/../../escaped")


def _one_doc_run(tmp_path: Path, filename: str, *, ingest_root: str,
                 source_hash: str = "h" * 64, extraction_method: str = "native") -> Path:
    row = _manifest_row(source_hash, "doc-1", filename, Path(filename).suffix.lower(),
                        ingest_root=ingest_root, relpath=filename,
                        extraction_method=extraction_method)
    return _write_run(tmp_path / "run", [row])


def _corpus_with(tmp_path: Path, name: str, data: bytes) -> Path:
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / name).write_bytes(data)
    return tmp_path / "corpus"


@pytest.mark.parametrize("ingest_root", ["s3://bucket/corpus", ""])
def test_a_run_with_no_local_root_is_refused_before_anything_is_written(
    tmp_path: Path, ingest_root: str,
):
    run_root = _one_doc_run(tmp_path, "a.pdf", ingest_root=ingest_root)

    with pytest.raises(ValueError, match="root"):
        build_bundle(run_root, RemoteStore.from_uri(str(tmp_path / "dest")), run_id="run-A")

    assert not (tmp_path / "dest").exists()


def test_a_records_only_run_exports_with_no_root_as_unsupported_basis(tmp_path: Path):
    run_root = _one_doc_run(tmp_path, "rec-1", ingest_root="", extraction_method="records")
    dest = tmp_path / "dest"

    result = build_bundle(run_root, RemoteStore.from_uri(str(dest)), run_id="run-A")

    index = pq.read_table(str(dest / "run-A" / SOURCE_INDEX_FILENAME)).to_pylist()
    assert [(r["status"], r["raw_key"]) for r in index] == [("unsupported_basis", None)]
    assert result.sources_by_status == {"unsupported_basis": 1}


def test_a_run_id_that_disagrees_with_the_run_stamp_is_refused(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)

    with pytest.raises(ValueError, match="does not match"):
        build_bundle(run_root, RemoteStore.from_uri(str(tmp_path / "dest")), run_id="run-B")

    assert not (tmp_path / "dest").exists()


def test_raw_key_extension_is_lowercased_like_the_manifest_ext(tmp_path: Path):
    corpus_dir = _corpus_with(tmp_path, "R.PDF", b"%PDF upper")
    source_hash = _source_hash(str(corpus_dir / "R.PDF"))
    run_root = _one_doc_run(tmp_path, "R.PDF", ingest_root=qualify_root(corpus_dir),
                            source_hash=source_hash)
    dest = tmp_path / "dest"

    build_bundle(run_root, RemoteStore.from_uri(str(dest)), run_id="run-A")

    (row,) = pq.read_table(str(dest / "run-A" / SOURCE_INDEX_FILENAME)).to_pylist()
    assert row["ext"] == ".pdf"
    assert row["raw_key"] == raw_key_for(source_hash, ".pdf")
    assert (dest / "run-A" / row["raw_key"]).is_file()


def test_a_hash_mismatch_is_reported_and_not_copied(tmp_path: Path):
    corpus_dir = _corpus_with(tmp_path, "a.pdf", b"%PDF re-saved with different bytes")
    run_root = _one_doc_run(tmp_path, "a.pdf", ingest_root=qualify_root(corpus_dir))
    dest = tmp_path / "dest"

    result = build_bundle(run_root, RemoteStore.from_uri(str(dest)), run_id="run-A")

    (row,) = pq.read_table(str(dest / "run-A" / SOURCE_INDEX_FILENAME)).to_pylist()
    assert (row["status"], row["raw_key"], result.sources_copied) == ("hash_mismatch", None, 0)


def test_documents_dir_is_mirrored_recursively_not_just_top_level_parquet(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    shard_dir = run_root / "documents"
    (shard_dir / "audit").mkdir()
    (shard_dir / "audit" / "notes.json").write_text('{"ok": true}')
    (shard_dir / "checkpoint.state").write_text("progress")
    dest = tmp_path / "dest"

    result = build_bundle(run_root, RemoteStore.from_uri(str(dest)), run_id="run-A")

    bundle_corpus = dest / "run-A" / CORPUS_DIRNAME
    assert (bundle_corpus / "audit" / "notes.json").read_text() == '{"ok": true}'
    assert (bundle_corpus / "checkpoint.state").read_text() == "progress"
    # manifest.parquet + 2 batch shards + the 2 extra files added above.
    assert result.corpus_files == 5


def test_a_failed_source_upload_does_not_abort_the_export(tmp_path: Path, monkeypatch):
    run_root, _ = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))
    original_upload = store.upload_file

    def flaky_upload(local_path: Path, rel: str) -> str:
        if f"/{SOURCES_DIRNAME}/" in f"/{rel}":
            raise OSError("simulated upload failure")
        return original_upload(local_path, rel)

    monkeypatch.setattr(store, "upload_file", flaky_upload)

    result = build_bundle(run_root, store, run_id="run-A")

    # The export completed: both sidecars landed despite the failed source.
    assert (dest / "run-A" / SOURCE_INDEX_FILENAME).is_file()
    assert (dest / "run-A" / EGRESS_MANIFEST_FILENAME).is_file()
    assert (dest / "run-A" / CORPUS_DIRNAME / "manifest.parquet").is_file()

    index = pq.read_table(str(dest / "run-A" / SOURCE_INDEX_FILENAME)).to_pylist()
    by_doc = {r["doc_id"]: r for r in index}
    assert by_doc["doc-1"]["status"] == "upload_failed"
    assert by_doc["doc-1"]["raw_key"] is None
    assert by_doc["doc-1-dup"]["status"] == "upload_failed"
    assert result.sources_copied == 0
    assert result.sources_by_status["upload_failed"] == 2

    descriptor = json.loads((dest / "run-A" / EGRESS_MANIFEST_FILENAME).read_text())
    assert descriptor["sources_by_status"]["upload_failed"] == 2


def test_stale_sources_from_a_previous_export_are_removed_on_re_export(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.pdf").write_bytes(b"%PDF-A")
    (corpus_dir / "b.pdf").write_bytes(b"%PDF-B")
    root = qualify_root(corpus_dir)
    hash_a = _source_hash(str(corpus_dir / "a.pdf"))
    hash_b = _source_hash(str(corpus_dir / "b.pdf"))

    run1 = _write_run(tmp_path / "run1", [
        _manifest_row(hash_a, "doc-a", "a.pdf", ".pdf", ingest_root=root, relpath="a.pdf"),
        _manifest_row(hash_b, "doc-b", "b.pdf", ".pdf", ingest_root=root, relpath="b.pdf"),
    ])
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))
    build_bundle(run1, store, run_id="run-A")

    key_a = raw_key_for(hash_a, ".pdf")
    key_b = raw_key_for(hash_b, ".pdf")
    assert (dest / "run-A" / key_a).is_file()
    assert (dest / "run-A" / key_b).is_file()

    # The corpus moved on: b.pdf is gone, and this run only ever saw a.pdf.
    run2 = _write_run(tmp_path / "run2", [
        _manifest_row(hash_a, "doc-a", "a.pdf", ".pdf", ingest_root=root, relpath="a.pdf"),
    ])
    build_bundle(run2, store, run_id="run-A")

    assert (dest / "run-A" / key_a).is_file()
    assert not (dest / "run-A" / key_b).exists()


def test_a_stale_delete_failure_does_not_abort_the_export(tmp_path: Path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.pdf").write_bytes(b"%PDF-A")
    (corpus_dir / "b.pdf").write_bytes(b"%PDF-B")
    root = qualify_root(corpus_dir)
    hash_a = _source_hash(str(corpus_dir / "a.pdf"))
    hash_b = _source_hash(str(corpus_dir / "b.pdf"))

    run1 = _write_run(tmp_path / "run1", [
        _manifest_row(hash_a, "doc-a", "a.pdf", ".pdf", ingest_root=root, relpath="a.pdf"),
        _manifest_row(hash_b, "doc-b", "b.pdf", ".pdf", ingest_root=root, relpath="b.pdf"),
    ])
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))
    build_bundle(run1, store, run_id="run-A")

    key_b = raw_key_for(hash_b, ".pdf")
    original_delete = store.delete

    def flaky_delete(rel: str) -> None:
        if rel.endswith(key_b):
            raise OSError("simulated delete failure")
        return original_delete(rel)

    monkeypatch.setattr(store, "delete", flaky_delete)

    # b.pdf dropped from the corpus, so its stale copy is targeted for
    # cleanup — but the delete itself fails, and the export must still finish.
    run2 = _write_run(tmp_path / "run2", [
        _manifest_row(hash_a, "doc-a", "a.pdf", ".pdf", ingest_root=root, relpath="a.pdf"),
    ])
    build_bundle(run2, store, run_id="run-A")

    assert (dest / "run-A" / SOURCE_INDEX_FILENAME).is_file()
    assert (dest / "run-A" / EGRESS_MANIFEST_FILENAME).is_file()


def test_a_failed_upload_does_not_delete_a_previously_copied_source(
    tmp_path: Path, monkeypatch,
):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.pdf").write_bytes(b"%PDF-A")
    root = qualify_root(corpus_dir)
    hash_a = _source_hash(str(corpus_dir / "a.pdf"))
    run_root = _write_run(tmp_path / "run", [
        _manifest_row(hash_a, "doc-a", "a.pdf", ".pdf", ingest_root=root, relpath="a.pdf"),
    ])
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))
    build_bundle(run_root, store, run_id="run-A")

    key_a = raw_key_for(hash_a, ".pdf")
    assert (dest / "run-A" / key_a).is_file()

    original_upload = store.upload_file

    def flaky_upload(local_path: Path, rel: str) -> str:
        if f"/{SOURCES_DIRNAME}/" in f"/{rel}":
            raise OSError("simulated upload failure")
        return original_upload(local_path, rel)

    monkeypatch.setattr(store, "upload_file", flaky_upload)
    build_bundle(run_root, store, run_id="run-A")

    # The re-export's own upload failed, but the file from the first,
    # successful export must not be treated as stale and deleted.
    assert (dest / "run-A" / key_a).is_file()
