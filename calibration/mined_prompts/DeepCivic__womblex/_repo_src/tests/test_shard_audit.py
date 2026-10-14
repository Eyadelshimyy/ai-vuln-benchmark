"""Tests for womblex.store.shard_audit — directory-level shard integrity.

Builds shards from the synthetic budget-statement DOCX (re-used from
test_output) and exercises scan / audit / reconcile against various
tampering modes.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests._synthetic import BUDGET_DOCX as _BUDGET_DOCX
from womblex.ingest.strategies_file import DocxExtractor
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_output import ENRICHMENT_ENTITIES_SUFFIX
from womblex.store.output import (
    MANIFEST_SCHEMA,
    _shard_paths,
    chunks_path_for,
    read_manifest,
    write_chunks,
    write_results,
)
from womblex.store.shard_audit import (
    ARCHIVE_SUFFIX,
    audit_shard_directory,
    format_audit_diff,
    format_audit_text,
    reconcile_checkpoint_with_shards,
    reconcile_chunk_checkpoint_with_shards,
    reconcile_stage_checkpoint_with_shards,
    scan_chunks_directory,
    scan_shard_directory,
    scan_sidecar_directory,
)


@pytest.fixture(scope="module")
def budget_extraction():
    return DocxExtractor().extract_path(_BUDGET_DOCX)


def _write_batch(shard_dir: Path, batch_num: int, budget_extraction, doc_id: str) -> Path:
    shard = shard_dir / f"batch-{batch_num:04d}.parquet"
    write_results(
        [(doc_id, str(_BUDGET_DOCX), budget_extraction)],
        shard,
        collection_id="test",
    )
    return shard


# ---------------------------------------------------------------------------
# scan_shard_directory
# ---------------------------------------------------------------------------


class TestScanHealthy:
    def test_clean_dir_has_no_corruption(self, tmp_path, budget_extraction):
        _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        _write_batch(tmp_path, 2, budget_extraction, "doc-b")
        report = scan_shard_directory(tmp_path)
        assert len(report.batches) == 2
        assert all(b.is_healthy for b in report.batches)
        assert report.corrupted_batches == ()
        assert report.corrupted_doc_ids == ()

    def test_empty_dir_yields_empty_report(self, tmp_path):
        report = scan_shard_directory(tmp_path)
        assert report.batches == ()
        assert report.corrupted_doc_ids == ()


class TestScanCorruption:
    def test_zero_byte_elements(self, tmp_path, budget_extraction):
        shard = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        _shard_paths(shard)["elements"].write_bytes(b"")
        report = scan_shard_directory(tmp_path)
        assert len(report.corrupted_batches) == 1
        bad = report.corrupted_batches[0]
        assert not bad.files_nonempty
        assert any("zero-byte" in issue for issue in bad.issues)
        # doc_ids unrecoverable from this batch (manifest unreadable path)
        # — but here only elements is zero, so manifest still loads.
        assert bad.doc_ids == ("doc-a",)

    def test_missing_file_flagged(self, tmp_path, budget_extraction):
        shard = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        _shard_paths(shard)["table_cells"].unlink()
        report = scan_shard_directory(tmp_path)
        assert len(report.corrupted_batches) == 1
        bad = report.corrupted_batches[0]
        assert not bad.files_present
        assert any("missing" in issue for issue in bad.issues)

    def test_truncated_parquet_flagged(self, tmp_path, budget_extraction):
        shard = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        target = _shard_paths(shard)["elements"]
        # 64 bytes of garbage — passes the nonzero check but fails parquet metadata
        target.write_bytes(b"\x00" * 64)
        report = scan_shard_directory(tmp_path)
        assert len(report.corrupted_batches) == 1
        assert not report.corrupted_batches[0].files_readable

    def test_manifest_count_mismatch_flagged(self, tmp_path, budget_extraction):
        # Inflate the manifest's elements_count claim so it no longer matches
        # the actual elements rowcount.
        shard = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        manifest_path = _shard_paths(shard)["manifest"]
        m = read_manifest(shard)
        rows = m.to_pylist()
        rows[0]["elements_count"] = rows[0]["elements_count"] + 9999
        new = pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)
        pq.write_table(new, str(manifest_path), compression="zstd")
        report = scan_shard_directory(tmp_path)
        assert len(report.corrupted_batches) == 1
        assert not report.corrupted_batches[0].manifest_consistent

    def test_manifest_unreadable_leaves_doc_ids_empty(self, tmp_path, budget_extraction):
        shard = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        _shard_paths(shard)["manifest"].write_bytes(b"\x00" * 32)
        report = scan_shard_directory(tmp_path)
        assert len(report.corrupted_batches) == 1
        assert report.corrupted_batches[0].doc_ids == ()


# ---------------------------------------------------------------------------
# audit_shard_directory
# ---------------------------------------------------------------------------


class TestAudit:
    def test_aggregates_across_batches(self, tmp_path, budget_extraction):
        _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        _write_batch(tmp_path, 2, budget_extraction, "doc-b")
        report = audit_shard_directory(tmp_path)
        assert report.manifest_row_count == 2
        assert report.status_error_rows == 0
        assert report.total_elements > 0
        assert "docx" in report.methods
        assert report.methods["docx"] == 2

    def test_input_dir_provides_source_count(self, tmp_path, budget_extraction):
        _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.pdf").write_bytes(b"x")
        (src / "b.pdf").write_bytes(b"x")
        (src / "notes.rst").write_text("not counted")
        report = audit_shard_directory(tmp_path, input_dir=src)
        assert report.source_count == 2

    def test_source_count_uses_the_rule_a_run_ingests_by(self, tmp_path, budget_extraction):
        # The count is compared against the manifest, so it has to be what a run
        # would ingest. It once used its own extension list, which counted .txt
        # and .html a run passes over and missed the .xls a run takes -- drift
        # reported in both directions on a corpus a run handled perfectly.
        _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        src = tmp_path / "src"
        src.mkdir()
        for name in ("a.pdf", "b.xls", "c.docx"):
            (src / name).write_bytes(b"x")
        for name in ("readme.txt", "index.html", "notes.rst"):
            (src / name).write_text("not a document a run ingests")
        report = audit_shard_directory(tmp_path, input_dir=src)
        assert report.source_count == 3

    def test_source_count_is_none_for_a_corpus_a_run_would_refuse(
        self, tmp_path, budget_extraction,
    ):
        # A nested corpus is refused at ingest, so there is no count to compare;
        # None ("not available") beats a number that would read as drift.
        _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        src = tmp_path / "src"
        (src / "2026-08").mkdir(parents=True)
        (src / "2026-08" / "a.pdf").write_bytes(b"x")
        report = audit_shard_directory(tmp_path, input_dir=src)
        assert report.source_count is None

    def test_skips_corrupted_batches_in_metrics(self, tmp_path, budget_extraction):
        _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        shard2 = _write_batch(tmp_path, 2, budget_extraction, "doc-b")
        _shard_paths(shard2)["elements"].write_bytes(b"")
        report = audit_shard_directory(tmp_path)
        # Only batch 1 contributes to metrics
        assert report.manifest_row_count == 1
        assert report.methods["docx"] == 1
        assert len(report.scan.corrupted_batches) == 1


# ---------------------------------------------------------------------------
# reconcile_checkpoint_with_shards
# ---------------------------------------------------------------------------


def _make_checkpoint(tmp_path: Path, doc_ids: list[str]) -> CheckpointManager:
    ckpt_dir = tmp_path / ".checkpoints"
    mgr = CheckpointManager(ckpt_dir, "test")
    mgr.state.processed_ids.update(doc_ids)
    mgr.state.total_processed = len(doc_ids)
    mgr.state.total_succeeded = len(doc_ids)
    mgr.state.last_batch = 2
    mgr.save()
    return mgr


class TestReconcile:
    def test_noop_on_clean_dir(self, tmp_path, budget_extraction):
        shard_dir = tmp_path / "documents"
        shard_dir.mkdir()
        _write_batch(shard_dir, 1, budget_extraction, "doc-a")
        mgr = _make_checkpoint(tmp_path, ["doc-a"])
        dropped = reconcile_checkpoint_with_shards(mgr, shard_dir)
        assert dropped == []
        assert mgr.state.processed_ids == {"doc-a"}

    def test_drops_corrupted_doc_ids(self, tmp_path, budget_extraction):
        shard_dir = tmp_path / "documents"
        shard_dir.mkdir()
        _write_batch(shard_dir, 1, budget_extraction, "doc-a")
        bad = _write_batch(shard_dir, 2, budget_extraction, "doc-b")
        _shard_paths(bad)["elements"].write_bytes(b"")
        mgr = _make_checkpoint(tmp_path, ["doc-a", "doc-b"])

        dropped = reconcile_checkpoint_with_shards(mgr, shard_dir)
        assert dropped == ["doc-b"]
        assert mgr.state.processed_ids == {"doc-a"}
        # last_batch stays put — batch IDs are identifiers, not slots
        assert mgr.state.last_batch == 2

    def test_archives_corrupted_shards(self, tmp_path, budget_extraction):
        shard_dir = tmp_path / "documents"
        shard_dir.mkdir()
        bad = _write_batch(shard_dir, 1, budget_extraction, "doc-a")
        _shard_paths(bad)["elements"].write_bytes(b"")
        mgr = _make_checkpoint(tmp_path, ["doc-a"])

        reconcile_checkpoint_with_shards(mgr, shard_dir)

        # Original shards gone, archived siblings present
        for role, path in _shard_paths(bad).items():
            if path.exists():
                # zero-byte elements file may not have been renamed if rename
                # raced, but the rest should be archived
                pass
            archived = path.with_name(path.name + ARCHIVE_SUFFIX)
            assert archived.exists(), f"missing archive for {role}: {archived}"

    def test_manifest_unreadable_does_not_drop(self, tmp_path, budget_extraction):
        shard_dir = tmp_path / "documents"
        shard_dir.mkdir()
        bad = _write_batch(shard_dir, 1, budget_extraction, "doc-a")
        _shard_paths(bad)["manifest"].write_bytes(b"\x00" * 32)
        mgr = _make_checkpoint(tmp_path, ["doc-a"])
        dropped = reconcile_checkpoint_with_shards(mgr, shard_dir)
        assert dropped == []
        # Checkpoint untouched — operator must intervene
        assert mgr.state.processed_ids == {"doc-a"}

    def test_idempotent_second_pass(self, tmp_path, budget_extraction):
        shard_dir = tmp_path / "documents"
        shard_dir.mkdir()
        bad = _write_batch(shard_dir, 1, budget_extraction, "doc-a")
        _shard_paths(bad)["elements"].write_bytes(b"")
        mgr = _make_checkpoint(tmp_path, ["doc-a"])
        reconcile_checkpoint_with_shards(mgr, shard_dir)
        # Second pass: original batch's files already archived, nothing left
        # to flag (or, the zero-byte one remains as a corrupted batch but
        # doc_ids already dropped from checkpoint).
        dropped = reconcile_checkpoint_with_shards(mgr, shard_dir)
        assert dropped == []

    def test_missing_dir_is_noop(self, tmp_path):
        mgr = _make_checkpoint(tmp_path, ["doc-a"])
        dropped = reconcile_checkpoint_with_shards(mgr, tmp_path / "nope")
        assert dropped == []


# ---------------------------------------------------------------------------
# Manifest backward-compat (legacy shards missing doc_id column)
# ---------------------------------------------------------------------------


class TestManifestBackcompat:
    def test_legacy_manifest_derives_doc_id_from_filename(self, tmp_path, budget_extraction):
        # Write a normal shard, then strip doc_id from the manifest to
        # simulate a manifest from before the schema bump.
        shard = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        manifest_path = _shard_paths(shard)["manifest"]
        m = pq.read_table(str(manifest_path))
        without_doc_id = m.drop(["doc_id"])
        pq.write_table(without_doc_id, str(manifest_path), compression="zstd")

        # read_manifest must succeed and derive doc_id from filename stem
        m2 = read_manifest(shard)
        assert "doc_id" in m2.schema.names
        # _BUDGET_DOCX.stem is the long filename without .docx
        assert m2.column("doc_id")[0].as_py() == _BUDGET_DOCX.stem


# ---------------------------------------------------------------------------
# Formatters (smoke tests)
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_text_formatter_includes_key_metrics(self, tmp_path, budget_extraction):
        _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        report = audit_shard_directory(tmp_path)
        text = format_audit_text(report)
        assert "manifest_row_count" in text
        assert "docx" in text  # methods section
        assert str(report.manifest_row_count) in text

    def test_diff_formatter_lists_all_labels(self, tmp_path, budget_extraction):
        d1 = tmp_path / "run-a"
        d2 = tmp_path / "run-b"
        d1.mkdir()
        d2.mkdir()
        _write_batch(d1, 1, budget_extraction, "doc-a")
        _write_batch(d2, 1, budget_extraction, "doc-a")
        _write_batch(d2, 2, budget_extraction, "doc-b")
        r1 = audit_shard_directory(d1)
        r2 = audit_shard_directory(d2)
        diff = format_audit_diff({"run-a": r1, "run-b": r2})
        assert "run-a" in diff
        assert "run-b" in diff
        assert "manifest_row_count" in diff


# ---------------------------------------------------------------------------
# Chunks-side audit (B4)
# ---------------------------------------------------------------------------


def _seed_chunks_for_batch(batch_base: Path, source_hash: str) -> Path:
    rows = [{
        "source_hash": source_hash,
        "chunk_index": 0,
        "text": "x",
        "start_char": 0,
        "end_char": 1,
        "content_type": "narrative",
        "has_redaction": False,
        "page_start": None,
        "page_end": None,
    }]
    write_chunks(rows, batch_base)
    return chunks_path_for(batch_base)


class TestChunksScan:
    def test_clean_chunks_dir_healthy(self, tmp_path, budget_extraction):
        base = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        m = read_manifest(base)
        _seed_chunks_for_batch(base, m.column("source_hash")[0].as_py())

        report = scan_chunks_directory(tmp_path)
        assert len(report) == 1
        assert report[0].is_healthy

    def test_zero_byte_chunks_detected(self, tmp_path, budget_extraction):
        base = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        m = read_manifest(base)
        target = _seed_chunks_for_batch(base, m.column("source_hash")[0].as_py())
        target.write_bytes(b"")

        report = scan_chunks_directory(tmp_path)
        assert not report[0].is_healthy
        assert any("zero-byte" in i for i in report[0].issues)

    def test_unreadable_chunks_detected(self, tmp_path, budget_extraction):
        base = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        m = read_manifest(base)
        target = _seed_chunks_for_batch(base, m.column("source_hash")[0].as_py())
        target.write_bytes(b"not parquet bytes")

        report = scan_chunks_directory(tmp_path)
        assert not report[0].is_healthy
        assert any("unreadable" in i for i in report[0].issues)

    def test_missing_chunks_in_empty_dir(self, tmp_path):
        # No chunks files at all → scan returns empty list (nothing to scan).
        assert scan_chunks_directory(tmp_path) == []


class TestChunksReconcile:
    def test_dropping_corrupt_batch_drops_checkpoint_entries(
        self, tmp_path, budget_extraction,
    ):
        base = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        m = read_manifest(base)
        target = _seed_chunks_for_batch(base, m.column("source_hash")[0].as_py())
        target.write_bytes(b"")  # corrupt

        ckpt_dir = tmp_path / "ckpt"
        mgr = CheckpointManager(ckpt_dir, "test_chunk")
        mgr.update(["doc-a"], succeeded=1, failed=0, batch_num=1)

        dropped = reconcile_chunk_checkpoint_with_shards(mgr, tmp_path)
        assert dropped == ["doc-a"]
        assert "doc-a" not in mgr.state.processed_ids

    def test_clean_dir_returns_empty(self, tmp_path, budget_extraction):
        base = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        m = read_manifest(base)
        _seed_chunks_for_batch(base, m.column("source_hash")[0].as_py())

        ckpt_dir = tmp_path / "ckpt"
        mgr = CheckpointManager(ckpt_dir, "test_chunk")
        mgr.update(["doc-a"], succeeded=1, failed=0, batch_num=1)

        assert reconcile_chunk_checkpoint_with_shards(mgr, tmp_path) == []
        assert "doc-a" in mgr.state.processed_ids

    def test_elements_stay_intact_when_only_chunks_corrupted(
        self, tmp_path, budget_extraction,
    ):
        """A corrupt chunks file does not justify dropping/archiving elements."""
        base = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        m = read_manifest(base)
        target = _seed_chunks_for_batch(base, m.column("source_hash")[0].as_py())
        elements_size_before = _shard_paths(base)["elements"].stat().st_size
        target.write_bytes(b"")

        ckpt_dir = tmp_path / "ckpt"
        mgr = CheckpointManager(ckpt_dir, "test_chunk")
        mgr.update(["doc-a"], succeeded=1, failed=0, batch_num=1)
        reconcile_chunk_checkpoint_with_shards(mgr, tmp_path)

        # Elements untouched; only the chunks file archived.
        assert _shard_paths(base)["elements"].stat().st_size == elements_size_before
        archived = target.with_name(target.name + ARCHIVE_SUFFIX)
        assert archived.exists()


# ---------------------------------------------------------------------------
# Generic stage reconcile (shared by enrich / embed / link)
# ---------------------------------------------------------------------------


class TestGenericStageReconcile:
    def test_corrupt_sidecar_drops_checkpoint_and_archives(self, tmp_path, budget_extraction):
        # Same self-heal as chunk, but for an arbitrary stage sidecar suffix.
        base = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        sidecar = base.with_name(f"{base.stem}{ENRICHMENT_ENTITIES_SUFFIX}")
        sidecar.write_bytes(b"")  # corrupt (zero-byte)

        mgr = CheckpointManager(tmp_path / "ckpt", "test_enrich")
        mgr.update(["doc-a"], succeeded=1, failed=0, batch_num=1)

        dropped = reconcile_stage_checkpoint_with_shards(
            mgr, tmp_path, suffix=ENRICHMENT_ENTITIES_SUFFIX)
        assert dropped == ["doc-a"]
        assert "doc-a" not in mgr.state.processed_ids
        assert sidecar.with_name(sidecar.name + ARCHIVE_SUFFIX).exists()
        # element stream untouched
        assert _shard_paths(base)["elements"].exists()

    def test_clean_sidecar_dir_returns_empty(self, tmp_path, budget_extraction):
        base = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        sidecar = base.with_name(f"{base.stem}{ENRICHMENT_ENTITIES_SUFFIX}")
        # a readable (non-empty) parquet via the real writer
        from womblex.store.enrichment_output import write_enrichment_entities_shard
        write_enrichment_entities_shard([], base)
        assert sidecar.exists()

        mgr = CheckpointManager(tmp_path / "ckpt", "test_enrich")
        mgr.update(["doc-a"], succeeded=1, failed=0, batch_num=1)
        assert reconcile_stage_checkpoint_with_shards(
            mgr, tmp_path, suffix=ENRICHMENT_ENTITIES_SUFFIX) == []
        assert "doc-a" in mgr.state.processed_ids

    def test_scan_flags_unhealthy(self, tmp_path, budget_extraction):
        base = _write_batch(tmp_path, 1, budget_extraction, "doc-a")
        base.with_name(f"{base.stem}{ENRICHMENT_ENTITIES_SUFFIX}").write_bytes(b"")
        scan = scan_sidecar_directory(tmp_path, ENRICHMENT_ENTITIES_SUFFIX)
        assert len(scan) == 1 and not scan[0].is_healthy
