"""Tests for store/retention.py — run_id generation and rolling purge."""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from womblex.store.output import ELEMENTS_SUFFIX, MANIFEST_SCHEMA
from womblex.store.retention import (
    RunDescription,
    apply_retention,
    describe_run,
    generate_run_id,
    is_safe_run_id,
    list_runs,
    most_recent_run,
)


def _make_run(output_root: Path, name: str) -> Path:
    run_dir = output_root / name / "documents"
    run_dir.mkdir(parents=True)
    (run_dir / "batch-0001.elements.parquet").write_bytes(b"stub")
    return output_root / name


def _manifest_row(doc_id: str) -> dict:
    return {
        "source_hash": f"hash-{doc_id}", "collection_id": "test", "doc_id": doc_id,
        "filename": f"{doc_id}.pdf", "ext": ".pdf", "extraction_method": "native",
        "elements_count": 3, "table_cells_count": 0, "form_fields_count": 0,
        "status": "ok", "error": "", "extracted_at_iso": "2026-08-16T00:00:00Z",
        "parser_version": "2.0",
    }


def _write_manifest_shard(shard_dir: Path, doc_ids: list[str]) -> None:
    """A real, schema-correct manifest shard — enough for describe_run's count."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([_manifest_row(d) for d in doc_ids], schema=MANIFEST_SCHEMA)
    pq.write_table(table, str(shard_dir / "batch-0001._manifest.parquet"))
    (shard_dir / f"batch-0001{ELEMENTS_SUFFIX}").write_bytes(b"stub")


def _make_checkpoint(checkpoint_dir: Path, name: str) -> Path:
    ckpt = checkpoint_dir / name
    ckpt.mkdir(parents=True)
    (ckpt / "ds_checkpoint.json").write_text("{}")
    return ckpt


class TestGenerateRunId:
    def test_format_is_basic_iso_timestamp(self) -> None:
        rid = generate_run_id()
        assert re.fullmatch(r"run-\d{8}T\d{6}Z", rid)

    def test_two_calls_sort_in_order(self) -> None:
        import time
        a = generate_run_id()
        time.sleep(1.0)
        b = generate_run_id()
        assert a < b


class TestListRuns:
    def test_empty_returns_empty(self, tmp_path: Path) -> None:
        assert list_runs(tmp_path) == []

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert list_runs(tmp_path / "nope") == []

    def test_ignores_files(self, tmp_path: Path) -> None:
        _make_run(tmp_path, "run-a")
        (tmp_path / "stray.txt").write_text("x")
        runs = list_runs(tmp_path)
        assert [r.name for r in runs] == ["run-a"]

    def test_sorts_oldest_first(self, tmp_path: Path) -> None:
        _make_run(tmp_path, "run-20260101T000000Z")
        _make_run(tmp_path, "run-20260301T000000Z")
        _make_run(tmp_path, "run-20260201T000000Z")
        runs = list_runs(tmp_path)
        names = [r.name for r in runs]
        assert names == [
            "run-20260101T000000Z",
            "run-20260201T000000Z",
            "run-20260301T000000Z",
        ]


class TestMostRecentRun:
    def test_returns_none_when_empty(self, tmp_path: Path) -> None:
        assert most_recent_run(tmp_path) is None

    def test_returns_newest(self, tmp_path: Path) -> None:
        _make_run(tmp_path, "run-20260101T000000Z")
        _make_run(tmp_path, "run-20260301T000000Z")
        _make_run(tmp_path, "run-20260201T000000Z")
        latest = most_recent_run(tmp_path)
        assert latest is not None
        assert latest.name == "run-20260301T000000Z"


class TestApplyRetentionRolling:
    def test_keeps_window_purges_older(self, tmp_path: Path) -> None:
        output_root = tmp_path / "out"
        ckpt_root = tmp_path / "ckpt"
        # 4 prior runs + 1 current
        for name in [
            "run-20260101T000000Z",
            "run-20260102T000000Z",
            "run-20260103T000000Z",
            "run-20260104T000000Z",
        ]:
            _make_run(output_root, name)
            _make_checkpoint(ckpt_root, name)
        current = "run-20260105T000000Z"
        _make_run(output_root, current)
        _make_checkpoint(ckpt_root, current)

        purged = apply_retention(
            output_root, ckpt_root,
            current_run_id=current, policy="rolling", keep=2,
        )

        purged_names = sorted(p.name for p in purged)
        # keep=2 means current + 1 previous; older 3 purged
        assert purged_names == [
            "run-20260101T000000Z",
            "run-20260102T000000Z",
            "run-20260103T000000Z",
        ]
        # current still on disk
        assert (output_root / current).is_dir()
        # 4th-newest preserved
        assert (output_root / "run-20260104T000000Z").is_dir()
        # checkpoints purged in lockstep
        assert not (ckpt_root / "run-20260101T000000Z").exists()
        assert (ckpt_root / "run-20260104T000000Z").is_dir()
        assert (ckpt_root / current).is_dir()

    def test_keep_1_purges_all_except_current(self, tmp_path: Path) -> None:
        output_root = tmp_path / "out"
        ckpt_root = tmp_path / "ckpt"
        _make_run(output_root, "run-old")
        current = "run-new"
        _make_run(output_root, current)

        purged = apply_retention(
            output_root, ckpt_root,
            current_run_id=current, policy="rolling", keep=1,
        )
        assert [p.name for p in purged] == ["run-old"]
        assert (output_root / current).is_dir()

    def test_no_purge_when_under_window(self, tmp_path: Path) -> None:
        output_root = tmp_path / "out"
        _make_run(output_root, "run-a")
        current = "run-b"
        _make_run(output_root, current)
        purged = apply_retention(
            output_root, tmp_path / "ckpt",
            current_run_id=current, policy="rolling", keep=3,
        )
        assert purged == []
        assert (output_root / "run-a").is_dir()
        assert (output_root / current).is_dir()

    def test_preserves_current_even_if_oldest(self, tmp_path: Path) -> None:
        """Resume scenario: the current run id is older than others."""
        output_root = tmp_path / "out"
        current = "run-20260101T000000Z"
        _make_run(output_root, current)
        _make_run(output_root, "run-20260201T000000Z")
        _make_run(output_root, "run-20260301T000000Z")
        # keep=1 — would normally purge everything but current
        purged = apply_retention(
            output_root, tmp_path / "ckpt",
            current_run_id=current, policy="rolling", keep=1,
        )
        purged_names = sorted(p.name for p in purged)
        assert purged_names == [
            "run-20260201T000000Z",
            "run-20260301T000000Z",
        ]
        assert (output_root / current).is_dir()

    def test_missing_checkpoint_dir_is_ok(self, tmp_path: Path) -> None:
        output_root = tmp_path / "out"
        _make_run(output_root, "run-old")
        current = "run-new"
        _make_run(output_root, current)
        # checkpoint dir doesn't exist — should not raise
        purged = apply_retention(
            output_root, tmp_path / "missing-ckpt",
            current_run_id=current, policy="rolling", keep=1,
        )
        assert [p.name for p in purged] == ["run-old"]


class TestApplyRetentionKeepAll:
    def test_no_purge_under_keep_all(self, tmp_path: Path) -> None:
        output_root = tmp_path / "out"
        for name in ["run-a", "run-b", "run-c", "run-d", "run-e"]:
            _make_run(output_root, name)
        purged = apply_retention(
            output_root, tmp_path / "ckpt",
            current_run_id="run-e", policy="keep_all", keep=2,
        )
        assert purged == []
        assert len(list_runs(output_root)) == 5

    def test_keep_value_ignored_under_keep_all(self, tmp_path: Path) -> None:
        output_root = tmp_path / "out"
        _make_run(output_root, "run-a")
        # keep=99 still no-op under keep_all
        purged = apply_retention(
            output_root, tmp_path / "ckpt",
            current_run_id="run-a", policy="keep_all", keep=99,
        )
        assert purged == []


class TestListRunsRunPrefixFilter:
    """Only `run-*`-named dirs are subject to retention; legacy / hand-named
    dirs (e.g. `documents/`, `output-pre-kcluster-…/`) are preserved."""

    def test_ignores_non_run_dirs(self, tmp_path: Path) -> None:
        _make_run(tmp_path, "run-current")
        # legacy + hand-named dirs that retention must NOT touch
        (tmp_path / "documents").mkdir()
        (tmp_path / "output-pre-kcluster-2026-05-17").mkdir()
        (tmp_path / "baseline-snapshot").mkdir()
        runs = list_runs(tmp_path)
        names = [r.name for r in runs]
        assert names == ["run-current"]

    def test_apply_retention_does_not_purge_legacy_dirs(self, tmp_path: Path) -> None:
        output_root = tmp_path / "out"
        _make_run(output_root, "run-current")
        # legacy artefacts the user wants preserved
        (output_root / "documents").mkdir()
        (output_root / "documents" / "batch-0001.elements.parquet").write_bytes(b"legacy")
        (output_root / "output-pre-kcluster-2026-05-17").mkdir()
        (output_root / "output-pre-kcluster-2026-05-17" / "stub.parquet").write_bytes(b"older")

        # keep=1 would normally purge everything except current
        purged = apply_retention(
            output_root, tmp_path / "ckpt",
            current_run_id="run-current", policy="rolling", keep=1,
        )
        assert purged == []
        # Legacy still on disk
        assert (output_root / "documents").is_dir()
        assert (output_root / "documents" / "batch-0001.elements.parquet").exists()
        assert (output_root / "output-pre-kcluster-2026-05-17").is_dir()


class TestApplyRetentionValidation:
    def test_unknown_policy_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown retention policy"):
            apply_retention(
                tmp_path / "out", tmp_path / "ckpt",
                current_run_id="run-x", policy="weird", keep=2,
            )

    def test_keep_below_one_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="retention.keep must be >= 1"):
            apply_retention(
                tmp_path / "out", tmp_path / "ckpt",
                current_run_id="run-x", policy="rolling", keep=0,
            )

    def test_idempotent_no_runs(self, tmp_path: Path) -> None:
        purged = apply_retention(
            tmp_path / "out", tmp_path / "ckpt",
            current_run_id="run-x", policy="rolling", keep=2,
        )
        assert purged == []


class TestDescribeRun:
    def test_no_documents_dir_yet(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-empty"
        run_dir.mkdir()
        assert describe_run(run_dir) == RunDescription(
            run_id="run-empty", document_count=None, stages=(),
            created_at=None, updated_at=None,
        )

    def test_shard_manifest_only(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-a"
        _write_manifest_shard(run_dir / "documents", ["doc-a", "doc-b"])
        desc = describe_run(run_dir)
        assert desc.run_id == "run-a"
        assert desc.document_count == 2
        assert desc.stages == ("extract",)
        assert desc.created_at is not None
        assert desc.updated_at is not None
        assert desc.created_at <= desc.updated_at

    def test_prefers_consolidated_manifest_over_shards(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-a"
        _write_manifest_shard(run_dir / "documents", ["doc-a"])
        # Consolidated manifest disagrees with the shard — describe_run must
        # read the consolidated one, the way `womblex manifest` publishes it.
        table = pa.Table.from_pylist([_manifest_row("doc-a"), _manifest_row("doc-b")], schema=MANIFEST_SCHEMA)
        pq.write_table(table, str(run_dir / "manifest.parquet"))
        assert describe_run(run_dir).document_count == 2

    def test_stages_reflect_sidecars_present(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "run-a" / "documents"
        _write_manifest_shard(shard_dir, ["doc-a"])
        (shard_dir / "batch-0001.chunks.parquet").write_bytes(b"stub")
        (shard_dir / "batch-0001.money_spans.parquet").write_bytes(b"stub")
        assert describe_run(tmp_path / "run-a").stages == ("extract", "chunk", "money")


def test_is_safe_run_id_admits_only_a_single_path_segment() -> None:
    """The root/run_id join is where containment is enforced."""
    assert is_safe_run_id("run-20260101T000000Z")
    for run_id in ["../escape", "a/b", "a\\b", "..", ".", "", "/abs", "a\x00b"]:
        assert not is_safe_run_id(run_id)
