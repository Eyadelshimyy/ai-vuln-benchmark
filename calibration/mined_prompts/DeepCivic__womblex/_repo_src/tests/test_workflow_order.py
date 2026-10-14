"""The up-front ordering check: shared by the local runner, a store dispatch and the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._synthetic import REGISTER_CSV
from tests.test_workflow_runner import _resolve, _stored
from womblex.batch import process_batch
from womblex.cloud.workflow_order import (
    OrderCode,
    StageOrderError,
    check_dispatch_order,
    check_step_order,
    optional_order_warnings,
    present_from_names,
)
from womblex.cloud.workflow_runner import WorkflowRefusedError, run_workflow_local
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.store.remote import RemoteStore

_EXTRACTION = ["b1.elements.parquet", "b1.table_cells.parquet", "b1._manifest.parquet"]


def _config(tmp_path: Path, **sections) -> WomblexConfig:
    return WomblexConfig(
        dataset=DatasetConfig(name="t"),
        paths=PathsConfig(input_root=tmp_path, output_root=tmp_path, checkpoint_dir=tmp_path),
        **sections,
    )


def _codes(issues) -> list[tuple[str, str, str]]:
    return [(i.stage, i.code.value, i.suffix) for i in issues]


def test_overlay_produced_earlier_is_clean(tmp_path: Path) -> None:
    cfg = _config(tmp_path, processing={"text_source": "normalised"})
    present = present_from_names(_EXTRACTION)
    assert check_step_order(["normalise", "chunk"], cfg, present=present) == []


def test_overlay_produced_later_is_out_of_order(tmp_path: Path) -> None:
    cfg = _config(tmp_path, processing={"text_source": "normalised"})
    issues = check_step_order(["chunk", "normalise"], cfg, present=present_from_names(_EXTRACTION))
    assert _codes(issues) == [("chunk", "input_out_of_order", ".normalised_text.parquet")]
    assert issues[0].producer == "normalise"


def test_existing_copy_is_stale_when_a_later_step_rewrites_it(tmp_path: Path) -> None:
    cfg = _config(tmp_path, processing={"text_source": "normalised"})
    present = present_from_names([*_EXTRACTION, "b1.normalised_text.parquet"])
    issues = check_step_order(["chunk", "normalise"], cfg, present=present)
    assert [i.code for i in issues] == [OrderCode.INPUT_STALE]
    # The same copy is fine when nothing in the workflow rewrites it.
    assert check_step_order(["chunk"], cfg, present=present) == []


def test_missing_input_names_its_producer(tmp_path: Path) -> None:
    cfg = _config(tmp_path, processing={"text_source": "normalised"})
    issues = check_step_order(["chunk"], cfg, present=present_from_names(_EXTRACTION))
    assert [(i.code, i.producer) for i in issues] == [(OrderCode.INPUT_MISSING, "normalise")]
    assert "add 'normalise' before 'chunk'" in issues[0].message


def test_in_place_steps_neither_satisfy_nor_invalidate(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    # graph-refresh rewrites entities it must already have; it cannot create them.
    present = present_from_names([*_EXTRACTION, "b1.chunks.parquet"])
    assert {i.suffix for i in check_step_order(["graph-refresh"], cfg, present=present)} == {
        ".enrichment_entities.parquet", ".graph_edges.parquet"}
    # ...and a link before it is not made stale by its in-place rewrite.
    held = present_from_names([*_EXTRACTION, "b1.chunks.parquet", "b1.enrichment_entities.parquet",
                               "b1.graph_edges.parquet"])
    assert check_step_order(["link", "graph-refresh"], cfg, present=held) == []


def test_optional_inputs_are_not_ordered(tmp_path: Path) -> None:
    # enrich reads chunks only when present; the canonical order puts chunk after it.
    assert check_step_order(["enrich", "chunk"], _config(tmp_path),
                            present=present_from_names(_EXTRACTION)) == []


def test_unknown_disk_and_pending_extraction(tmp_path: Path) -> None:
    cfg = _config(tmp_path, processing={"text_source": "normalised"})
    assert check_step_order(["normalise", "chunk"], cfg, present=None) == []
    assert _codes(check_step_order(["chunk", "normalise"], cfg, present=None))[0][1] == "input_out_of_order"
    # Nothing landed yet: extraction's siblings are on their way, a downstream layer is not.
    assert check_step_order(["normalise"], cfg, present={}, extraction_pending=True) == []
    assert [i.code for i in check_step_order(["embed"], cfg, present={}, extraction_pending=True)] == [
        OrderCode.INPUT_MISSING]
    assert check_step_order(["normalise"], cfg, present={})  # local: nothing is coming


def test_store_dispatch_refuses_and_codes_are_data(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    (store_dir / "runs/r1/documents").mkdir(parents=True)
    for name in [*_EXTRACTION, "b1.normalised_text.parquet"]:
        (store_dir / "runs/r1/documents" / name).write_bytes(b"")
    cfg = _config(tmp_path, processing={"text_source": "normalised"})
    store = RemoteStore.from_uri(str(store_dir))
    check_dispatch_order(["normalise", "chunk"], cfg, store, "runs/r1/documents")
    with pytest.raises(StageOrderError) as e:
        check_dispatch_order(["chunk", "normalise"], cfg, store, "runs/r1/documents")
    assert e.value.issues[0].as_dict()["code"] == "input_stale"
    assert isinstance(e.value, ValueError)  # the console and API map it to 400


def test_rerunning_a_misordered_workflow_over_a_stale_overlay_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "shards"
    out.mkdir()
    process_batch([REGISTER_CSV], _config(tmp_path), batch_num=1, shard_dir=out)
    config = {"processing": {"text_source": "normalised"}}
    first = run_workflow_local(_stored(["normalise", "chunk"], config=config), out, resolve_preset=_resolve)
    assert first.exit_code == 0
    chunks = sorted(out.glob("*.chunks.parquet"))
    before = [p.stat().st_mtime_ns for p in chunks]

    with pytest.raises(WorkflowRefusedError) as e:
        run_workflow_local(_stored(["chunk", "normalise"], config=config), out, resolve_preset=_resolve)
    assert [i.code for i in e.value.issues] == [OrderCode.INPUT_STALE]
    assert [p.stat().st_mtime_ns for p in chunks] == before  # nothing re-ran


def test_optional_input_produced_later_against_pipeline_order_warns(tmp_path: Path) -> None:
    cfg = _config(tmp_path, processing={"text_source": "spellfix"})
    present = present_from_names(_EXTRACTION)
    steps = ["spellfix", "normalise", "chunk"]
    assert check_step_order(steps, cfg, present=present) == []  # not a refusal...
    [warning] = optional_order_warnings(steps, cfg)  # ...but said
    assert warning.startswith("spellfix: .normalised_text.parquet")
    assert optional_order_warnings(["enrich", "chunk"], cfg) == []  # canonical order


class _Unlistable:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def list_files(self, *_a, **_k):
        raise self.error


def test_unlistable_store_is_warned_and_other_errors_propagate(tmp_path: Path, caplog) -> None:
    cfg = _config(tmp_path)
    check_dispatch_order(["chunk"], cfg, _Unlistable(PermissionError("AccessDenied")), "runs/r/documents")
    assert "stored layers not checked" in caplog.text and "AccessDenied" in caplog.text
    with pytest.raises(TypeError):
        check_dispatch_order(["chunk"], cfg, _Unlistable(TypeError("bug")), "runs/r/documents")


def test_enqueue_stages_checks_the_store_before_any_row(tmp_path: Path, monkeypatch, caplog) -> None:
    import argparse

    from womblex.cli import cloud

    (tmp_path / "store/runs/r1/documents").mkdir(parents=True)
    for name in _EXTRACTION:
        (tmp_path / "store/runs/r1/documents" / name).write_bytes(b"")
    config = tmp_path / "c.yaml"
    config.write_text(f"dataset: {{name: t}}\npaths: {{input_root: {tmp_path}, output_root: {tmp_path}, checkpoint_dir: {tmp_path}}}\n"
                      "processing: {text_source: normalised}\n")
    monkeypatch.setattr("womblex.cloud.dbos_app.ensure_schema", lambda _d: pytest.fail("row written"))
    parser = argparse.ArgumentParser()
    cloud._register_enqueue_stages(parser)
    args = parser.parse_args(["--config", str(config), "--run-id", "r1", "--stages", "chunk",
                              "--store", str(tmp_path / "store"), "--dsn", "sqlite:///x.db"])
    assert cloud.cmd_enqueue_stages(args) == 1
    assert "[input_missing] chunk: .normalised_text.parquet" in caplog.text
