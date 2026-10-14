"""The local multi-stage runner: config resolution, refusal before work, honest step outcomes."""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests._synthetic import REGISTER_CSV
from womblex.batch import process_batch
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.cloud.workflow_runner import (
    WorkflowRefusedError,
    resolve_workflow_config,
    run_workflow_local,
)
from womblex.cloud.workflow_store import (
    StoredWorkflow,
    WorkflowStore,
    combine_config,
    ensure_workflow_schema,
    validate_workflow_spec,
    workflow_digest,
)
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.process.text_overlay import MissingOverlayError
from womblex.ui.presets import Preset, build_preset_record, preset_filename, serialise_preset_record

_PRESET = Preset(
    name="saved-offline", description="", formats=(), source="saved",
    config={"chunking": {"chunk_size": 120}, "processing": {"models_check": "off"}},
)


def _resolve(name: str) -> Preset | None:
    return _PRESET if name == _PRESET.name else None


def _stored(steps: list[str], **extra) -> StoredWorkflow:
    spec = validate_workflow_spec({
        "preset": _PRESET.name, "steps": [{"stage": s} for s in steps], **extra,
    })
    return StoredWorkflow("wf", None, spec, workflow_digest(spec), "", "")


@pytest.fixture
def shards(tmp_path: Path) -> Path:
    cfg = WomblexConfig(
        dataset=DatasetConfig(name="t"),
        paths=PathsConfig(input_root=tmp_path, output_root=tmp_path, checkpoint_dir=tmp_path),
    )
    out = tmp_path / "shards"
    out.mkdir()
    process_batch([REGISTER_CSV], cfg, batch_num=1, shard_dir=out)
    return out


# --- configuration -----------------------------------------------------------


def test_combine_config_merges_keys_and_drops_identity() -> None:
    base = {"chunking": {"chunk_size": 100, "overlap": 0.1, "tags": ["a"]}, "dataset": {"name": "x"}}
    combined = combine_config(
        base, {"chunking": {"chunk_size": 50, "overlap": None, "tags": ["b"]}, "paths": {}, "money": {}},
    )
    # A named key replaces, null drops it (schema default), a list is replaced whole.
    assert combined == {"chunking": {"chunk_size": 50, "tags": ["b"]}, "money": {}}
    assert combine_config(base, {"chunking": {"chunk_size": 50}})["chunking"]["overlap"] == 0.1
    assert base["chunking"]["chunk_size"] == 100  # the preset is not mutated


def test_resolver_combines_preset_and_workflow_config() -> None:
    stored = _stored(["chunk"], config={"money": {"enabled": True}})
    config = resolve_workflow_config(stored, _resolve)
    assert config.chunking.chunk_size == 120  # from the saved preset
    assert config.dataset.name == _PRESET.name


def test_resolver_refuses_unknown_preset_and_unloadable_combination() -> None:
    with pytest.raises(WorkflowRefusedError, match="unknown preset"):
        resolve_workflow_config(_stored(["chunk"]), lambda _n: None)
    bad = _stored(["chunk"], config={})
    bad.spec.config["chunking"] = {"no_such_field": 1}  # stored row that no longer loads
    with pytest.raises(WorkflowRefusedError, match="does not load"):
        resolve_workflow_config(bad, _resolve)


# --- save validates ----------------------------------------------------------


def test_save_refuses_an_unchecked_spec(tmp_path: Path) -> None:
    from womblex.cloud.workflow_store import StoredWorkflowSpec, WorkflowStep
    dsn = f"sqlite:///{tmp_path}/wf.db"
    ensure_workflow_schema(dsn)
    unchecked = StoredWorkflowSpec(preset="p", steps=(WorkflowStep(stage="nope"),))
    with WorkflowStore(dsn) as store:
        with pytest.raises(ValueError, match="unknown stage"):
            store.save("wf", unchecked)
        bad_overlay = StoredWorkflowSpec(preset="p", config={"chunking": {"nope": 1}},
                                         steps=(WorkflowStep(stage="chunk"),))
        with pytest.raises(ValidationError):
            store.save("wf", bad_overlay)
        assert store.get("wf") is None


# --- running -----------------------------------------------------------------


def test_runs_real_stages_in_list_order(shards: Path) -> None:
    result = run_workflow_local(_stored(["normalise", "chunk"]), shards, resolve_preset=_resolve)
    assert [(s.stage, s.status) for s in result.steps] == [("normalise", "ok"), ("chunk", "ok")]
    assert result.exit_code == 0
    assert list(shards.glob("*.normalised_text.parquet")) and list(shards.glob("*.chunks.parquet"))


def test_refuses_before_any_step_runs(shards: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    monkeypatch.setitem(STAGE_CONTRACTS, "normalise", replace(
        STAGE_CONTRACTS["normalise"], run=lambda *_: ran.append("normalise")))

    def refuse(_config) -> None:
        raise RuntimeError("no resource")

    monkeypatch.setitem(STAGE_CONTRACTS, "chunk", replace(STAGE_CONTRACTS["chunk"], preflight=refuse))
    with pytest.raises(WorkflowRefusedError, match="chunk preflight failed"):
        run_workflow_local(_stored(["normalise", "chunk"]), shards, resolve_preset=_resolve)
    assert ran == []  # the first step never started: the refusal came first


def test_a_failed_step_is_recorded_and_later_steps_still_run(
    shards: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    def missing(_dir, _config, _ctx) -> None:
        raise MissingOverlayError("normalised_text missing")

    def crash(_dir, _config, _ctx) -> None:
        raise RuntimeError("boom")

    monkeypatch.setitem(STAGE_CONTRACTS, "normalise", replace(STAGE_CONTRACTS["normalise"], run=missing))
    monkeypatch.setitem(STAGE_CONTRACTS, "spellfix", replace(STAGE_CONTRACTS["spellfix"], run=crash))
    with caplog.at_level(logging.INFO):
        result = run_workflow_local(
            _stored(["normalise", "spellfix", "chunk"]), shards, resolve_preset=_resolve,
        )
    by_stage = {s.stage: s for s in result.steps}
    assert by_stage["normalise"].status == "failed"
    assert by_stage["normalise"].error == "normalised_text missing"
    assert by_stage["spellfix"].error == "RuntimeError: boom"
    assert by_stage["chunk"].status == "ok"
    assert by_stage["chunk"].ran_after_failed == ("normalise", "spellfix")
    assert result.exit_code == 1
    assert "running after failed step(s) normalise, spellfix" in caplog.text


def test_warns_about_stored_but_unacted_parts(shards: Path, caplog: pytest.LogCaptureFixture) -> None:
    stored = _stored(
        ["normalise"],
        conditions=[{"column": "ext", "op": "eq", "value": ".csv"}],
        triggers=[{"cron": "0 3 * * *"}],
        destinations=[{"kind": "webhook", "target": "https://example.invalid/hook"}],
    )
    with caplog.at_level(logging.WARNING):
        result = run_workflow_local(stored, shards, resolve_preset=_resolve)
    assert result.not_acted_on == ("destinations", "triggers")
    assert "stored but not acted on" in caplog.text
    assert result.exit_code == 0


def test_cli_exit_codes(shards: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    from womblex.cli.workflow import cmd_run_workflow

    dsn = f"sqlite:///{tmp_path}/wf.db"
    ensure_workflow_schema(dsn)
    with WorkflowStore(dsn) as store:
        store.save("good", _stored(["normalise"]).spec)
    def args(wid: str, **kw) -> argparse.Namespace:
        base = {"workflow_id": wid, "shards": shards, "dsn": dsn, "presets_dir": None,
                "ingest": None, "force": False, "models_check": None}
        return argparse.Namespace(**{**base, **kw})

    assert cmd_run_workflow(args("missing")) == 1
    # "saved-offline" is not a built-in and no presets dir holds it: refused, exit 1.
    assert cmd_run_workflow(args("good")) == 1
    record, _ = build_preset_record(name=_PRESET.name, description="", formats=(), config=_PRESET.config)
    (tmp_path / preset_filename(_PRESET.name)).write_text(serialise_preset_record(record))
    for mod in ("fastapi", "womblex.ui.deps", "womblex.ui.readers"):  # base install: no [ui] extra
        monkeypatch.setitem(sys.modules, mod, None)
    assert cmd_run_workflow(args("good", presets_dir=tmp_path)) == 0
    assert list(shards.glob("*.normalised_text.parquet"))
