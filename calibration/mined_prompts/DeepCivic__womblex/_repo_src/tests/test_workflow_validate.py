"""`womblex validate`: the runner's pre-run checks, read-only, with the same stable codes."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_workflow_runner import _PRESET, _resolve, _stored, shards  # noqa: F401
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.cloud.workflow_order import OrderCode
from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError, run_workflow_local
from womblex.cloud.workflow_store import WorkflowStore, ensure_workflow_schema
from womblex.cloud.workflow_validate import validate_workflow

_NORMALISED = {"processing": {"text_source": "normalised"}}


def _snapshot(d: Path) -> dict[str, int]:
    return {p.name: p.stat().st_mtime_ns for p in d.iterdir()}


def test_valid_workflow_is_reported_and_nothing_is_written(shards: Path) -> None:  # noqa: F811
    before = _snapshot(shards)
    report = validate_workflow(_stored(["normalise", "chunk"]), resolve_preset=_resolve, shard_dir=shards)
    assert report.valid and report.code is None
    assert report.stages == ("normalise", "chunk") and report.shards_checked
    assert _snapshot(shards) == before


def test_refuses_what_the_runner_refuses_with_the_same_codes(shards: Path) -> None:  # noqa: F811
    stored = _stored(["chunk", "normalise"], config=_NORMALISED)
    report = validate_workflow(stored, resolve_preset=_resolve, shard_dir=shards)
    with pytest.raises(WorkflowRefusedError) as e:
        run_workflow_local(stored, shards, resolve_preset=_resolve)
    assert not report.valid
    assert report.code == e.value.code.value == RefusalCode.STEP_ORDER.value
    assert [i["code"] for i in report.issues] == [i.code.value for i in e.value.issues]
    assert report.issues and report.issues[0]["code"] in {c.value for c in OrderCode}


def test_unknown_preset_and_every_failed_step_are_reported(shards: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    stored = _stored(["normalise", "chunk"])
    assert validate_workflow(stored, resolve_preset=lambda _n: None).code == RefusalCode.PRESET_UNKNOWN.value

    def refuse(_config) -> None:
        raise RuntimeError("no resource")

    for stage in ("normalise", "chunk"):
        monkeypatch.setitem(STAGE_CONTRACTS, stage, replace(STAGE_CONTRACTS[stage], preflight=refuse))
    report = validate_workflow(stored, resolve_preset=_resolve, shard_dir=shards)
    assert report.code == RefusalCode.STEP_PRECONDITION.value
    assert [s["stage"] for s in report.steps] == ["normalise", "chunk"]


def test_without_shards_only_the_step_order_is_checked() -> None:
    report = validate_workflow(_stored(["normalise", "chunk"]), resolve_preset=_resolve)
    assert report.valid and not report.shards_checked
    assert not validate_workflow(
        _stored(["chunk", "normalise"], config=_NORMALISED), resolve_preset=_resolve,
    ).valid


def test_cli_json_and_exit_codes(shards: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    from womblex.cli.workflow import cmd_validate

    dsn = f"sqlite:///{tmp_path}/wf.db"
    ensure_workflow_schema(dsn)
    with WorkflowStore(dsn) as store:
        store.save("good", _stored(["normalise"]).spec)
        store.save("misordered", _stored(["chunk", "normalise"], config=_NORMALISED).spec)
    from womblex.ui.presets import build_preset_record, preset_filename, serialise_preset_record

    record, _ = build_preset_record(name=_PRESET.name, description="", formats=(), config=_PRESET.config)
    (tmp_path / preset_filename(_PRESET.name)).write_text(serialise_preset_record(record))

    def args(wid: str, **kw) -> argparse.Namespace:
        base = {"workflow_id": wid, "shards": shards, "dsn": dsn, "presets_dir": tmp_path,
                "models_check": None, "json": True}
        return argparse.Namespace(**{**base, **kw})

    assert cmd_validate(args("missing")) == 1
    assert json.loads(capsys.readouterr().out)["code"] == RefusalCode.WORKFLOW_UNAVAILABLE.value
    assert cmd_validate(args("good")) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["valid"] is True and body["stages"] == ["normalise"]
    assert cmd_validate(args("good", json=False, presets_dir=None)) == 1
    assert "refused [preset_unknown]" in capsys.readouterr().out
    assert cmd_validate(args("misordered", json=False)) == 1
    out = capsys.readouterr().out
    assert "refused [step_order]\n" in out and out.count("move 'normalise' before 'chunk'") == 1
