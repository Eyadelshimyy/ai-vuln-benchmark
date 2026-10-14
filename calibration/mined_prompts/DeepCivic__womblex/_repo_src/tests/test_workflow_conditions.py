"""Conditions: guards over manifest columns, decided before any step runs."""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pytest
from pydantic import ValidationError

from tests.test_workflow_runner import _PRESET, _resolve, _stored, shards  # noqa: F401
from womblex.cloud.workflow_conditions import ConditionRefused, require_conditions
from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError, run_workflow_local
from womblex.cloud.workflow_store import StoredWorkflow, WorkflowCondition, validate_workflow_spec
from womblex.cloud.workflow_validate import validate_workflow
from womblex.store.output import read_manifest


def _cond(column: str, op: str, value: object) -> dict:
    return {"column": column, "op": op, "value": value}


def _status(shard_dir: Path) -> str:
    return read_manifest(shard_dir).column("status").to_pylist()[0]


@pytest.mark.parametrize("condition", [
    _cond("filename_lower", "eq", "x"),          # not a manifest column
    _cond("status", "gt", "a"),                  # order operator on text
    _cond("elements_count", "eq", "many"),       # text against a number column
    _cond("status", "in", "ok"),                 # in needs a list
    _cond("status", "in", []),
    {**_cond("status", "eq", "ok"), "stage": "pii"},  # a condition cannot name a step
])
def test_a_condition_that_could_never_be_evaluated_is_not_stored(condition: dict) -> None:
    with pytest.raises((ValueError, ValidationError)):
        validate_workflow_spec({"preset": "p", "steps": [{"stage": "normalise"}], "conditions": [condition]})


def test_null_never_satisfies_a_condition_not_even_ne() -> None:
    table = pa.table({"doc_id": ["a", "b"], "status": ["ok", None]})
    for op in ("eq", "ne"):
        with pytest.raises(ConditionRefused) as e:
            require_conditions([WorkflowCondition(column="status", op=op, value="x" if op == "ne" else "ok")], table)
        assert e.value.code == "condition_failed" and "1 of 2" in str(e.value)


def test_a_holding_condition_runs_and_a_failing_one_stops_before_any_step(shards: Path) -> None:  # noqa: F811
    ok = _status(shards)
    held = _stored(["normalise"], conditions=[_cond("status", "eq", ok), _cond("elements_count", "gt", 0)])
    assert run_workflow_local(held, shards, resolve_preset=_resolve).exit_code == 0
    for f in shards.glob("*.normalised_text.parquet"):
        f.unlink()

    failing = _stored(["normalise"], conditions=[_cond("status", "ne", ok)])
    with pytest.raises(WorkflowRefusedError) as e:
        run_workflow_local(failing, shards, resolve_preset=_resolve)
    assert e.value.code is RefusalCode.CONDITION_FAILED and "status" in str(e.value)
    assert not list(shards.glob("*.normalised_text.parquet"))  # no step ran


def test_no_manifest_rows_cannot_be_evaluated_and_validate_says_when_it_did_not_try(
    shards: Path,  # noqa: F811
) -> None:
    empty = read_manifest(shards).slice(0, 0)
    with pytest.raises(ConditionRefused) as e:
        require_conditions([WorkflowCondition(column="status", op="eq", value="ok")], empty)
    assert e.value.code == "condition_unevaluable"
    stored = _stored(["normalise"], conditions=[_cond("status", "ne", _status(shards))])
    assert validate_workflow(stored, resolve_preset=_resolve).valid  # no shard dir: not evaluated
    report = validate_workflow(stored, resolve_preset=_resolve, shard_dir=shards)
    assert report.code == RefusalCode.CONDITION_FAILED.value


def test_dispatch_evaluates_conditions_against_the_store_before_any_row(
    shards: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    pytest.importorskip("dbos")
    from womblex.cloud.workflow_dispatch import dispatch_workflow

    monkeypatch.setenv("WOMBLEX_DBOS_PATH", str(tmp_path / "dbos.sqlite"))
    store = tmp_path / "store"
    shutil.copytree(shards, store / "runs/r1/documents")

    def go(op: str) -> object:
        stored = _stored(["normalise"], conditions=[_cond("status", op, _status(shards))])
        return dispatch_workflow(stored, resolve_preset=_resolve, run_id="r1",
                                 store_uri=str(store), dsn=None, dry_run=True)

    assert go("eq").stages == ("normalise",)  # type: ignore[attr-defined]
    with pytest.raises(WorkflowRefusedError) as e:
        go("ne")
    assert e.value.code is RefusalCode.CONDITION_FAILED
    with pytest.raises(WorkflowRefusedError) as e:  # an empty store: nothing to evaluate
        dispatch_workflow(_stored(["normalise"], conditions=[_cond("status", "eq", "ok")]),
                          resolve_preset=_resolve, run_id="r2", store_uri=str(tmp_path / "none"),
                          dsn=None, dry_run=True)
    assert e.value.code is RefusalCode.CONDITION_UNEVALUABLE


@pytest.fixture
def dispatchable(shards: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:  # noqa: F811
    """The shard dir mirrored into a store as run r1, and an existing local DBOS database."""
    pytest.importorskip("dbos")
    from womblex.cloud import dbos_app

    monkeypatch.setenv("WOMBLEX_DBOS_PATH", str(tmp_path / "dbos.sqlite"))
    dbos_app.ensure_schema(None)
    store = tmp_path / "store"
    shutil.copytree(shards, store / "runs/r1/documents")
    return store, tmp_path / "dbos.sqlite"


def _every_entry_point(
    stored: StoredWorkflow, shards: Path, store: Path, database: Path,  # noqa: F811
) -> dict[str, str]:
    """The code each of run-workflow, validate, plan and a real dispatch refuses *stored* with."""
    from womblex.cloud.jobs import RunBoard
    from womblex.cloud.workflow_dispatch import dispatch_workflow
    from womblex.cloud.workflow_plan import plan_workflow

    on_disk, before = sorted(p.name for p in shards.iterdir()), database.read_bytes()
    codes = {"validate": validate_workflow(stored, resolve_preset=_resolve, shard_dir=shards).code}
    for name, call in (
        ("run", lambda: run_workflow_local(stored, shards, resolve_preset=_resolve)),
        ("plan", lambda: plan_workflow(stored, shards, resolve_preset=_resolve)),
        ("dispatch", lambda: dispatch_workflow(stored, resolve_preset=_resolve, run_id="r1",
                                               store_uri=str(store), dsn=None, dry_run=False)),
    ):
        with pytest.raises(WorkflowRefusedError) as e:
            call()
        codes[name] = e.value.code.value
    assert sorted(p.name for p in shards.iterdir()) == on_disk  # no step wrote anything
    assert database.read_bytes() == before  # no queue row, not even a schema touch
    with RunBoard(None) as board:
        assert board.list_jobs("r1") == []
    return codes


def test_every_entry_point_refuses_a_failing_condition_alike(
    shards: Path, dispatchable: tuple[Path, Path],  # noqa: F811
) -> None:
    stored = _stored(["normalise"], conditions=[_cond("status", "ne", _status(shards))])
    assert set(_every_entry_point(stored, shards, *dispatchable).values()) == {RefusalCode.CONDITION_FAILED.value}


@pytest.mark.parametrize("manifest", ["absent", "empty"])
def test_an_empty_manifest_is_unevaluable_and_an_absent_one_is_refused_first(
    shards: Path, dispatchable: tuple[Path, Path], manifest: str,  # noqa: F811
) -> None:
    import pyarrow.parquet as pq

    store, database = dispatchable
    for d in (shards, store / "runs/r1/documents"):
        for f in d.glob("*._manifest.parquet"):
            empty = read_manifest(f.parent / f.name.removesuffix("._manifest.parquet")).slice(0, 0)
            f.unlink()
            if manifest == "empty":
                pq.write_table(empty, f)
    stored = _stored(["normalise"], conditions=[_cond("elements_count", "gte", 0)])
    codes = _every_entry_point(stored, shards, store, database)
    # An absent manifest is also a missing step input locally, refused first by the order check;
    # dispatch expects extraction to supply it, so it reaches the conditions.
    local = RefusalCode.STEP_ORDER if manifest == "absent" else RefusalCode.CONDITION_UNEVALUABLE
    assert codes == {"validate": local.value, "run": local.value, "plan": local.value,
                     "dispatch": RefusalCode.CONDITION_UNEVALUABLE.value}


def test_null_fails_every_operator() -> None:
    table = pa.table({"doc_id": ["a"], "status": pa.array([None], pa.string()),
                      "elements_count": pa.array([None], pa.int64())})
    for c in (WorkflowCondition(column="status", op="not_in", value=["x"]),
              WorkflowCondition(column="status", op="in", value=["x"]),
              WorkflowCondition(column="elements_count", op="lt", value=10**9)):
        with pytest.raises(ConditionRefused) as e:
            require_conditions([c], table)
        assert e.value.code == "condition_failed"


def test_a_row_stored_before_conditions_were_checked_is_refused_on_load(
    shards: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],  # noqa: F811
) -> None:
    import argparse
    import json

    from womblex.cli.workflow import cmd_validate
    from womblex.cloud.workflow_store import StoredWorkflowSpec, WorkflowStore, ensure_workflow_schema

    dsn = f"sqlite:///{tmp_path}/wf.db"
    ensure_workflow_schema(dsn)
    old = StoredWorkflowSpec.model_validate({  # pydantic accepts it; check_spec would not
        "preset": "p", "steps": [{"stage": "normalise"}],
        "conditions": [_cond("manifest.doc_type", "eq", "pdf")],
    })
    with WorkflowStore(dsn) as store:
        store._write("old", old, None)  # as merge 1 stored it, past today's save-time check
        stored = store.get("old")
    assert stored is not None
    with pytest.raises(WorkflowRefusedError) as e:
        run_workflow_local(stored, shards, resolve_preset=_resolve)
    assert e.value.code is RefusalCode.SPEC_INVALID and "manifest.doc_type" in str(e.value)
    args = argparse.Namespace(workflow_id="old", shards=shards, dsn=dsn, presets_dir=None,
                              models_check=None, json=True)
    assert cmd_validate(args) == 1
    body = json.loads(capsys.readouterr().out)  # one JSON object
    assert body["code"] == RefusalCode.SPEC_INVALID.value


def test_validate_and_plan_json_print_one_object_for_a_failing_condition(
    shards: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],  # noqa: F811
) -> None:
    import argparse
    import json

    from womblex.cli.workflow import cmd_plan
    from womblex.cloud.workflow_store import WorkflowStore, ensure_workflow_schema
    from womblex.ui.presets import build_preset_record, preset_filename, serialise_preset_record

    dsn = f"sqlite:///{tmp_path}/wf.db"
    ensure_workflow_schema(dsn)
    with WorkflowStore(dsn) as store:
        store.save("c", _stored(["normalise"], conditions=[_cond("status", "ne", _status(shards))]).spec)
    record, _ = build_preset_record(name=_PRESET.name, description="", formats=(), config=_PRESET.config)
    (tmp_path / preset_filename(_PRESET.name)).write_text(serialise_preset_record(record))
    args = argparse.Namespace(workflow_id="c", shards=shards, dsn=dsn, presets_dir=tmp_path,
                              models_check=None, json=True, assume_price=[])
    from womblex.cli.workflow import cmd_validate

    for cmd in (cmd_validate, cmd_plan):
        capsys.readouterr()
        assert cmd(args) == 1
        assert json.loads(capsys.readouterr().out)["code"] == RefusalCode.CONDITION_FAILED.value
