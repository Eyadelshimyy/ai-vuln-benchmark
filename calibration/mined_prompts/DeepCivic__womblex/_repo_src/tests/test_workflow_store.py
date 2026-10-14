"""Tests for the stored workflow object: validation, digest, and the table."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from womblex.cloud.workflow_store import (
    WorkflowOwnedError,
    WorkflowStore,
    ensure_workflow_schema,
    is_safe_workflow_id,
    validate_workflow_spec,
    workflow_digest,
)

_MINIMAL = {
    "preset": "DEFAULT-Isaacus",
    "steps": [{"stage": "enrich"}, {"stage": "chunk"}],
}


@pytest.fixture
def store(tmp_path):
    dsn = f"sqlite:///{tmp_path}/workflows.db"
    ensure_workflow_schema(dsn)
    ensure_workflow_schema(dsn)  # idempotent: a second call must not raise
    with WorkflowStore(dsn) as s:
        yield s


# --- validation --------------------------------------------------------------


def test_validate_minimal_spec() -> None:
    spec = validate_workflow_spec(_MINIMAL)
    assert [s.stage for s in spec.steps] == ["enrich", "chunk"]
    assert spec.preset == "DEFAULT-Isaacus"
    assert spec.conditions == spec.destinations == spec.triggers == ()


@pytest.mark.parametrize(("steps", "match"), [
    ([{"stage": "not-a-stage"}], "unknown stage"),
    ([], "at least one step"),
    ([{"stage": "chunk"}, {"stage": "chunk"}], "more than once"),
])
def test_validate_rejects_bad_steps(steps: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_workflow_spec({**_MINIMAL, "steps": steps})


@pytest.mark.parametrize("cron", ["not a cron", "a b c d e", "61 * * * *"])
def test_validate_rejects_bad_cron(cron: str) -> None:
    with pytest.raises(ValueError, match="valid cron"):
        validate_workflow_spec({**_MINIMAL, "triggers": [{"cron": cron}]})


def test_validate_checks_timezone_and_accepts_dbos_seconds_cron() -> None:
    triggers = [{"cron": "*/30 * * * * *", "timezone": "Australia/Sydney"}]
    validate_workflow_spec({**_MINIMAL, "triggers": triggers})
    with pytest.raises(ValueError, match="unknown timezone"):
        validate_workflow_spec({**_MINIMAL, "triggers": [{**triggers[0], "timezone": "Not/AZone"}]})


def test_validate_rejects_bad_config_overlay() -> None:
    bad = {**_MINIMAL, "config": {"not_a_real_section": {"foo": "bar"}}}
    with pytest.raises(ValidationError):
        validate_workflow_spec(bad)


def test_validate_overlay_ignores_run_identity_like_a_preset() -> None:
    raw = {**_MINIMAL, "config": {"dataset": {"name": "x"}, "paths": {"input_root": "."}}}
    validate_workflow_spec(raw)  # dropped by `_validate_overlay`, not a TypeError


def test_validate_round_trips_conditions_destinations_and_triggers() -> None:
    raw = {
        **_MINIMAL,
        "conditions": [{"column": "ext", "op": "eq", "value": ".pdf"}],
        "destinations": [{"kind": "webhook", "target": "https://example.invalid/hook"}],
        "triggers": [{"cron": "0 0 * * *"}],
    }
    spec = validate_workflow_spec(raw)
    assert spec.conditions[0].column == "ext"
    assert spec.destinations[0].kind == "webhook"
    assert spec.triggers[0].cron == "0 0 * * *"


@pytest.mark.parametrize(("workflow_id", "safe"), [
    ("ok-1", True), ("a", True), ("A.b_c-9", True),
    ("", False), (".", False), ("..", False), ("../escape", False), ("a/b", False),
])
def test_workflow_id_safety(workflow_id: str, safe: bool) -> None:
    assert is_safe_workflow_id(workflow_id) is safe


# --- digest -------------------------------------------------------------------


def test_digest_stable_then_changes_with_spec() -> None:
    spec_a = validate_workflow_spec(_MINIMAL)
    spec_b = validate_workflow_spec(_MINIMAL)
    spec_c = validate_workflow_spec({**_MINIMAL, "steps": [{"stage": "chunk"}]})
    assert workflow_digest(spec_a) == workflow_digest(spec_b)
    assert workflow_digest(spec_a).startswith("sha256:")
    assert workflow_digest(spec_a) != workflow_digest(spec_c)


# --- storage ------------------------------------------------------------------


def test_round_trip_save_get_list_delete(store) -> None:
    spec = validate_workflow_spec(_MINIMAL)
    stored = store.save("wf-1", spec)
    assert stored.id == "wf-1"
    assert stored.digest == workflow_digest(spec)

    fetched = store.get("wf-1")
    assert fetched is not None and fetched.spec.preset == "DEFAULT-Isaacus"
    assert [w.id for w in store.list()] == ["wf-1"]

    store.delete("wf-1")
    assert store.get("wf-1") is None
    assert store.list() == []


def test_save_rejects_unsafe_id(store) -> None:
    spec = validate_workflow_spec(_MINIMAL)
    with pytest.raises(ValueError, match="unsafe workflow id"):
        store.save("../escape", spec)


def test_save_update_same_owner_replaces(store) -> None:
    spec_a = validate_workflow_spec(_MINIMAL)
    spec_b = validate_workflow_spec({**_MINIMAL, "steps": [{"stage": "chunk"}]})
    store.save("wf-1", spec_a, owner="alice")
    first = store.get("wf-1")
    updated = store.save("wf-1", spec_b, owner="alice")
    assert updated.created_at == first.created_at
    assert updated.digest == workflow_digest(spec_b)


def test_owner_conflict_on_save_and_delete(store) -> None:
    spec = validate_workflow_spec(_MINIMAL)
    store.save("wf-1", spec, owner="alice")
    with pytest.raises(WorkflowOwnedError):
        store.save("wf-1", spec, owner="bob")
    with pytest.raises(WorkflowOwnedError):
        store.delete("wf-1", owner="bob")
    store.save("wf-1", spec, owner=None)  # unscoped caller is not checked


def test_save_losing_an_insert_race_checks_the_winners_owner(store, monkeypatch) -> None:
    spec = validate_workflow_spec(_MINIMAL)
    store.save("wf-1", spec, owner="alice")
    real_get_row = store._get_row
    calls = iter([None])  # the first read misses alice's row, as a racing save would
    monkeypatch.setattr(store, "_get_row", lambda conn, wid: next(calls, real_get_row(conn, wid)))
    with pytest.raises(WorkflowOwnedError):
        store.save("wf-1", spec, owner="bob")


def test_get_and_list_scoped_to_owner(store) -> None:
    spec = validate_workflow_spec(_MINIMAL)
    store.save("wf-1", spec, owner="alice")
    store.save("wf-2", spec, owner="bob")
    assert store.get("wf-1", owner="bob") is None
    assert store.get("wf-1", owner="alice") is not None
    assert [w.id for w in store.list(owner="alice")] == ["wf-1"]
    assert {w.id for w in store.list()} == {"wf-1", "wf-2"}
