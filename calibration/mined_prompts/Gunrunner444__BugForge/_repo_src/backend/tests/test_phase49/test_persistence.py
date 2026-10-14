from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from app.discovery.orchestration import (
    MemoryStore,
    ResumeError,
    ResumeStatus,
    SqlStore,
    dump_state,
    load_state,
)
from app.discovery.orchestration.model import (
    SCHEMA_VERSION,
    ExecutionPhase,
    OrchestratorState,
    PersistenceError,
)
from tests.test_phase49.phase49_support import (
    fuzz_engine,
    make_orchestrator,
    make_request,
    static_engine,
)


def _finished(tmp_path, store=None):
    orch = make_orchestrator([static_engine(), fuzz_engine()], make_request(tmp_path), store=store)
    orch.run()
    return orch


def test_state_round_trips_through_canonical_text(tmp_path) -> None:
    state = _finished(tmp_path).state
    text = dump_state(state)
    loaded = load_state(text)
    assert loaded == state
    assert dump_state(loaded) == text
    assert json.loads(text)["schema_version"] == SCHEMA_VERSION


def test_dump_is_stable_for_equal_state(tmp_path) -> None:
    first = dump_state(_finished(tmp_path).state)
    second = dump_state(_finished(tmp_path).state)
    assert first == second


def _tamper(text: str, mutate) -> str:
    data = json.loads(text)
    mutate(data)
    return json.dumps(data)


@pytest.mark.parametrize(
    ("mutate", "status"),
    [
        (lambda d: d.update(round=99), ResumeStatus.CORRUPT),
        (lambda d: d.update(schema_version=SCHEMA_VERSION + 1), ResumeStatus.UNSUPPORTED_VERSION),
        (lambda d: d.update(schema_version=0), ResumeStatus.MIGRATION_REQUIRED),
        (lambda d: d.pop("schema_version"), ResumeStatus.CORRUPT),
        (lambda d: d.update(schema_version=True), ResumeStatus.CORRUPT),
        (lambda d: d.update(state="verified"), ResumeStatus.CORRUPT),
        (lambda d: d.update(unexpected="field"), ResumeStatus.CORRUPT),
        (lambda d: d["identity"].update(campaign_id=5), ResumeStatus.CORRUPT),
    ],
)
def test_a_modified_or_unknown_document_is_refused(tmp_path, mutate, status) -> None:
    text = dump_state(_finished(tmp_path).state)
    with pytest.raises(ResumeError) as excinfo:
        load_state(_tamper(text, mutate))
    assert excinfo.value.status is status


def test_not_json_and_not_an_object_are_corrupt() -> None:
    for text in ("{", "[]", "null", '"x"'):
        with pytest.raises(ResumeError) as excinfo:
            load_state(text)
        assert excinfo.value.status is ResumeStatus.CORRUPT
    with pytest.raises(ResumeError):
        load_state(" " * 2_100_000)


def test_deeply_nested_documents_are_refused_not_crashed() -> None:
    nested = "[" * 5000 + "]" * 5000
    with pytest.raises(ResumeError) as excinfo:
        load_state(nested)
    assert excinfo.value.status is ResumeStatus.CORRUPT
    data = {"schema_version": SCHEMA_VERSION, "identity": {"campaign_id": nested}}
    with pytest.raises(ResumeError):
        load_state(json.dumps(data))


def test_a_state_too_large_to_load_is_not_saved(tmp_path, monkeypatch) -> None:
    from app.discovery.orchestration import store as store_module

    state = _finished(tmp_path).state
    monkeypatch.setattr(store_module, "MAX_DOCUMENT_BYTES", 100)
    with pytest.raises(PersistenceError, match="larger"):
        dump_state(state)


def test_a_document_cannot_claim_negative_evidence_proves_safety(tmp_path) -> None:
    from app.discovery.orchestration.model import NegativeEvidence

    state = _finished(tmp_path).state
    state.negatives = (
        NegativeEvidence("not_observed", "fuzzing", "ityfuzz", "Vault.withdraw", "safe", True),
    )
    with pytest.raises(ResumeError, match="safety"):
        load_state(dump_state(state))


def test_negative_budget_and_broken_history_are_corrupt(tmp_path) -> None:
    state = _finished(tmp_path).state
    state.budget.consumed["rounds"] = -1
    with pytest.raises(ResumeError, match="negative budget"):
        load_state(dump_state(state))
    state = _finished(tmp_path).state
    state.decisions = state.decisions[1:]
    with pytest.raises(ResumeError):
        load_state(dump_state(state))


def test_memory_store_refuses_to_rewrite_decision_history(tmp_path) -> None:
    store = MemoryStore()
    orch = _finished(tmp_path, store)
    forged = load_state(dump_state(orch.state))
    from dataclasses import replace

    changed = replace(forged.decisions[0], rationale="rewritten").sealed()
    forged.decisions = (changed, *forged.decisions[1:])
    with pytest.raises(PersistenceError, match="immutable"):
        store.save(forged)


def test_resume_restores_state_and_never_raises_budget(tmp_path) -> None:
    store = MemoryStore()
    request = make_request(tmp_path)
    first = make_orchestrator([static_engine(), fuzz_engine()], request, store=store, max_rounds=1)
    first.run()
    second = make_orchestrator(
        [static_engine(), fuzz_engine()], request, store=store, max_rounds=6, max_engines=6
    )
    restored = second.start()
    assert restored.resume_status is ResumeStatus.RESTORED
    assert restored.resume_count == 1
    assert restored.budget.limits["rounds"] == 1
    assert restored.budget.consumed["rounds"] >= 1
    assert restored.evidence == first.state.evidence
    assert restored.decisions == first.state.decisions


def test_resume_of_an_interrupted_run_marks_the_started_execution_unknown(tmp_path) -> None:
    store = MemoryStore()
    request = make_request(tmp_path)
    orch = make_orchestrator([static_engine(), fuzz_engine()], request, store=store)
    orch.step()
    # simulate a crash after the engine started but before the result was stored
    state = orch.state
    from dataclasses import replace

    started = replace(
        state.executions[0],
        execution_key="ex_crashed",
        phase=ExecutionPhase.STARTED,
        status="started",
        consumed={"attempts": 1, "engines": 1, "rounds": 1, "executions": 1, "fuzz_runs": 1},
    )
    state.executions = (*state.executions, started)
    state.state = OrchestratorState.EXECUTING
    state.budget.reserved = {"attempts": 1}
    store.save(state)

    resumed = make_orchestrator([static_engine(), fuzz_engine()], request, store=store)
    restored = resumed.start()
    crashed = next(r for r in restored.executions if r.execution_key == "ex_crashed")
    assert crashed.phase is ExecutionPhase.UNKNOWN
    assert crashed.retryable is True
    assert restored.budget.reserved == {}
    assert restored.budget.consumed["fuzz_runs"] == 1
    assert restored.state in {OrchestratorState.CONTINUE, OrchestratorState.PLANNING}
    assert restored.decisions[-1].source.value == "recovery"
    final = resumed.run()
    assert "ex_crashed" in {r.execution_key for r in final.executions}
    assert sum(r.execution_key == "ex_crashed" for r in final.executions) == 1


def test_a_changed_snapshot_or_compiler_makes_the_campaign_stale(tmp_path) -> None:
    store = MemoryStore()
    request = make_request(tmp_path)
    original = make_orchestrator([static_engine()], request, store=store)
    original.run()
    for key, value in (("source_snapshot", "snap2"), ("compiler_configuration", "solc-0.8.25")):
        changed = make_request(tmp_path, **{key: value})
        from app.discovery.orchestration import identity_from_request

        identity = identity_from_request(changed, campaign_id=original.identity.campaign_id)
        other = make_orchestrator([static_engine()], changed, store=store, identity=identity)
        with pytest.raises(ResumeError) as excinfo:
            other.start()
        assert excinfo.value.status is ResumeStatus.STALE_IDENTITY


def test_corrupt_stored_document_stops_resume(tmp_path) -> None:
    store = MemoryStore()
    request = make_request(tmp_path)
    first = make_orchestrator([static_engine()], request, store=store)
    first.run()
    campaign = first.identity.campaign_id
    store.documents[campaign] = store.documents[campaign].replace('"round":1', '"round":7')
    again = make_orchestrator([static_engine()], request, store=store)
    with pytest.raises(ResumeError) as excinfo:
        again.start()
    assert excinfo.value.status is ResumeStatus.CORRUPT


def test_resume_from_stopped_does_nothing_without_an_explicit_request(tmp_path) -> None:
    store = MemoryStore()
    request = make_request(tmp_path, fork="true", sequence_id="s")
    from tests.test_phase49.phase49_runtime import FORK_EXTRA, runtime_engine

    request.extra.update(FORK_EXTRA)
    engines = [static_engine(), fuzz_engine(), runtime_engine()]
    first = make_orchestrator(engines, request, store=store)
    first.run()
    assert first.state.state is OrchestratorState.STOPPED
    quiet = make_orchestrator(engines, request, store=store)
    assert quiet.start().state is OrchestratorState.STOPPED
    assert quiet.step() is False
    loud = make_orchestrator(engines, request, store=store)
    assert loud.start(explicit_resume=True).state is OrchestratorState.PLANNING


def test_unavailable_engines_are_superseded_when_they_come_back(tmp_path) -> None:
    store = MemoryStore()
    request = make_request(tmp_path)
    down = static_engine()
    down.available = False
    first = make_orchestrator([down], request, store=store)
    assert first.run().state is OrchestratorState.STOPPED
    assert first.state.executions == ()
    down.available = True
    again = make_orchestrator([down], request, store=store)
    again.start(explicit_resume=True)
    assert again.run().evidence


# ---- SQL store ----------------------------------------------------------------------------


@pytest.fixture
def session():
    from app.models.discovery_orchestration import (
        DBOrchestrationCampaign,
        DBOrchestrationDecision,
    )

    engine = create_engine("sqlite://")
    DBOrchestrationCampaign.__table__.create(engine)  # type: ignore[attr-defined]
    DBOrchestrationDecision.__table__.create(engine)  # type: ignore[attr-defined]
    with Session(engine) as value:
        yield value


def test_sql_store_round_trip_and_decision_history(session, tmp_path) -> None:
    store = SqlStore(session)
    orch = _finished(tmp_path, store)
    campaign = orch.identity.campaign_id
    loaded = SqlStore(session).load(campaign)
    assert loaded == orch.state
    history = store.decisions(campaign)
    assert history == orch.state.decisions
    assert [r.sequence for r in history] == list(range(1, len(history) + 1))
    from app.models.discovery_orchestration import DBOrchestrationCampaign

    row = session.scalars(select(DBOrchestrationCampaign)).one()
    assert row.state == orch.state.state.value
    assert row.stop_reason == orch.state.stop_reason
    assert row.revision >= 2


def test_sql_store_detects_a_concurrent_writer_and_unloaded_overwrites(session, tmp_path) -> None:
    store = SqlStore(session)
    orch = _finished(tmp_path, store)
    campaign = orch.identity.campaign_id
    stranger = SqlStore(session)
    state = stranger.load(campaign)
    assert state is not None
    store.save(orch.state)
    with pytest.raises(PersistenceError, match="changed"):
        stranger.save(state)
    fresh = SqlStore(session)
    with pytest.raises(PersistenceError, match="not loaded"):
        fresh.save(orch.state)
    assert fresh.load("missing") is None


def test_sql_store_refuses_rewritten_history(session, tmp_path) -> None:
    from dataclasses import replace

    store = SqlStore(session)
    orch = _finished(tmp_path, store)
    forged = load_state(dump_state(orch.state))
    forged.decisions = (replace(forged.decisions[0], rationale="x").sealed(), *forged.decisions[1:])
    with pytest.raises(PersistenceError, match="immutable"):
        store.save(forged)


def test_sql_store_rejects_a_tampered_row(session, tmp_path) -> None:
    from app.models.discovery_orchestration import DBOrchestrationCampaign

    store = SqlStore(session)
    orch = _finished(tmp_path, store)
    row = session.get(DBOrchestrationCampaign, orch.identity.campaign_id)
    assert row is not None
    row.document = row.document.replace('"round":2', '"round":5')
    session.commit()
    with pytest.raises(ResumeError):
        SqlStore(session).load(orch.identity.campaign_id)


def test_orchestrator_resumes_from_the_sql_store(session, tmp_path) -> None:
    request = make_request(tmp_path)
    first = make_orchestrator([static_engine(), fuzz_engine()], request, store=SqlStore(session))
    first.run()
    second = make_orchestrator([static_engine(), fuzz_engine()], request, store=SqlStore(session))
    restored = second.start()
    assert restored.resume_status is ResumeStatus.RESTORED
    assert second.run().decision_number == first.state.decision_number


# ---- migration ----------------------------------------------------------------------------


def _load_migration():
    path = Path(__file__).resolve().parents[2] / "alembic/versions/027_phase49_orchestration.py"
    spec = importlib.util.spec_from_file_location("migration_027", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_027_follows_026_and_matches_the_models() -> None:
    script = ScriptDirectory.from_config(Config("alembic.ini"))
    revision = script.get_revision("027")
    assert revision is not None and revision.down_revision == "026"
    assert script.get_current_head() in {"027", "028"}  # 028 (Phase 51) follows 027


def test_migration_027_upgrades_and_downgrades_cleanly() -> None:
    from app.models.base import Base

    module = _load_migration()
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            module.upgrade()
        names = set(inspect(connection).get_table_names())
        assert {"discovery_orchestration_campaigns", "discovery_orchestration_decisions"} <= names
        for table in ("discovery_orchestration_campaigns", "discovery_orchestration_decisions"):
            migrated = {c["name"] for c in inspect(connection).get_columns(table)}
            assert migrated == set(Base.metadata.tables[table].c.keys())
        with Operations.context(context):
            module.downgrade()
        assert not {"discovery_orchestration_campaigns", "discovery_orchestration_decisions"} & set(
            inspect(connection).get_table_names()
        )
