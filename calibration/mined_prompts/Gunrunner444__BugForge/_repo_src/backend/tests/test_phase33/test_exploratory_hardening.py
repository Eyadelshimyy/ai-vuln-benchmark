"""Phase 33 exploratory persistence, identity, and sandbox boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.mock_provider import MockLLMProvider
from app.execution.base import ExecutionResult, TestExecutor
from app.execution.docker_executor import collect_output_artifacts
from app.repositories.security_agent_repo import SecurityAgentRepository
from app.security_agent.agent import ResearchSession
from app.security_agent.budget import SessionBudget
from app.security_agent.schemas import ExploratoryTestArgs, ResearchHypothesis
from app.security_agent.states import ResearchMode
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.errors import SafetyLimitExceededError
from app.security_testing.exploratory import (
    ExploratoryClassification,
    ExploratoryContext,
    ExploratoryEngine,
    ExploratoryTestCandidate,
    ExploratoryTestPolicy,
    profile_availability,
    release_exploratory_engine,
    snapshot_hash,
    validate_solidity_test,
)
from app.security_testing.safety import SafetyLimits

_POLICY = ExploratoryTestPolicy(enabled=True, max_tests=4, max_iterations=4, max_seconds=30)


class RecordingExecutor(TestExecutor):
    def __init__(self, result: ExecutionResult | None = None) -> None:
        self.calls = 0
        self.result = result or ExecutionResult(1, "fail", "", 0.01)

    async def execute(self, config):
        self.calls += 1
        return self.result

    async def is_available(self) -> bool:
        return True


def _candidate(**overrides: object) -> ExploratoryTestCandidate:
    payload: dict[str, object] = {
        "target": "src/thing.py",
        "language": "python",
        "framework": "pytest",
        "target_file": "src/thing.py",
        "target_symbol": "Thing.method",
        "hypothesis_id": "hyp-a",
        "test_code": "def test_boundary():\n    assert True\n",
        "rationale": "boundary",
        "expected_behavior": "zero stays zero",
        "oracle": "assertion",
        "project_id": "proj",
        "candidate_hash": "model-supplied-hash",
    }
    payload.update(overrides)
    return ExploratoryTestCandidate(**payload)  # type: ignore[arg-type]


def _context(session: ResearchSession, root: Path, **overrides: object) -> ExploratoryContext:
    payload: dict[str, object] = {
        "session_id": session.id,
        "project_id": session.project_id,
        "mode": "lab",
        "hypotheses": {item.id: item.target for item in session.hypotheses},
        "repo_path": str(root),
        "commit_sha": "abc123",
        "snapshot_hash": snapshot_hash(str(root)),
        "graph": session.graph,
    }
    payload.update(overrides)
    return ExploratoryContext(**payload)  # type: ignore[arg-type]


def _session(tmp_path: Path) -> ResearchSession:
    engine = SecurityTestEngine.lab("lab", limits=SafetyLimits.lab())
    session = ResearchSession(
        project_id="proj",
        target="http://127.0.0.1/health",
        mode=ResearchMode.LAB,
        engine=engine,
        provider=MockLLMProvider(),
        repo_root=str(tmp_path),
    )
    session.hypotheses.append(
        ResearchHypothesis(
            id="hyp-a",
            title="boundary",
            vulnerability_class="logic",
            target="src/thing.py",
            reason="check zero",
        )
    )
    return session


def _engine(session: ResearchSession, executor: TestExecutor) -> ExploratoryEngine:
    return ExploratoryEngine(
        policy=_POLICY,
        executor=executor,
        docker_available=True,
        pytest_ready=True,
        records=session.exploratory_attempts,
    )


def test_snapshot_hash_depends_on_bytes(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "f.txt").write_bytes(b"aaaa")
    (right / "f.txt").write_bytes(b"bbbb")
    assert snapshot_hash(str(left)) != snapshot_hash(str(right))
    (right / "f.txt").write_bytes(b"aaaa")
    assert snapshot_hash(str(left)) == snapshot_hash(str(right))
    git = left / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    assert snapshot_hash(str(left)) == snapshot_hash(str(right))


def test_candidate_identity_uses_server_fields(tmp_path: Path) -> None:
    session = _session(tmp_path)
    context = _context(session, tmp_path)
    engine = _engine(session, RecordingExecutor())
    first = engine._identity(_candidate(), context, "python-pytest-v1")
    assert first != "model-supplied-hash"
    other = engine._identity(_candidate(hypothesis_id="hyp-b"), context, "python-pytest-v1")
    file_changed = engine._identity(
        _candidate(target_file="src/other.py"), context, "python-pytest-v1"
    )
    oracle_changed = engine._identity(_candidate(oracle="invariant"), context, "python-pytest-v1")
    behavior_changed = engine._identity(
        _candidate(expected_behavior="one stays one"), context, "python-pytest-v1"
    )
    assert len({first, other, file_changed, oracle_changed, behavior_changed}) == 5


@pytest.mark.asyncio
async def test_parent_binding_rejects_cross_identity(tmp_path: Path) -> None:
    session = _session(tmp_path)
    engine = _engine(session, RecordingExecutor())
    context = _context(session, tmp_path)
    first = await engine.run(_candidate(), context)
    cross_hyp = await engine.run(
        _candidate(
            hypothesis_id="hyp-a",
            parent_attempt_id=first.attempt_id,
            test_code="def test_child():\n    assert True\n",
            project_id="other",
        ),
        context,
    )
    assert cross_hyp.classification == ExploratoryClassification.INVALID.value
    session.hypotheses.append(
        ResearchHypothesis(
            id="hyp-b",
            title="other",
            vulnerability_class="logic",
            target="src/thing.py",
            reason="other",
        )
    )
    moved = _context(session, tmp_path)
    rejected = await engine.run(
        _candidate(
            hypothesis_id="hyp-b",
            parent_attempt_id=first.attempt_id,
            test_code="def test_other_hyp():\n    assert True\n",
        ),
        moved,
    )
    assert rejected.classification == ExploratoryClassification.INVALID.value
    snapshot = _context(session, tmp_path, snapshot_hash="different")
    changed = await engine.run(
        _candidate(
            parent_attempt_id=first.attempt_id,
            test_code="def test_snap():\n    assert True\n",
        ),
        snapshot,
    )
    assert changed.classification == ExploratoryClassification.INVALID.value


def test_artifact_collector_does_not_follow_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "out"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "ok.txt").write_text("safe", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("host-secret", encoding="utf-8")
    (root / "escape").symlink_to(secret)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (root / "linked-out").symlink_to(outside)
    (root / ("n" * 200)).write_text("long", encoding="utf-8")
    artifacts = collect_output_artifacts(str(root))
    assert artifacts.get("nested/ok.txt") == "safe"
    assert "host-secret" not in "".join(artifacts.values())
    assert "outside" not in "".join(artifacts.values())
    assert all(len(name) <= 128 for name in artifacts)


def test_missing_pytest_is_unavailable_not_a_failure() -> None:
    assert profile_availability("python", "pytest", pytest_ready=False, forge_available=False) == (
        "UNAVAILABLE"
    )
    assert profile_availability("ruby", "rspec", pytest_ready=True, forge_available=True) == (
        "UNSUPPORTED"
    )


@pytest.mark.asyncio
async def test_missing_pytest_does_not_execute(tmp_path: Path) -> None:
    session = _session(tmp_path)
    executor = RecordingExecutor()
    engine = ExploratoryEngine(
        policy=_POLICY,
        executor=executor,
        docker_available=True,
        pytest_ready=False,
        records=session.exploratory_attempts,
    )
    attempt = await engine.run(_candidate(), _context(session, tmp_path))
    assert attempt.classification == ExploratoryClassification.UNAVAILABLE.value
    assert attempt.executed is False
    assert executor.calls == 0
    assert "pip" in attempt.detail


def test_foundry_host_cheatcodes_are_blocked() -> None:
    base = "pragma solidity ^0.8.20;\ncontract T { function test_ok() public { assert(true); } }\n"
    for snippet in (
        'vm.getEnv("KEY")',
        'vm.readFile("x")',
        'vm.writeLine("x", "y")',
        'vm.createSelectFork("http://rpc")',
        'vm.ffi("ls")',
    ):
        assert (validate_solidity_test(base + snippet + "\n") or "").startswith("BLOCKED")
    with pytest.raises(Exception):
        ExploratoryTestArgs.model_validate(
            {
                "hypothesis_id": "hyp",
                "language": "solidity",
                "framework": "foundry",
                "target_symbol": "T.test_ok",
                "test_code": base,
                "expected_behavior": "ok",
                "oracle": "assertion",
                "root": "/tmp",
            }
        )


def test_exploratory_budget_buckets_stay_separate() -> None:
    budget = SessionBudget(max_exploratory_tests=4, max_exploratory_iterations=3)
    budget.plan("exploratory_tests", 1)
    budget.reserve("exploratory_tests", 1)
    budget.plan("exploratory_iterations", 1)
    budget.plan("exploratory_seconds", 2)
    snap = budget.snapshot()
    assert snap["planned"]["exploratory_tests"] == 1
    assert snap["reserved"]["exploratory_tests"] == 1
    assert snap["planned"]["tool_calls"] == 0
    budget.consume("exploratory")
    consumed = budget.snapshot()["consumed"]
    assert consumed["exploratory_tests"] == 1
    assert consumed["exploratory_iterations"] == 1
    assert budget.remaining()["exploratory_tests"] == 3
    with pytest.raises(SafetyLimitExceededError):
        SessionBudget(max_exploratory_seconds=1, exploratory_seconds=1).consume("exploratory")


@pytest.mark.asyncio
async def test_attempt_dedup_and_flaky_survive_reconstruction(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    session = _session(tmp_path)
    failing = RecordingExecutor(ExecutionResult(1, "assert failed", "", 0.02))
    engine = _engine(session, failing)
    context = _context(session, tmp_path)
    first = await engine.run(_candidate(), context)
    assert first.classification == ExploratoryClassification.FAILED.value
    repo = SecurityAgentRepository(db_session)
    await repo.save_session(session)
    await db_session.commit()
    db_session.expire_all()
    release_exploratory_engine(session.id)
    restored_agent = await repo.reconstruct(session.id)
    assert restored_agent is not None
    passing = RecordingExecutor(ExecutionResult(0, "ok", "", 0.01))
    restored = ExploratoryEngine(
        policy=_POLICY,
        executor=passing,
        docker_available=True,
        pytest_ready=True,
        records=restored_agent.session.exploratory_attempts,
    )
    duplicate = await restored.run(_candidate(), context)
    assert duplicate.classification == ExploratoryClassification.DUPLICATE.value
    replay = restored.replay(first.attempt_id, context)
    assert replay is not None
    assert replay.target_file == "src/thing.py"
    assert replay.target != restored_agent.session.project_id
    assert replay.oracle == "assertion"
    changed = _context(session, tmp_path, snapshot_hash="other", commit_sha="def")
    assert restored.replay(first.attempt_id, changed) is None
    second = await restored.run(
        _candidate(rationale="replay:again", test_code=_candidate().test_code), context
    )
    assert second.classification == ExploratoryClassification.PASSED.value
    assert restored.repeat_behavior(first.candidate_id) == "flaky"
    child = await restored.run(
        _candidate(
            test_code="def test_control():\n    assert False\n",
            parent_attempt_id=first.attempt_id,
            follow_up="negative_control",
            rationale="control",
        ),
        context,
    )
    assert child.parent_attempt == first.attempt_id
    assert any(rel == "motivates" for _src, _dst, rel in restored_agent.session.graph.edges)
    assert all(
        node.extra.get("verified") is not True
        for node in restored_agent.session.graph.nodes.values()
    )
