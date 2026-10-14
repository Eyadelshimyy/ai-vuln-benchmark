"""Phase 34 regressions for exploratory identity, sandboxing, and evidence."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.mock_provider import MockLLMProvider
from app.domain.evidence import VERIFICATION_PROVENANCE, EvidenceProvenance
from app.execution.base import ExecutionConfig, ExecutionResult, TestExecutor
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
    profile_for,
    release_exploratory_engine,
    snapshot_hash,
    validate_solidity_test,
)
from app.security_testing.safety import SafetyLimits

_POLICY = ExploratoryTestPolicy(enabled=True, max_tests=4, max_iterations=4, max_seconds=30)
_SOLIDITY = "pragma solidity ^0.8.20;\ncontract T { function test_ok() public { assert(true); } }\n"


class RecordingExecutor(TestExecutor):
    def __init__(self, result: ExecutionResult | None = None) -> None:
        self.calls = 0
        self.config: ExecutionConfig | None = None
        self.result = result or ExecutionResult(1, "fail", "", 0.01)

    async def execute(self, config: ExecutionConfig) -> ExecutionResult:
        self.calls += 1
        self.config = config
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


def _session(tmp_path: Path, hypothesis_id: str = "hyp-a") -> ResearchSession:
    session = ResearchSession(
        project_id="proj",
        target="http://127.0.0.1/health",
        mode=ResearchMode.LAB,
        engine=SecurityTestEngine.lab("lab", limits=SafetyLimits.lab()),
        provider=MockLLMProvider(),
        repo_root=str(tmp_path),
    )
    session.hypotheses.append(
        ResearchHypothesis(
            id=hypothesis_id,
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


def test_snapshot_hash_changes_with_bytes_not_size(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "same.txt").write_bytes(b"aaaa")
    (right / "same.txt").write_bytes(b"bbbb")
    assert len((left / "same.txt").read_bytes()) == len((right / "same.txt").read_bytes())
    assert snapshot_hash(str(left)) != snapshot_hash(str(right))
    (right / "same.txt").write_bytes(b"aaaa")
    assert snapshot_hash(str(left)) == snapshot_hash(str(right))


def test_identity_includes_hypothesis_target_and_oracle(tmp_path: Path) -> None:
    session = _session(tmp_path)
    context = _context(session, tmp_path)
    engine = _engine(session, RecordingExecutor())
    first = engine._identity(_candidate(), context, "python-pytest-v1")
    assert first != "model-supplied-hash"
    variants = [
        engine._identity(_candidate(hypothesis_id="hyp-b"), context, "python-pytest-v1"),
        engine._identity(_candidate(target_file="src/other.py"), context, "python-pytest-v1"),
        engine._identity(
            _candidate(expected_behavior="one stays one"), context, "python-pytest-v1"
        ),
        engine._identity(_candidate(oracle="invariant"), context, "python-pytest-v1"),
    ]
    assert first not in variants
    assert len(set(variants)) == 4


@pytest.mark.asyncio
async def test_parent_binding_rejects_other_scopes(tmp_path: Path) -> None:
    session = _session(tmp_path)
    engine = _engine(session, RecordingExecutor())
    context = _context(session, tmp_path)
    first = await engine.run(_candidate(), context)
    child_code = "def test_child():\n    assert True\n"
    other_session = _context(session, tmp_path, session_id="other-session")
    rejected_session = await engine.run(
        _candidate(parent_attempt_id=first.attempt_id, test_code=child_code),
        other_session,
    )
    assert rejected_session.classification == ExploratoryClassification.INVALID.value
    other_project = _context(session, tmp_path, project_id="other-project")
    rejected_project = await engine.run(
        _candidate(parent_attempt_id=first.attempt_id, test_code=child_code, project_id=""),
        other_project,
    )
    assert rejected_project.classification == ExploratoryClassification.INVALID.value
    session.hypotheses.append(
        ResearchHypothesis(
            id="hyp-b",
            title="other",
            vulnerability_class="logic",
            target="src/thing.py",
            reason="other",
        )
    )
    rejected_hyp = await engine.run(
        _candidate(
            hypothesis_id="hyp-b",
            parent_attempt_id=first.attempt_id,
            test_code=child_code,
        ),
        _context(session, tmp_path),
    )
    assert rejected_hyp.classification == ExploratoryClassification.INVALID.value
    rejected_snap = await engine.run(
        _candidate(parent_attempt_id=first.attempt_id, test_code=child_code),
        _context(session, tmp_path, snapshot_hash="other-snapshot"),
    )
    assert rejected_snap.classification == ExploratoryClassification.INVALID.value


def test_artifact_symlink_and_long_name_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "out"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "ok.txt").write_text("safe", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("host-secret", encoding="utf-8")
    (root / "escape").symlink_to(secret)
    (root / "linked-out").symlink_to(tmp_path / "missing-outside")
    (root / ("n" * 200)).write_text("long", encoding="utf-8")
    artifacts = collect_output_artifacts(str(root))
    assert artifacts.get("nested/ok.txt") == "safe"
    assert "host-secret" not in "".join(artifacts.values())
    assert all(len(name) <= 128 for name in artifacts)


def test_python_profile_is_honest_and_offline(tmp_path: Path) -> None:
    assert profile_availability("python", "pytest", pytest_ready=False, forge_available=False) == (
        "UNAVAILABLE"
    )
    assert profile_availability("python", "pytest", pytest_ready=True, forge_available=False) == (
        "AVAILABLE"
    )
    assert profile_for("python", "pytest").network == "none"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_missing_pytest_does_not_execute_or_use_the_network(tmp_path: Path) -> None:
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
    ready = ExploratoryEngine(
        policy=_POLICY,
        executor=executor,
        docker_available=True,
        pytest_ready=True,
        records=session.exploratory_attempts,
    )
    await ready.run(
        _candidate(test_code="def test_other():\n    assert True\n"), _context(session, tmp_path)
    )
    assert executor.config is not None
    assert executor.config.allow_network is False


def test_foundry_command_and_cheatcodes_are_server_owned() -> None:
    profile = profile_for("solidity", "foundry", forge_available=True)
    assert profile is not None
    assert profile.container_command()[:4] == [
        "forge",
        "test",
        "--root",
        "/bugforge-output/project",
    ]
    command = " ".join(profile.container_command())
    assert "--fork-url" not in command
    assert profile.network == "none"
    for snippet in (
        'vm.envUint("KEY")',
        'vm.getEnv("KEY")',
        'vm.readFile("x")',
        'vm.writeFile("x", "y")',
        'vm.createFork("http://rpc")',
        'vm.rpc("eth_chainId", "")',
        'vm.ffi("ls")',
        'process("ls")',
    ):
        reason = validate_solidity_test(_SOLIDITY + snippet + "\n")
        assert reason is not None and reason.startswith("BLOCKED")
    with pytest.raises(ValidationError):
        ExploratoryTestArgs.model_validate(
            {
                "hypothesis_id": "hyp",
                "language": "solidity",
                "framework": "foundry",
                "target_symbol": "T.test_ok",
                "test_code": _SOLIDITY,
                "expected_behavior": "ok",
                "oracle": "assertion",
                "command": "forge test --root /tmp",
            }
        )


def test_budget_plan_reserve_consume_stays_on_exploratory_buckets() -> None:
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
async def test_reconstruction_keeps_dedup_replay_and_flaky_history(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    session = _session(tmp_path, "hyp-phase34")
    failing = RecordingExecutor(ExecutionResult(1, "assert failed", "", 0.02))
    engine = _engine(session, failing)
    context = _context(session, tmp_path)
    candidate = _candidate(hypothesis_id="hyp-phase34")
    first = await engine.run(candidate, context)
    assert first.classification == ExploratoryClassification.FAILED.value
    assert any(relation == "motivates" for _src, _dst, relation in session.graph.edges)
    assert any(relation == "executes" for _src, _dst, relation in session.graph.edges)
    assert EvidenceProvenance.SANDBOX_EXECUTION not in VERIFICATION_PROVENANCE
    repo = SecurityAgentRepository(db_session)
    await repo.save_session(session)
    await db_session.commit()
    db_session.expire_all()
    release_exploratory_engine(session.id)
    restored_agent = await repo.reconstruct(session.id)
    assert restored_agent is not None
    restored = ExploratoryEngine(
        policy=_POLICY,
        executor=RecordingExecutor(ExecutionResult(0, "ok", "", 0.01)),
        docker_available=True,
        pytest_ready=True,
        records=restored_agent.session.exploratory_attempts,
    )
    duplicate = await restored.run(candidate, context)
    assert duplicate.classification == ExploratoryClassification.DUPLICATE.value
    replay = restored.replay(first.attempt_id, context)
    assert replay is not None
    assert replay.target_file == "src/thing.py"
    assert replay.target_symbol == "Thing.method"
    assert replay.oracle == "assertion"
    assert replay.expected_behavior == "zero stays zero"
    assert replay.target != restored_agent.session.project_id
    assert (
        restored.replay(first.attempt_id, _context(session, tmp_path, snapshot_hash="changed"))
        is None
    )
    second = await restored.run(
        _candidate(hypothesis_id="hyp-phase34", rationale="replay:again"), context
    )
    assert second.classification == ExploratoryClassification.PASSED.value
    assert restored.repeat_behavior(first.candidate_id) == "flaky"
    child = await restored.run(
        _candidate(
            hypothesis_id="hyp-phase34",
            test_code="def test_control():\n    assert False\n",
            parent_attempt_id=first.attempt_id,
            follow_up="negative_control",
            rationale="control",
        ),
        context,
    )
    assert child.parent_attempt == first.attempt_id
    assert child.verified is False
