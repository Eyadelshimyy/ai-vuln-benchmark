"""Phase 35 Foundry workspace, exact replay, and fail-closed scope."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.mock_provider import MockLLMProvider
from app.execution.base import ExecutionConfig, ExecutionResult, TestExecutor
from app.repositories.security_agent_repo import SecurityAgentRepository
from app.security_agent.agent import ResearchSession, SecurityResearchAgent
from app.security_agent.evidence_graph import EvidenceGraph
from app.security_agent.schemas import ResearchHypothesis, ToolCallRequest
from app.security_agent.states import ResearchMode
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.errors import RestrictedActivityError
from app.security_testing.exploratory import (
    ExploratoryClassification,
    ExploratoryContext,
    ExploratoryEngine,
    ExploratoryTestCandidate,
    ExploratoryTestPolicy,
    prepare_foundry_workspace,
    profile_for,
    release_exploratory_engine,
    validate_solidity_test,
)
from app.security_testing.safety import SafetyLimits
from app.security_testing.target_manifest import TargetManifest, solidity_live_testing_permitted

_POLICY = ExploratoryTestPolicy(enabled=True, lab_only=True, max_tests=4, max_iterations=4)
_SOLIDITY = "pragma solidity ^0.8.20;\ncontract T { function test_ok() public { assert(true); } }\n"
_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "phase35_foundry"


class RecordingExecutor(TestExecutor):
    def __init__(self, result: ExecutionResult | None = None) -> None:
        self.configs: list[ExecutionConfig] = []
        self.result = result or ExecutionResult(0, "ok", "", 0.01)

    async def execute(self, config: ExecutionConfig) -> ExecutionResult:
        self.configs.append(config)
        return self.result

    async def is_available(self) -> bool:
        return True


def _candidate(**overrides: object) -> ExploratoryTestCandidate:
    payload: dict[str, object] = {
        "target": "src/Vault.sol",
        "language": "solidity",
        "framework": "foundry",
        "target_symbol": "Vault.probe",
        "hypothesis_id": "hyp-p35",
        "test_code": _SOLIDITY,
        "rationale": "probe vault",
        "expected_behavior": "probe returns one",
        "oracle": "assertion",
        "project_id": "proj",
        "session_id": "sess",
        "target_file": "src/Vault.sol",
        "candidate_hash": "model-supplied",
    }
    payload.update(overrides)
    return ExploratoryTestCandidate(**payload)  # type: ignore[arg-type]


def _context(repo: Path, **overrides: object) -> ExploratoryContext:
    payload: dict[str, object] = {
        "session_id": "sess",
        "project_id": "proj",
        "mode": "lab",
        "hypotheses": {"hyp-p35": "src/Vault.sol"},
        "repo_path": str(repo),
        "commit_sha": "abc",
        "snapshot_hash": "snap",
        "graph": EvidenceGraph(),
    }
    payload.update(overrides)
    return ExploratoryContext(**payload)  # type: ignore[arg-type]


def test_foundry_workspace_keeps_the_repository_read_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copytree(_FIXTURE, repo, dirs_exist_ok=True)
    before = (repo / "src" / "Vault.sol").read_bytes()
    output = tmp_path / "output"
    output.mkdir()
    project = prepare_foundry_workspace(repo, output, _SOLIDITY)
    assert project == output / "project"
    test_path = project / "test" / "Exploratory.t.sol"
    assert test_path.is_file()
    assert test_path.read_text(encoding="utf-8") == _SOLIDITY
    assert (project / "src" / "Vault.sol").read_text(encoding="utf-8").startswith("// SPDX")
    assert 'import "./Token.sol"' in (project / "src" / "Vault.sol").read_text(encoding="utf-8")
    assert (repo / "src" / "Vault.sol").read_bytes() == before
    assert not (repo / "test" / "Exploratory.t.sol").exists()
    profile = profile_for("solidity", "foundry", forge_available=True)
    assert profile is not None
    command = profile.container_command()
    assert command[:4] == ["forge", "test", "--root", "/bugforge-output/project"]
    assert "test/Exploratory.t.sol" in command
    assert "--fork-url" not in command
    assert "--offline" in command
    assert profile.network == "none"


@pytest.mark.asyncio
async def test_foundry_execution_classifies_toolchain_separately(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(_FIXTURE, repo)
    missing = ExploratoryEngine(policy=_POLICY, docker_available=True, forge_available=False)
    unavailable = await missing.run(_candidate(), _context(repo))
    assert unavailable.classification == ExploratoryClassification.UNAVAILABLE.value
    assert "forge" in unavailable.detail
    assert unavailable.executed is False
    cases = (
        (ExecutionResult(0, "ok", "", 0.01), ExploratoryClassification.PASSED, False),
        (
            ExecutionResult(1, "[FAIL. Reason: assertion failed]", "", 0.01),
            ExploratoryClassification.FAILED,
            True,
        ),
        (
            ExecutionResult(1, "Compiler run failed\nError (2314): Expected identifier", "", 0.01),
            ExploratoryClassification.COMPILATION_FAILED,
            False,
        ),
        (
            ExecutionResult(1, "", "", 0.01, timed_out=True),
            ExploratoryClassification.TIMEOUT,
            False,
        ),
        (
            ExecutionResult(1, "", "", 0.01, error_message="sandbox"),
            ExploratoryClassification.ERROR,
            False,
        ),
    )
    for result, classification, meaningful in cases:
        executor = RecordingExecutor(result)
        engine = ExploratoryEngine(
            policy=_POLICY,
            executor=executor,
            docker_available=True,
            forge_available=True,
        )
        context = _context(repo)
        attempt = await engine.run(_candidate(candidate_hash="ignore-me"), context)
        assert attempt.classification == classification.value
        assert attempt.meaningful is meaningful
        assert attempt.verified is False
        assert attempt.candidate_id != "ignore-me"
        assert executor.configs[0].allow_network is False
        assert executor.configs[0].command[3] == "/bugforge-output/project"
        assert executor.configs[0].environment == {"FOUNDRY_OFFLINE": "1"}
        extras = [
            node.extra
            for node in context.graph.nodes.values()
            if node.kind == "exploratory_attempt"
        ]
        assert extras and extras[0].get("network") == "none"
        assert extras[0].get("toolchain_failure") is (
            classification is ExploratoryClassification.COMPILATION_FAILED
        )


def test_cheatcode_families_fail_closed() -> None:
    snippets = {
        'vm.envString("KEY")': "BLOCKED",
        'vm.setEnv("KEY", "VALUE")': "BLOCKED",
        'vm.readFileBinary("x")': "BLOCKED",
        'vm.writeFileBinary("x", "")': "BLOCKED",
        'vm.createSelectFork("http://rpc")': "BLOCKED",
        "vm.rollFork(1)": "BLOCKED",
        'vm.rpcUrl("mainnet")': "BLOCKED",
        'vm.tryFfi("ls")': "BLOCKED",
        'vm.ffi("ls")': "BLOCKED",
        'vm.prompt("x")': "BLOCKED",
        "vm.projectRoot()": "BLOCKED",
        'vm.exists("foundry.toml")': "BLOCKED",
        '// forge-config: default.fs_permissions = [{ access = "read", path = "/"}]': "BLOCKED",
        "vm.broadcast()": "UNSUPPORTED",
    }
    for snippet, prefix in snippets.items():
        reason = validate_solidity_test(_SOLIDITY + snippet + "\n")
        assert reason is not None and reason.startswith(prefix)
    assert validate_solidity_test(_SOLIDITY + "vm.prank(address(1));\n") is None


def test_scope_manifest_fails_closed() -> None:
    web = TargetManifest(
        program_id="program",
        asset_id="api.example",
        target_type="web",
        allowed_modes=("live",),
        active_testing=True,
        operator_approved=True,
        provenance="operator",
        scope_version="1",
        timestamp="2026-09-23T00:00:00Z",
    )
    allowed, reason = solidity_live_testing_permitted(web)
    assert allowed is False
    assert "smart contract" in reason
    unnamed = solidity_live_testing_permitted(None)
    assert unnamed[0] is False
    contract = TargetManifest(
        program_id="program",
        asset_id="0xabc",
        target_type="smart_contract",
        language="solidity",
        source="git@example",
        commit="abc",
        snapshot="snap",
        allowed_modes=("lab",),
        active_testing=False,
        operator_approved=False,
        provenance="operator",
        scope_version="1",
        timestamp="2026-09-23T00:00:00Z",
    )
    assert solidity_live_testing_permitted(contract)[0] is False
    ready = TargetManifest(
        program_id="program",
        asset_id="0xabc",
        target_type="smart_contract",
        language="solidity",
        allowed_modes=("live",),
        active_testing=True,
        operator_approved=True,
        provenance="operator",
        scope_version="2",
        timestamp="2026-09-23T00:00:00Z",
    )
    assert solidity_live_testing_permitted(ready)[0] is True


@pytest.mark.asyncio
async def test_replay_is_exact_or_refused(tmp_path: Path, db_session: AsyncSession) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "thing.py").write_text("x = 1\n", encoding="utf-8")
    session = ResearchSession(
        project_id="proj",
        target="http://127.0.0.1/health",
        mode=ResearchMode.LAB,
        engine=SecurityTestEngine.lab(
            "lab", hosts=("127.0.0.1",), allow_active_testing=True, limits=SafetyLimits.lab()
        ),
        provider=MockLLMProvider(),
    )
    session.hypotheses.append(
        ResearchHypothesis(
            id="hyp-p35",
            title="boundary",
            vulnerability_class="logic",
            target="src/thing.py",
            reason="check",
        )
    )
    code = "def test_boundary():\n    assert True\n"
    candidate = ExploratoryTestCandidate(
        target="src/thing.py",
        language="python",
        framework="pytest",
        target_symbol="Thing.method",
        hypothesis_id="hyp-p35",
        test_code=code,
        rationale="boundary",
        expected_behavior="stays true",
        oracle="assertion",
        project_id="proj",
        session_id=session.id,
        target_file="src/thing.py",
    )
    context = ExploratoryContext(
        session_id=session.id,
        project_id="proj",
        mode="lab",
        hypotheses={"hyp-p35": "src/thing.py"},
        repo_path=str(repo),
        commit_sha="abc",
        snapshot_hash="snap",
        graph=session.graph,
    )
    engine = ExploratoryEngine(
        policy=ExploratoryTestPolicy(enabled=True),
        executor=RecordingExecutor(),
        docker_available=True,
        pytest_ready=True,
        records=session.exploratory_attempts,
    )
    first = await engine.run(candidate, context)
    assert first.executed is True
    repo_store = SecurityAgentRepository(db_session)
    await repo_store.save_session(session)
    await db_session.commit()
    db_session.expire_all()
    release_exploratory_engine(session.id)
    restored_agent = await repo_store.reconstruct(session.id)
    assert restored_agent is not None
    restored = ExploratoryEngine(
        policy=ExploratoryTestPolicy(enabled=True),
        executor=RecordingExecutor(),
        docker_available=True,
        pytest_ready=True,
        records=restored_agent.session.exploratory_attempts,
    )
    replay = restored.replay(first.attempt_id, context)
    assert replay is not None
    assert hashlib.sha256(replay.test_code.encode("utf-8")).hexdigest() == first.test_hash
    duplicate = await restored.run(candidate, context)
    assert duplicate.classification == ExploratoryClassification.DUPLICATE.value
    secret = 'def test_secret():\n    note = "api_key=supersecret"\n    assert note\n'
    secret_candidate = ExploratoryTestCandidate(
        target="src/thing.py",
        language="python",
        framework="pytest",
        target_symbol="Thing.method",
        hypothesis_id="hyp-p35",
        test_code=secret,
        rationale="secret",
        expected_behavior="stays true",
        oracle="assertion",
        project_id="proj",
        session_id=session.id,
        target_file="src/thing.py",
    )
    secret_engine = ExploratoryEngine(
        policy=ExploratoryTestPolicy(enabled=True),
        executor=RecordingExecutor(),
        docker_available=True,
        pytest_ready=True,
    )
    secret_attempt = await secret_engine.run(secret_candidate, context)
    assert secret_engine.replay(secret_attempt.attempt_id, context) is None
    assert secret_engine.records[-1].replayable is False
    assert secret_engine.records[-1].test_code == ""
    assert secret_engine.records[-1].test_hash == secret_attempt.test_hash


@pytest.mark.asyncio
async def test_agent_duplicate_guard_does_not_preempt_exploratory() -> None:
    session = ResearchSession(
        project_id="lab",
        target="http://127.0.0.1/health",
        mode=ResearchMode.LAB,
        engine=SecurityTestEngine.lab(
            "lab",
            hosts=("127.0.0.1", "localhost"),
            allow_active_testing=True,
            limits=SafetyLimits.lab(),
        ),
        provider=MockLLMProvider(),
    )
    agent = SecurityResearchAgent(session)

    async def _ok(_arguments: dict[str, object]) -> dict[str, object]:
        return {"quality": "success", "executed": True, "state": "completed"}

    agent.tools.replace_executor("http_request", _ok)
    agent.tools.replace_executor("exploratory_test", _ok)
    http = ToolCallRequest(
        tool="http_request",
        arguments={"method": "GET", "url": "http://127.0.0.1/health"},
        reason="loop",
    )
    await agent.request_tool(http)
    await agent.request_tool(http)
    with pytest.raises(RestrictedActivityError, match="repeated_identical"):
        await agent.request_tool(http)
    exploratory = ToolCallRequest(
        tool="exploratory_test",
        arguments={
            "hypothesis_id": "hyp",
            "language": "python",
            "framework": "pytest",
            "target_symbol": "Thing.method",
            "test_code": "def test_ok():\n    assert True\n",
            "expected_behavior": "ok",
            "oracle": "assertion",
        },
        reason="repeat",
    )
    await agent.request_tool(exploratory)
    await agent.request_tool(exploratory)
    third = await agent.request_tool(exploratory)
    assert third.get("quality") == "success"


def test_solc_executable_is_honest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("solc") is None:
        pytest.skip("solc executable is unavailable")
    from app.parsing.engine import parse_source
    from app.parsing.solidity_project import build_compiler_project

    class _Settings:
        solidity_host_compiler = True

    monkeypatch.setattr("app.core.config.get_settings", lambda: _Settings())
    repo = tmp_path / "repo"
    shutil.copytree(_FIXTURE, repo)
    graphs = {}
    for path in sorted(repo.rglob("*.sol")):
        text = path.read_text(encoding="utf-8")
        graph = parse_source("solidity", path, text)
        public = path.resolve().relative_to(repo.resolve()).as_posix()
        graph.file_path = public
        graphs[public] = graph
    project = build_compiler_project(repo, graphs)
    assert project.status in {"AVAILABLE", "FAILED", "INCOMPLETE"}
    if project.status != "AVAILABLE":
        assert project.layouts == ()
        assert project.selectors == ()
        assert project.detail
    else:
        assert project.complete
        assert "src/Token.sol" in project.sources
        assert "src/Vault.sol" in project.sources


def test_forge_executable_is_honest(tmp_path: Path) -> None:
    if shutil.which("forge") is None:
        pytest.skip("forge executable is unavailable")
    import os
    import subprocess

    from app.execution.base import ExecutionResult
    from app.security_testing.exploratory import _classify_execution

    repo = tmp_path / "repo"
    shutil.copytree(_FIXTURE, repo)
    before = {
        path.resolve().relative_to(repo.resolve()).as_posix(): path.read_bytes()
        for path in repo.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    output = tmp_path / "output"
    test_code = (
        "pragma solidity ^0.8.20;\n"
        'import "../src/Token.sol";\n'
        "contract ExploratoryTest {\n"
        "    function test_balance() public {\n"
        "        Token token = new Token();\n"
        "        assert(token.balanceOf(address(1)) == 1);\n"
        "    }\n"
        "}\n"
    )
    project = prepare_foundry_workspace(repo, output, test_code)
    assert (project / "test" / "Exploratory.t.sol").is_file()
    try:
        completed = subprocess.run(
            [
                "forge",
                "test",
                "--root",
                str(project),
                "--match-path",
                "test/Exploratory.t.sol",
                "--offline",
                "--out",
                str(output / "forge-out"),
                "--cache-path",
                str(output / "forge-cache"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "FOUNDRY_OFFLINE": "1"},
        )
    except subprocess.TimeoutExpired as exc:
        classification = _classify_execution(ExecutionResult(-1, "", str(exc), 60, timed_out=True))
        assert classification is ExploratoryClassification.TIMEOUT
        return
    classification = _classify_execution(
        ExecutionResult(
            completed.returncode or 0,
            completed.stdout,
            completed.stderr,
            0.1,
        )
    )
    assert classification in {
        ExploratoryClassification.PASSED,
        ExploratoryClassification.FAILED,
        ExploratoryClassification.COMPILATION_FAILED,
    }
    if classification is ExploratoryClassification.COMPILATION_FAILED:
        assert completed.returncode != 0
    after = {
        path.resolve().relative_to(repo.resolve()).as_posix(): path.read_bytes()
        for path in repo.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    assert after == before
