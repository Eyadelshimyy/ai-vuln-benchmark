"""Phase 44 replay binding.

Runners in this file are simulated unless a test is marked live. A simulated
failure is not a reproduction, and a passing run is not a proof.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.domain.evidence import EvidenceKind
from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_ir import build_semantic_program
from app.parsing.solidity_replay import (
    _sanitize_tree,
    binds_replay,
    classify_output,
    fork_replay,
    lifecycle_effect,
    prepare_replay,
    promote,
    replay_evidence,
    run_replay,
    sequence_limit,
)
from app.parsing.solidity_spec import specify
from app.parsing.solidity_state_transitions import analyze_state_transitions
from app.plugins import reset_plugin_catalog
from app.security_testing.exploratory import prepare_foundry_workspace

_FAILING = """
pragma solidity ^0.8.20;
contract Vault {
    uint256 public totalSupply;
    uint256 public balances;
    function mint() public { totalSupply = totalSupply + 1; }
}
"""

_CHAIN = """
pragma solidity ^0.8.20;
contract Vault {
    uint256 public totalSupply;
    uint256 public balances;
    function a() public { uint256 seen = totalSupply; b(); }
    function b() public { c(); }
    function c() public { totalSupply = totalSupply + 1; }
}
"""


def _model(tmp_path: Path, source: str, name: str = "Vault.sol"):
    reset_syntax_registry()
    reset_plugin_catalog()
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    program = build_semantic_program(parse_source("solidity", path, source))
    return program, source, analyze_state_transitions(program)


def _accounting(model):
    return next(item for item in model.paths if item.path_id.startswith("accounting:"))


def test_sequence_bounds_only_tighten() -> None:
    assert sequence_limit(None) == 4
    assert sequence_limit(2) == 2
    assert sequence_limit(10) == 4


def test_failing_candidate_checks_the_relation_not_a_balance(tmp_path: Path) -> None:
    program, text, model = _model(tmp_path, _FAILING)
    path = _accounting(model)
    spec = specify(path, model, program)
    sequence, artifact = prepare_replay(path, spec, text)
    assert artifact.status == "encoded"
    assert sequence.completeness == "complete"
    assert sequence.actors[0].identity == "user"
    assert "owner" not in sequence.actors[0].permissions
    assert artifact.harness.count("new Vault(") == 1
    assert "target.mint()" in artifact.harness
    assert "totalSupply() != target.balances()" in artifact.harness
    assert "assertTrue(true)" not in artifact.harness
    assert binds_replay(artifact, spec, text, path, sequence)[0]
    called: list[str] = []

    def runner(harness: str) -> tuple[int, str, str]:
        called.append(harness)
        return 1, f"Suite result: FAILED\n[FAIL] {artifact.fail_token}\n", ""

    result = run_replay(path, spec, text, runner=runner)
    assert called
    assert result.execution == "simulated"
    assert result.status == "candidate_violation"
    assert result.status != "reproduced"
    assert result.bound is False
    assert replay_evidence(result).kind is EvidenceKind.TOOL_STATUS
    assert lifecycle_effect(result) == "none"


def test_safe_execution_is_not_a_proof(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 public totalSupply;
        uint256 public balances;
        function mint() public {
            totalSupply = totalSupply + 1;
            balances = balances + 1;
        }
    }
    """
    _program, _text, model = _model(tmp_path, source)
    assert not any(item.path_id.startswith("accounting:") for item in model.paths)
    failing_program, failing_text, failing_model = _model(tmp_path, _FAILING)
    path = _accounting(failing_model)
    spec = specify(path, failing_model, failing_program)
    result = run_replay(
        path,
        spec,
        failing_text,
        runner=lambda _harness: (0, "Suite result: ok", ""),
    )
    assert result.status == "executed_no_violation"
    assert result.status != "proved_safe"
    assert any("not a proof" in item for item in result.diagnostics)


def test_failures_are_not_reproductions() -> None:
    token = "bugforge-fail:abc"
    assert classify_output("Compiler run failed\n" + token, token) == "compile_failed"
    assert classify_output("Setup failed\n" + token, token) == "setup_failed"
    assert classify_output("Suite result: FAILED\n[FAIL] other", token) == "unknown"
    assert classify_output("Failed to resolve import", token) == "unavailable"
    assert classify_output("forge timed out", token) == "timeout"
    assert (
        promote(
            "candidate_violation",
            execution="simulated",
            environment="real-target",
            bound=True,
        )
        == "candidate_violation"
    )
    assert (
        promote(
            "candidate_violation",
            execution="forge",
            environment="real-target",
            bound=True,
        )
        == "reproduced"
    )
    assert (
        promote(
            "candidate_violation",
            execution="forge",
            environment="environmental/model",
            bound=True,
        )
        == "candidate_violation"
    )
    assert (
        promote(
            "executed_no_violation",
            execution="forge",
            environment="real-target",
            bound=True,
        )
        == "executed_no_violation"
    )
    assert promote("proved_safe", execution="forge", environment="real-target", bound=True) == (
        "unknown"
    )
    assert promote("setup_failed", execution="forge", environment="real-target", bound=True) == (
        "setup_failed"
    )


def test_multi_transaction_state_is_not_reset(tmp_path: Path) -> None:
    program, text, model = _model(tmp_path, _CHAIN)
    path = next(
        item
        for item in model.paths
        if item.path_id.startswith("chain:")
        and item.function_ids[0].startswith("Vault.a")
        and item.function_ids[-1].startswith("Vault.c")
        and item.relationship == "candidate_security_path"
    )
    spec = specify(path, model, program)
    sequence, artifact = prepare_replay(path, spec, text)
    assert artifact.status == "encoded"
    assert [step.function for step in sequence.steps] == ["a", "b", "c"]
    assert artifact.harness.count("new Vault(") == 1
    body = artifact.harness.split("function test_", 1)[1]
    assert body.index("target.a()") < body.index("target.b()") < body.index("target.c()")
    assert "totalSupply() != target.balances()" in artifact.harness


def test_parameterized_calls_use_typed_literals(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 public totalSupply;
        uint256 public balances;
        function mint(uint256 amount) public { totalSupply = totalSupply + amount; }
    }
    """
    program, text, model = _model(tmp_path, source)
    path = _accounting(model)
    spec = specify(path, model, program)
    sequence, artifact = prepare_replay(path, spec, text)
    assert artifact.status == "encoded"
    assert sequence.steps[0].arguments == ("1",)
    assert "target.mint(1)" in artifact.harness
    assert "mint(amount)" not in artifact.harness


def test_unsound_argument_is_unsupported(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 public totalSupply;
        uint256 public balances;
        function mint(string memory label) public { totalSupply = totalSupply + 1; }
    }
    """
    program, text, model = _model(tmp_path, source)
    path = _accounting(model)
    spec = specify(path, model, program)
    _sequence, artifact = prepare_replay(path, spec, text)
    assert artifact.status == "unsupported"
    assert "not sound" in artifact.reason


def test_constructor_and_initializer_are_not_invented(tmp_path: Path) -> None:
    constructed = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 public totalSupply;
        uint256 public balances;
        constructor(uint256 seed) { totalSupply = seed; }
        function mint() public { totalSupply = totalSupply + 1; }
    }
    """
    program, text, model = _model(tmp_path, constructed)
    path = _accounting(model)
    spec = specify(path, model, program)
    _sequence, artifact = prepare_replay(path, spec, text)
    assert artifact.status == "unsupported"
    assert "constructor" in artifact.reason
    initialized = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 public totalSupply;
        uint256 public balances;
        address owner;
        function initialize() public { owner = msg.sender; }
        function mint() public { totalSupply = totalSupply + 1; }
    }
    """
    program, text, model = _model(tmp_path, initialized, "Init.sol")
    path = _accounting(model)
    spec = specify(path, model, program)
    _sequence, artifact = prepare_replay(path, spec, text)
    assert artifact.status == "encoded"
    assert ".initialize(" not in artifact.harness
    assert "initializer was not called" in _sequence.assumptions


def test_reentrancy_callback_is_not_two_plain_calls(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 public totalSupply;
        uint256 public balances;
        function mint() public {
            uint256 seen = totalSupply;
            msg.sender.call("");
            totalSupply = seen + 1;
        }
    }
    """
    program, text, model = _model(tmp_path, source)
    reentrancy = next(item for item in model.paths if item.path_id.startswith("reentrancy:"))
    own = specify(reentrancy, model, program)
    _sequence, plain = prepare_replay(reentrancy, own, text)
    assert plain.status == "unsupported"
    assert "receive()" not in plain.harness
    equality = next(item for item in model.invariants if item.category == "equality")
    attached = replace(reentrancy, invariant_id=equality.invariant_id)
    spec = specify(attached, model, program)
    sequence, artifact = prepare_replay(attached, spec, text)
    assert artifact.status == "encoded"
    assert sequence.actors[0].identity == "attacker"
    assert "owner" not in sequence.actors[0].permissions
    assert "receive() external payable" in artifact.harness
    test_body = artifact.harness.split("function test_", 1)[1]
    assert "target.mint()" not in test_body
    assert artifact.harness.count("target.mint()") == 2
    assert "totalSupply() != target.balances()" in artifact.harness


def test_oracle_proxy_and_fork_are_not_real_reproductions(tmp_path: Path) -> None:
    program, text, model = _model(tmp_path, _FAILING)
    base = _accounting(model)
    proxy = replace(base, path_id="changes-implementation:slot")
    proxy_spec = specify(proxy, model, program)
    _sequence, proxy_artifact = prepare_replay(proxy, proxy_spec, text)
    assert proxy_artifact.status == "unsupported"
    assert "proxy" in proxy_artifact.reason
    assert "vm.store" not in proxy_artifact.harness
    oracle = replace(base, path_id="oracle:feed")
    oracle_spec = specify(oracle, model, program)
    _sequence, oracle_artifact = prepare_replay(oracle, oracle_spec, text)
    assert oracle_artifact.status == "unsupported"
    assert oracle_artifact.environment == "environmental/model"
    assert "mock" in oracle_artifact.reason
    fork = fork_replay()
    assert fork.status == "unavailable"
    assert "public infrastructure" in fork.diagnostics[0]
    assert "http" not in " ".join(fork.diagnostics)
    assert (
        promote(
            "candidate_violation",
            execution="forge",
            environment="environmental/model",
            bound=True,
        )
        != "reproduced"
    )


def test_imports_follow_the_project_or_stay_unsupported(tmp_path: Path) -> None:
    imported = """
    pragma solidity ^0.8.20;
    import "./Lib.sol";
    contract Vault {
        uint256 public totalSupply;
        uint256 public balances;
        function mint() public { totalSupply = totalSupply + 1; }
    }
    """
    program, text, model = _model(tmp_path, imported)
    path = _accounting(model)
    spec = specify(path, model, program)
    _sequence, artifact = prepare_replay(path, spec, text)
    assert artifact.status == "unsupported"
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "foundry.toml").write_text(
        '[profile.default]\nsrc = "src"\nsolc = "0.8.20"\netherscan_api_key = "secret"\n',
        encoding="utf-8",
    )
    (root / "src" / "Lib.sol").write_text("contract Lib {}\n", encoding="utf-8")
    vault = root / "src" / "Vault.sol"
    vault.write_text(imported, encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    program, text, model = _model(tmp_path, imported, "proj/src/Vault.sol")
    path = _accounting(model)
    spec = specify(path, model, program)
    sequence, artifact = prepare_replay(path, spec, text)
    assert artifact.status == "encoded"
    assert artifact.mode == "project"
    assert 'import {Vault} from "../src/Vault.sol";' in artifact.harness
    assert "contract Lib" not in artifact.harness
    assert binds_replay(artifact, spec, text, path, sequence)[0]
    before = vault.read_text(encoding="utf-8")
    output = tmp_path / "copy"
    project = prepare_foundry_workspace(root, output, "// removed\n")
    _sanitize_tree(project)
    assert vault.read_text(encoding="utf-8") == before
    assert "etherscan_api_key" not in (project / "foundry.toml").read_text(encoding="utf-8")
    assert "etherscan_api_key" in (root / "foundry.toml").read_text(encoding="utf-8")
    assert not (project / ".env").exists()
    assert (root / ".env").is_file()


def test_changed_artifacts_do_not_bind(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "foundry.toml").write_text('[profile.default]\nsrc = "src"\n', encoding="utf-8")
    vault = root / "src" / "Vault.sol"
    vault.write_text(_FAILING, encoding="utf-8")
    program, text, model = _model(tmp_path, _FAILING, "proj/src/Vault.sol")
    path = _accounting(model)
    spec = specify(path, model, program)
    sequence, artifact = prepare_replay(path, spec, text)
    assert binds_replay(artifact, spec, text, path, sequence)[0]
    assert binds_replay(artifact, spec, text, replace(path, path_id="other"), sequence)[0] is False
    assert binds_replay(artifact, spec, text + "\n", path, sequence)[0] is False
    changed = replace(artifact, harness=artifact.harness + "\n")
    assert binds_replay(changed, spec, text, path, sequence)[0] is False
    moved = replace(sequence, sequence_id="0" * 64)
    assert binds_replay(artifact, spec, text, path, moved)[1] == "sequence mismatch"
    toml = root / "foundry.toml"
    toml.write_text(toml.read_text(encoding="utf-8") + "optimizer = false\n", encoding="utf-8")
    assert binds_replay(artifact, spec, text, path, sequence)[1] == "dependency mismatch"
    spec_again = specify(path, model, program)
    assert spec_again.compiler_config != spec.compiler_config
    assert binds_replay(artifact, spec_again, text, path, sequence)[0] is False


def test_unsupported_plan_does_not_call_the_runner(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 public totalSupply;
        uint256 public balances;
        constructor(address owner) {}
        function mint() public { totalSupply = totalSupply + 1; }
    }
    """
    program, text, model = _model(tmp_path, source)
    path = _accounting(model)
    spec = specify(path, model, program)
    called: list[str] = []
    result = run_replay(
        path,
        spec,
        text,
        runner=lambda harness: called.append(harness) or (1, "Suite result: FAILED", ""),
    )
    assert called == []
    assert result.status == "unsupported"
    assert result.execution == "not_attempted"


def test_live_forge_project_replay(tmp_path: Path) -> None:
    """Host forge is not the sandbox. A missing image stays unavailable."""
    program, text, model = _model(tmp_path, _FAILING)
    path = _accounting(model)
    spec = specify(path, model, program)
    result = run_replay(path, spec, text)
    assert result.execution == "not_attempted"
    assert result.status == "unavailable"
    assert "host forge is not started" in " ".join(result.diagnostics)
