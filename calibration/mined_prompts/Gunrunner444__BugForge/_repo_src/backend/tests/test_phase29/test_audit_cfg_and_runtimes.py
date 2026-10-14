"""Phase 29: Phase 28 audit fixes, CFG semantics, and runtime bridges."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.discovery.external import (
    foundry_invocation,
    normalize_slither,
    parse_echidna_output,
    parse_foundry_output,
    parse_medusa_output,
)
from app.adapters.discovery.runtimes import GoTestEngine, cargo_test_command, go_test_command
from app.adapters.languages.capabilities import capability_matrix, promotion_stage
from app.discovery.capabilities import ResultStatus
from app.discovery.engine import AnalysisRequest
from app.discovery.process import ProcessResult, run_command
from app.discovery.results import DynamicResult
from app.discovery.scheduler import DiscoveryScheduler
from app.domain.evidence import EvidenceKind
from app.parsing.engine import parse_source
from app.parsing.keccak import function_selector
from app.parsing.solidity_cfg import (
    auth_dominates_sensitive,
    build_function_cfg,
    write_reachable_after,
)
from app.parsing.solidity_types import is_dynamic_type
from app.plugins import get_plugin_catalog
from app.security.engine import SecurityAnalysisEngine


def _ids(tmp_path: Path, source: str) -> set[str]:
    path = tmp_path / "C.sol"
    path.write_text(source, encoding="utf-8")
    scan = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    return {item.rule_id for item in scan.observations}


def test_abi_dynamic_classification_is_recursive() -> None:
    assert is_dynamic_type("uint256") is False
    assert is_dynamic_type("address") is False
    assert is_dynamic_type("bytes32") is False
    assert is_dynamic_type("bytes") is True
    assert is_dynamic_type("string") is True
    assert is_dynamic_type("uint256[]") is True
    assert is_dynamic_type("uint256[4]") is False
    assert is_dynamic_type("bytes32[4]") is False
    assert is_dynamic_type("(uint256,address)") is False
    assert is_dynamic_type("(string,address)") is True
    assert is_dynamic_type("(uint256[],address)") is True
    assert is_dynamic_type("(uint256,address)[4]") is False
    assert is_dynamic_type("((uint256,address),bytes32)") is False
    assert is_dynamic_type("Mystery") is False


def test_passing_foundry_suite_is_not_new_coverage() -> None:
    parsed = parse_foundry_output("Suite result: ok. 1 passed; 0 failed")
    assert parsed["assertion"] == ""
    coverage = parsed["coverage"]
    assert isinstance(coverage, dict)
    assert "new" not in coverage
    assert coverage.get("new_coverage") is None
    assert coverage["coverage_available"] == "false"


def test_coverage_delta_needs_a_real_baseline() -> None:
    first = parse_medusa_output("coverage: 52%\n")
    assert first["coverage"]["coverage_percent"] == "52"
    assert "new_coverage" not in first["coverage"]
    same = parse_medusa_output("coverage: 52%\n", baseline="52")
    assert same["coverage"]["new_coverage"] == "false"
    higher = parse_medusa_output("coverage: 57%\n", baseline="52")
    assert higher["coverage"]["new_coverage"] == "true"
    assert float(higher["coverage"]["coverage_delta"]) == 5


def test_scheduler_stagnation_uses_observed_percents(tmp_path: Path) -> None:
    scheduler = DiscoveryScheduler(engines=(), max_rounds=2)
    request = AnalysisRequest(tmp_path, "solidity", target="withdraw")

    def note(percent: str) -> None:
        scheduler.note_result(
            DynamicResult(
                engine="medusa",
                language="solidity",
                target="withdraw",
                status=ResultStatus.EXECUTED,
                executed=True,
                coverage={
                    "percent": percent,
                    "coverage_available": "true",
                    "coverage_source": "medusa",
                },
            ),
            request,
        )

    note("52")
    assert scheduler.feedback.new_coverage is False
    assert scheduler.feedback.stagnating is False
    note("52")
    assert scheduler.feedback.stagnating is True
    assert scheduler.feedback.new_coverage is False
    note("57")
    assert scheduler.feedback.new_coverage is True
    assert scheduler.feedback.stagnating is False


def test_process_states_are_distinct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = run_command(["bugforge-missing-tool"], cwd=tmp_path)
    assert missing == ProcessResult(
        False, False, None, "", "bugforge-missing-tool is not installed", False
    )

    monkeypatch.setattr("app.discovery.process.tool_path", lambda _name: "/usr/bin/true")

    def explode(*_args: object, **_kwargs: object) -> object:
        raise OSError("exec format error")

    monkeypatch.setattr("app.discovery.process.subprocess.run", explode)
    failed = run_command(["true"], cwd=tmp_path)
    assert failed.available is True
    assert failed.started is False
    assert failed.timed_out is False

    class _Done:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr("app.discovery.process.subprocess.run", lambda *_a, **_k: _Done())
    ok = run_command(["true"], cwd=tmp_path)
    assert ok.started is True and ok.return_code == 0 and ok.timed_out is False

    class _Bad:
        returncode = 2
        stdout = "nope"
        stderr = ""

    monkeypatch.setattr("app.discovery.process.subprocess.run", lambda *_a, **_k: _Bad())
    nonzero = run_command(["true"], cwd=tmp_path)
    assert nonzero.started is True and nonzero.return_code == 2

    def timeout(*_args: object, **_kwargs: object) -> object:
        import subprocess

        raise subprocess.TimeoutExpired(cmd=["true"], timeout=1)

    monkeypatch.setattr("app.discovery.process.subprocess.run", timeout)
    timed = run_command(["true"], cwd=tmp_path)
    assert timed.started is True and timed.timed_out is True and timed.return_code is None


def test_dynamic_evidence_keeps_oracle_kind() -> None:
    assertion = DynamicResult(
        engine="foundry",
        language="solidity",
        target="test_withdraw",
        status=ResultStatus.INTERESTING,
        executed=True,
        assertion="test_withdraw failed",
        oracle_kind="assertion",
    )
    evidence = assertion.to_evidence()
    assert evidence.kind is EvidenceKind.TEST_FAILURE
    assert evidence.metadata["verified"] == "false"
    assert evidence.metadata["oracle_kind"] == "assertion"
    symbolic = DynamicResult(
        engine="halmos",
        language="solidity",
        target="test_withdraw",
        status=ResultStatus.INTERESTING,
        executed=True,
        minimized_input="amount=1",
        oracle_kind="symbolic",
        metadata={"seed_source": "symbolic"},
    )
    assert symbolic.to_evidence().kind is EvidenceKind.FUZZING
    crash = DynamicResult(
        engine="native-fuzz",
        language="c",
        target="parse",
        status=ResultStatus.INTERESTING,
        executed=True,
        crash="segfault",
        oracle_kind="crash",
    )
    assert crash.to_evidence().kind is EvidenceKind.TEST_FAILURE
    sanitizer = DynamicResult(
        engine="sanitizer",
        language="c",
        target="parse",
        status=ResultStatus.INTERESTING,
        executed=True,
        sanitizer="heap-use-after-free",
        oracle_kind="sanitizer",
    )
    assert sanitizer.to_evidence().kind is EvidenceKind.FUZZING


def test_foundry_commands_match_real_tests(tmp_path: Path) -> None:
    request = AnalysisRequest(
        tmp_path, "solidity", target="Vault", function="withdraw", extra={"mode": "fuzz"}
    )
    argv, label = foundry_invocation("fuzz", request, tests=("test_withdraw",))
    assert argv == ["forge", "test"]
    assert label == "test"
    fuzz_request = AnalysisRequest(
        tmp_path, "solidity", target="Vault", function="testFuzz_withdraw", extra={"mode": "fuzz"}
    )
    fuzz_argv, fuzz_label = foundry_invocation("fuzz", fuzz_request, tests=("testFuzz_withdraw",))
    assert fuzz_label == "fuzz"
    assert "--match-test" in fuzz_argv and "testFuzz_withdraw" in fuzz_argv
    assert "--fuzz-runs" in fuzz_argv
    invariant_argv, invariant_label = foundry_invocation(
        "invariant",
        AnalysisRequest(tmp_path, "solidity", function="invariant_supply"),
        tests=("invariant_supply",),
    )
    assert invariant_label == "invariant"
    assert "invariant_supply" in invariant_argv
    build_argv, build_label = foundry_invocation("build", request)
    assert build_argv == ["forge", "build"] and build_label == "build"


def test_echidna_json_does_not_treat_exit_as_proof() -> None:
    passing = parse_echidna_output(
        '{"tests": [{"name": "echidna_supply", "status": "passing"}], "corpus": "4"}'
    )
    assert passing["assertion"] == ""
    assert passing["campaign_success"] == "true"
    falsified = parse_echidna_output(
        '{"tests": [{"name": "echidna_supply", "status": "falsified", "transactions": ["withdraw(1)"]}]}'
    )
    assert "falsified" in str(falsified["assertion"])
    assert "withdraw(1)" in str(falsified["sequence"])
    assert falsified["campaign_success"] == "false"


def test_slither_keeps_every_source_element() -> None:
    payload = """
    {"results": {"detectors": [{"check": "reentrancy-eth", "impact": "High",
      "elements": [
        {"name": "withdraw", "source_mapping": {"filename": "A.sol", "lines": [4]}},
        {"name": "balances", "source_mapping": {"filename": "A.sol", "lines": [9]}}
      ]}]}}
    """
    findings = normalize_slither(payload)
    assert [item.line for item in findings] == [4, 9]
    assert findings[0].function == "withdraw"
    assert findings[1].function == "balances"


def test_selectors_ignore_returns_and_locations(tmp_path: Path) -> None:
    assert function_selector("foo()") == "0xc2985578"
    assert function_selector("withdraw(uint256)") == "0x2e1a7d4d"
    assert function_selector("transfer(address,uint256)") == "0xa9059cbb"
    source = """
    pragma solidity ^0.8.20;
    struct Pair { uint256 a; address b; }
    contract C {
        function foo() external pure returns (uint) { return 1; }
        function foo(uint x) external pure returns (uint) { return x; }
        function use(Pair memory item) external pure {}
        function mystery(Unknown memory item) external pure {}
    }
    """
    graph = parse_source("solidity", tmp_path / "C.sol", source)
    functions = [event for event in graph.events if event.kind == "sol_function"]

    def fields(event: object) -> dict[str, str]:
        extra = str(getattr(event, "extra", ""))
        return {
            part.split("=", 1)[0]: part.split("=", 1)[1] for part in extra.split("|") if "=" in part
        }

    by_name = {
        (fields(event)["function"], fields(event)["params"]): fields(event) for event in functions
    }
    assert by_name[("foo", "")]["selector"] == "0xc2985578"
    assert by_name[("foo", "uint256")]["selector"] == function_selector("foo(uint256)")
    assert by_name[("use", "(uint256,address)")]["selector_status"] == "canonical"
    assert by_name[("mystery", "")]["selector_status"] == "unresolved"
    assert by_name[("mystery", "")]["selector"] == ""


def test_cfg_authorization_and_reentrancy_paths(tmp_path: Path) -> None:
    guarded = """
    function setOwner(address next) external {
        require(msg.sender == owner);
        owner = next;
    }
    """
    late = """
    function setOwner(address next) external {
        owner = next;
        require(msg.sender == owner);
    }
    """
    assert auth_dominates_sensitive(guarded) is True
    assert auth_dominates_sensitive(late) is False
    branched = """
    function poke(address t) external {
        if (t == address(0)) { t.call(""); }
        else { counter = 2; }
    }
    """
    assert write_reachable_after(branched, 't.call("")', "counter = 2") is False
    cfg = build_function_cfg(guarded)
    assert cfg.known is True and len(cfg.nodes) > 2
    late_ids = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract P {
            address owner;
            function setOwner(address next) external {
                owner = next;
                require(msg.sender == owner);
            }
        }
        """,
    )
    assert "sol.missing_authorization" in late_ids


def test_adversarial_names_and_comments_stay_quiet(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Noise {
        address implementation;
        address oracle;
        uint random;
        function foo() external pure returns (uint) {
            // delegatecall vulnerability
            uint256 x = 1;
            string memory note = "abi.encodePacked(string,string)";
            return x;
        }
        function withdraw() external view returns (uint) { return random; }
        function sum(uint[4] memory items) external pure returns (uint) {
            uint total;
            for (uint i; i < items.length; i++) { total += items[i]; }
            return total;
        }
        function active(uint end) external view {
            require(block.timestamp < end);
        }
    }
    """
    rules = _ids(tmp_path, source)
    assert "sol.arbitrary_delegatecall" not in rules
    assert "sol.encode_packed" not in rules
    assert "sol.insecure_randomness" not in rules
    assert "sol.stale_oracle" not in rules
    assert "sol.storage_collision" not in rules
    assert "sol.missing_authorization" not in rules
    assert "sol.unbounded_loop" not in rules
    assert "sol.unbounded_state_loop" not in rules


def test_state_growth_loop_is_not_called_an_external_loop(tmp_path: Path) -> None:
    rules = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract L {
            uint[] items;
            function grow(uint[] calldata extra) external {
                for (uint i; i < extra.length; i++) { items[i] = extra[i]; }
            }
        }
        """,
    )
    assert "sol.unbounded_loop" not in rules
    assert "sol.unbounded_state_loop" in rules


def test_vyper_stays_detection_only() -> None:
    catalog = get_plugin_catalog()
    vyper = catalog.languages.get("vyper")
    matrix = capability_matrix("vyper")
    assert promotion_stage("vyper") == "DETECTION_ONLY"
    assert matrix["parse"] == "UNSUPPORTED"
    assert matrix["security_rules"] == "UNSUPPORTED"
    assert matrix["external_static"] == "UNAVAILABLE_AT_RUNTIME"
    assert vyper.language_id == "vyper"
    go = capability_matrix("go")
    assert go["runtime_testing"] in {"LIMITED", "UNAVAILABLE_AT_RUNTIME"}
    assert go["fuzzing"] != "YES"


def test_go_and_cargo_commands_do_not_invent_fuzz_targets() -> None:
    request = AnalysisRequest(
        Path("."), "go", target="pkg", function="FuzzParse", extra={"mode": "fuzz"}
    )
    assert go_test_command(request) == (
        ["go", "test", "-fuzz", "^FuzzParse$", "-fuzztime", "1s"],
        "fuzz",
    )
    plain = AnalysisRequest(Path("."), "go", target="pkg", extra={"mode": "fuzz"})
    assert go_test_command(plain) is None
    cargo = cargo_test_command(AnalysisRequest(Path("."), "rust", function="parse"))
    assert cargo == (["cargo", "test", "parse"], "test")
    engine = GoTestEngine()
    missing = engine.start_campaign(AnalysisRequest(Path("."), "go", extra={"mode": "fuzz"}))
    assert missing.status is ResultStatus.NOT_IMPLEMENTED
    assert missing.executed is False
