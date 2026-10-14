"""Phase 42 verification bridge. A tool result is not a finding.

``encoded=True`` is not authority. These tests cover the bridge queue and the
unbound parsers. Bound harness behavior lives in the Phase 43 tests.
"""

from __future__ import annotations

from pathlib import Path

from app.domain.evidence import EvidenceKind
from app.domain.lifecycle_policy import independent_verification_items, positive_reproduction
from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_ir import build_semantic_program
from app.parsing.solidity_state_transitions import analyze_state_transitions
from app.parsing.solidity_verification import (
    _GENERATED,
    VerificationRequest,
    bridge,
    evidence_for,
    forge_harness,
    lifecycle_effect,
    parse_forge_output,
    parse_smt_output,
    run_forge,
    run_smt,
    smt_harness,
    tool_availability,
)
from app.plugins import reset_plugin_catalog


def _request(**overrides: object) -> VerificationRequest:
    base = dict(
        verification_id="ver:1",
        property_id="inv:asset",
        invariant_id="inv:asset",
        path_id="path:1",
        contract="Vault",
        function_id="Vault.mint:1",
        state_variables=("totalSupply",),
        preconditions=(),
        postcondition="supply increases with assets",
        path_conditions=(),
        assumptions=(),
        compiler_config="unspecified",
        encoded=False,
    )
    base.update(overrides)
    return VerificationRequest(**base)  # type: ignore[arg-type]


def _program(tmp_path: Path):
    reset_syntax_registry()
    reset_plugin_catalog()
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        function withdraw() external {
            uint256 owed = totalAssets;
            msg.sender.call("");
            totalAssets = owed;
        }
    }
    """
    return build_semantic_program(parse_source("solidity", tmp_path / "C.sol", source))


def test_bridge_does_not_verify_a_candidate(tmp_path: Path) -> None:
    program = _program(tmp_path)
    model = analyze_state_transitions(program)
    result = bridge(program, model)
    assert result.requests
    assert result.specifications
    assert len(result.specifications) == len(result.requests)
    assert {item.status for item in result.results} == {"not_requested"}
    assert all(item.encoded is False for item in result.requests)
    assert all(lifecycle_effect(item) == "none" for item in result.results)
    assert "solc" in result.tools
    assert "forge" in result.tools
    assert "semgrep" in result.tools


def test_smt_parser_does_not_treat_silence_or_success_as_proof() -> None:
    assert parse_smt_output("", "", 0) == "unknown"
    assert parse_smt_output("Warning: CHC: Assertion violation happens here", "", 0) == "unknown"
    assert parse_smt_output("SMTChecker timed out", "", 1) == "timeout"
    assert parse_smt_output("feature not yet implemented", "", 1) == "unsupported"
    assert parse_smt_output("Error: syntax", "", 1) == "failed"
    assert parse_smt_output("property proved", "", 0) == "unknown"


def test_encoded_flag_does_not_authorize_a_harness() -> None:
    request = _request(encoded=True)
    result = run_smt(
        request,
        "contract C {}",
        harness="// not generated\n",
        runner=lambda harness: (0, "property proved", ""),
    )
    assert result.status == "unsupported"
    assert result.bound is False
    assert evidence_for(result).kind is EvidenceKind.TOOL_STATUS
    assert not positive_reproduction(evidence_for(result))


def test_arbitrary_counterexample_text_is_not_authoritative() -> None:
    request = _request(encoded=True)
    harness = "// BUGFORGE GENERATED\ncontract C { function f() external pure { assert(false); } }"
    result = run_smt(
        request,
        "",
        harness=harness,
        runner=lambda _harness: (1, "Assertion violation\nCounterexample: sender = 0x1", ""),
    )
    assert result.status == "unsupported"
    assert result.counterexample is None
    assert result.bound is False
    evidence = evidence_for(result)
    assert evidence.kind is EvidenceKind.TOOL_STATUS
    assert not positive_reproduction(evidence)
    assert independent_verification_items((evidence,)) == ()
    assert lifecycle_effect(result) == "none"


def test_forge_failure_text_without_a_specification_is_not_reproduced() -> None:
    failing = "Suite result: FAILED\n[FAIL] test_candidate_sequence()\ncaller=Vault.withdraw"
    encoded = run_forge(_request(encoded=True), runner=lambda _harness: (1, failing, ""))
    assert encoded.status == "unsupported"
    assert encoded.bound is False
    assert encoded.counterexample is None
    assert evidence_for(encoded).kind is EvidenceKind.TOOL_STATUS
    assert not positive_reproduction(evidence_for(encoded))
    unbound = run_forge(_request(encoded=False), runner=lambda _harness: (1, failing, ""))
    assert unbound.status == "unsupported"
    passing = run_forge(_request(encoded=True), runner=lambda _harness: (0, "Suite result: ok", ""))
    assert passing.status == "unsupported"


def test_missing_tools_are_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("app.parsing.solidity_verification.shutil.which", lambda _name: None)
    assert run_smt(_request(), "contract C {}").status == "unavailable"
    assert run_forge(_request()).status == "unavailable"
    assert tool_availability()["solc"] == ""
    assert tool_availability()["semgrep"] == ""


def test_unencoded_harnesses_have_no_trivial_assertion() -> None:
    request = _request()
    solidity = smt_harness(request, "contract Vault {}")
    foundry = forge_harness(request, "contract Vault {}")
    assert _GENERATED in solidity
    assert _GENERATED in foundry
    assert "contract Vault {}" in solidity
    assert "assert(true)" not in solidity
    assert "assertTrue(true)" not in foundry
    assert "forge-std" not in foundry


def test_forge_parser_keeps_timeout_and_failure_distinct() -> None:
    assert parse_forge_output("") == "unknown"
    assert parse_forge_output("timed out") == "timeout"
    assert parse_forge_output("Compiler run failed") == "failed"
    assert parse_forge_output("Suite result: ok") == "unknown"
    assert parse_forge_output("Suite result: FAILED") == "unknown"
