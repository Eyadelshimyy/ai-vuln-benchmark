"""Phase 41 state transitions and Phase 40 property regressions."""

from __future__ import annotations

from pathlib import Path

from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_cross_dataflow import (
    analyze_asset_flow,
    analyze_reentrancy,
    graph_edges_are_connected,
)
from app.parsing.solidity_dataflow import analyze_dataflow
from app.parsing.solidity_ir import build_semantic_program
from app.parsing.solidity_state_transitions import find_candidate_exploit_paths
from app.plugins import reset_plugin_catalog

FALSE_NEGATIVES = (
    {
        "phase": "41",
        "vulnerability_class": "reentrancy",
        "fixture": "read-local-call-write",
        "expected_semantic_path": "state read -> local -> call -> state write",
        "previous_miss": "same variable name before and after a call",
        "corrected_behavior": "requires a dependency edge",
        "reason": "a shared name is not a data dependency",
    },
    {
        "phase": "41",
        "vulnerability_class": "reentrancy",
        "fixture": "same-path-unrelated-constant",
        "previous_behavior": "a nearby state read was treated as influencing a constant write",
        "corrected_behavior": "no state-read-influences-write edge without a value chain",
        "expected_semantic_relation": "unknown",
        "reason": "source distance is not a semantic dependency",
    },
    {
        "phase": "41",
        "vulnerability_class": "accounting",
        "fixture": "complete-supply-increase-without-assets",
        "previous_behavior": "a complete model returned unknown and only an incomplete model could be potential",
        "corrected_behavior": "a demonstrated supply increase without asset inflow is potential",
        "expected_semantic_relation": "supply-increased-without-asset-inflow",
        "reason": "ordinary completeness must not hide a modeled relation",
    },
    {
        "phase": "41",
        "vulnerability_class": "delegatecall",
        "fixture": "two-delegatecalls",
        "previous_behavior": "a function-wide query could describe the wrong call",
        "corrected_behavior": "each call id is separate and a missing call id stays unknown",
        "expected_semantic_relation": "call-id",
        "reason": "two delegatecalls in one function are different operations",
    },
    {
        "phase": "41",
        "vulnerability_class": "reentrancy",
        "fixture": "read-branch-call-write",
        "previous_behavior": "an assignment nested in a branch was classified as a read",
        "corrected_behavior": "the nested assignment stays a distinct write and can reach from the earlier read",
        "expected_semantic_relation": "state read -> branch -> external call -> dependent write",
        "reason": "control nesting is not a reason to drop a write",
    },
    {
        "phase": "41",
        "vulnerability_class": "call-resolution",
        "fixture": "external-target-same-name",
        "previous_behavior": "token.other() resolved to the caller's other()",
        "corrected_behavior": "an unresolved external target stays unknown",
        "expected_semantic_relation": "resolution unknown",
        "reason": "the same function name in the caller is not the callee",
    },
    {
        "phase": "41",
        "vulnerability_class": "accounting",
        "fixture": "msg-value-comment",
        "previous_behavior": "msg.value inside a comment was treated as asset inflow",
        "corrected_behavior": "a comment does not establish inflow, so the supply increase stays potential",
        "expected_semantic_relation": "supply-increased-without-asset-inflow",
        "reason": "comment text is not an executed value transfer",
    },
    {
        "phase": "41",
        "vulnerability_class": "invariant",
        "fixture": "caller-preserved-while-callee-writes",
        "previous_behavior": "a caller that only invokes a writing callee was marked preserved",
        "corrected_behavior": "the caller check is unknown unless resolved callees avoid the variables",
        "expected_semantic_relation": "unknown",
        "reason": "local non-interference does not survive an internal write",
    },
    {
        "phase": "41",
        "vulnerability_class": "identity",
        "fixture": "two-sources-same-contract-function-line",
        "corrected_behavior": "the shared identity is namespaced and the summaries are not merged",
        "reason": "contract name, function name, and line are not unique across sources",
    },
    {
        "phase": "41",
        "vulnerability_class": "invariant",
        "fixture": "unrelated-contracts-same-variable-names",
        "corrected_behavior": "each declaring contract keeps its own candidate invariant",
        "reason": "a shared variable name is not a cross-contract relation",
    },
    {
        "phase": "41",
        "vulnerability_class": "path",
        "fixture": "call-chain-without-shared-state",
        "corrected_behavior": "the chain status is reachable and records a reachable relationship",
        "reason": "reachability of a candidate function is not influence",
    },
    {
        "phase": "41",
        "vulnerability_class": "path",
        "fixture": "call-chain-with-shared-declaration",
        "corrected_behavior": "the chain is a candidate only when an earlier transition uses the same declaration",
        "reason": "a semantic connection is a shared declaration, not a call edge",
    },
    {
        "phase": "42",
        "vulnerability_class": "verification-binding",
        "fixture": "encoded-flag-and-arbitrary-harness",
        "corrected_behavior": "the result stays non-authoritative",
        "reason": "a caller flag cannot authorize a harness BugForge did not bind",
    },
    {
        "phase": "43",
        "vulnerability_class": "verification-binding",
        "fixture": "modified-or-stale-harness",
        "corrected_behavior": "a digest or specification mismatch stays unknown",
        "reason": "a generated banner is not a binding",
    },
    {
        "phase": "43",
        "vulnerability_class": "smt",
        "fixture": "unrelated-proved-or-counterexample",
        "corrected_behavior": "diagnostics that do not name the artifact stay unknown",
        "reason": "the words proved and counterexample are not the candidate property",
    },
    {
        "phase": "43",
        "vulnerability_class": "foundry",
        "fixture": "fail-without-specification-token",
        "corrected_behavior": "a failing suite without the fail token is not reproduced",
        "reason": "a runner string is not execution of the candidate",
    },
    {
        "phase": "43",
        "vulnerability_class": "encoding",
        "fixture": "unsupported-property",
        "corrected_behavior": "the harness contains no assertion and the status is unsupported",
        "reason": "a comment is not a predicate",
    },
    {
        "phase": "43",
        "vulnerability_class": "smt",
        "fixture": "external-function-internal-call",
        "corrected_behavior": "an external function is not encoded as an internal call",
        "reason": "mint() is not a valid call to an external function",
    },
    {
        "phase": "43",
        "vulnerability_class": "smt",
        "fixture": "preexisting-assert",
        "corrected_behavior": "binding uses the generated assertion marker, not the first assert",
        "reason": "a source assertion is not the candidate property",
    },
    {
        "phase": "44",
        "vulnerability_class": "replay",
        "fixture": "compile-or-setup-failure",
        "corrected_behavior": "compile and setup failures stay non-reproductions",
        "reason": "a broken harness is not the candidate property failing",
    },
    {
        "phase": "44",
        "vulnerability_class": "replay",
        "fixture": "simulated-fail-token",
        "corrected_behavior": "a simulated runner cannot return reproduced",
        "reason": "pasted tool text is not execution",
    },
)


def _program(tmp_path: Path, source: str):
    reset_syntax_registry()
    reset_plugin_catalog()
    return build_semantic_program(parse_source("solidity", tmp_path / "C.sol", source))


def test_reentrancy_requires_a_dependency_not_a_shared_name(tmp_path: Path) -> None:
    dependent = """
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
    independent = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        uint256 other;
        function withdraw() external {
            uint256 owed = totalAssets;
            msg.sender.call("");
            other = 1;
        }
    }
    """
    assert analyze_reentrancy(_program(tmp_path, dependent), "withdraw").status == "potential"
    assert analyze_reentrancy(_program(tmp_path, independent), "withdraw").status == "unknown"


def test_state_load_reaches_a_later_store(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint256) balances;
        uint256 totalAssets;
        function run(address user) external {
            uint256 amount = balances[user];
            uint256 fee = amount / 10;
            uint256 payout = amount - fee;
            totalAssets = payout;
        }
    }
    """
    flow = analyze_dataflow(_program(tmp_path, source))
    assert graph_edges_are_connected(flow)
    summary = flow._summary("run")
    assert summary is not None
    assert any(edge.kind == "local-dependency" for edge in summary.edges)
    assert any(edge.kind == "local-to-state" for edge in summary.edges)


def test_asset_flow_is_not_a_violation(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        function pay(address user, uint256 amount) external {
            user.call{value: amount}("");
        }
    }
    """
    result = analyze_asset_flow(_program(tmp_path, source), "pay")
    assert result.status == "unknown"


def test_candidate_path_is_not_reproduced(tmp_path: Path) -> None:
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
    paths = find_candidate_exploit_paths(_program(tmp_path, source))
    assert paths
    assert {item.status for item in paths} == {"candidate"}
    assert "reproduced" not in {item.status for item in paths}


def test_false_negative_registry() -> None:
    assert FALSE_NEGATIVES[0]["corrected_behavior"]
    for item in FALSE_NEGATIVES:
        assert item["phase"] in {"41", "42", "43", "44"}
        assert item["vulnerability_class"]
        assert item["fixture"]
        assert item["corrected_behavior"]
        assert item["reason"]
