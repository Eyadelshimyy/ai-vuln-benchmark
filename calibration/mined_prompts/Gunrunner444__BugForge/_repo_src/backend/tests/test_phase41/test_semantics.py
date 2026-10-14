"""Adversarial checks for bounded transitions. Unknown is not safe."""

from __future__ import annotations

from pathlib import Path

from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_cross_dataflow import (
    analyze_delegatecall,
    analyze_operation_authorization,
    analyze_reentrancy,
)
from app.parsing.solidity_dataflow import analyze_dataflow
from app.parsing.solidity_ir import build_semantic_program
from app.parsing.solidity_state_transitions import (
    MAX_DEPTH,
    analyze_accounting_transition,
    analyze_state_transitions,
    find_candidate_exploit_paths,
)
from app.parsing.solidity_value_flow import join_provenance
from app.plugins import reset_plugin_catalog

_FORBIDDEN = {"verified", "reproduced", "safe"}


def _program(tmp_path: Path, source: str):
    reset_syntax_registry()
    reset_plugin_catalog()
    return build_semantic_program(parse_source("solidity", tmp_path / "C.sol", source))


def test_duplicate_reads_stay_distinct_and_only_the_live_one_reaches(tmp_path: Path) -> None:
    stale = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint256) balances;
        function run(address user) external {
            uint256 a = balances[user];
            uint256 b = balances[user];
            msg.sender.call("");
            balances[user] = b;
        }
    }
    """
    fresh = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint256) balances;
        function run(address user) external {
            uint256 a = balances[user];
            msg.sender.call("");
            uint256 b = balances[user];
            balances[user] = b;
        }
    }
    """
    program = _program(tmp_path, stale)
    function = program.functions_named("run", "Vault")[0]
    reads = [item for item in function.access_sites if item.kind == "read"]
    assert len({item.operation_id for item in reads}) == len(reads) >= 2
    result = analyze_reentrancy(program, "run")
    assert result.status == "potential"
    later = max(reads, key=lambda item: item.span.start_byte)
    earlier = min(reads, key=lambda item: item.span.start_byte)
    assert later.operation_id in result.dependencies
    assert earlier.operation_id not in result.dependencies
    assert analyze_reentrancy(_program(tmp_path, fresh), "run").status == "unknown"


def test_constant_write_and_other_variable_are_not_reentrancy(tmp_path: Path) -> None:
    constant = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        function withdraw() external {
            uint256 owed = totalAssets;
            msg.sender.call("");
            totalAssets = 1;
        }
    }
    """
    other = """
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
    assert analyze_reentrancy(_program(tmp_path, constant), "withdraw").status == "unknown"
    assert analyze_reentrancy(_program(tmp_path, other), "withdraw").status == "unknown"
    flow = analyze_dataflow(_program(tmp_path, constant))
    summary = flow._summary("withdraw")
    assert summary is not None
    assert not any(edge.kind == "state-read-influences-write" for edge in summary.edges)


def test_helper_return_reaches_a_later_write_and_an_ignored_return_does_not(tmp_path: Path) -> None:
    used = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        function withdraw() external {
            uint256 owed = helper();
            msg.sender.call("");
            totalAssets = owed;
        }
        function helper() internal returns (uint256) { return totalAssets; }
    }
    """
    ignored = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        function withdraw() external {
            helper();
            msg.sender.call("");
            totalAssets = 1;
        }
        function helper() internal returns (uint256) { return totalAssets; }
    }
    """
    assert analyze_reentrancy(_program(tmp_path, used), "withdraw").status == "potential"
    assert analyze_reentrancy(_program(tmp_path, ignored), "withdraw").status == "unknown"


def test_delegatecalls_are_call_specific(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Proxy {
        address implementation;
        function run(address target, bytes calldata data) external {
            target.delegatecall(data);
            implementation.delegatecall(data);
        }
    }
    """
    program = _program(tmp_path, source)
    function = program.functions_named("run", "Proxy")[0]
    sites = [site for site in function.call_sites if site.call_type == "delegatecall"]
    assert len(sites) == 2
    results = [analyze_delegatecall(program, "run", site.call_id) for site in sites]
    assert {item.call_ids for item in results} == {(sites[0].call_id,), (sites[1].call_id,)}
    assert results[0].provenance == "attacker"
    assert results[1].provenance == "state"
    wide = analyze_delegatecall(program, "run")
    assert wide.status == "unknown"
    assert "call id is required" in wide.summary
    flow = analyze_dataflow(program)
    summary = flow._summary("run")
    assert summary is not None
    returns = [value.identity for value in summary.values if value.kind == "external-return"]
    assert len(returns) == len(set(returns)) == 2


def test_authorization_is_per_write_of_the_same_path(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        address owner;
        uint256 balances;
        function set(uint256 value) external {
            balances = 1;
            require(msg.sender == owner);
            balances = value;
        }
    }
    """
    program = _program(tmp_path, source)
    function = program.functions_named("set", "Vault")[0]
    writes = [item for item in function.access_sites if item.kind != "read"]
    assert [item.guard_status for item in writes] == ["unguarded", "guarded"]
    assert writes[1].guard_dominates == "yes"
    assert "msg.sender" in writes[1].guard_predicate
    first = analyze_operation_authorization(program, function.identity, writes[0].operation_id)
    second = analyze_operation_authorization(program, function.identity, writes[1].operation_id)
    assert first.status == "potential"
    assert second.status == "satisfied"


def test_provenance_keeps_attacker_influence() -> None:
    assert join_provenance({"trusted_constant", "calldata"}) == "calldata"
    assert join_provenance({"caller", "calldata"}) == "attacker"
    assert join_provenance({"state", "unknown"}) == "unknown"
    assert join_provenance({"state", "trusted_constant"}) == "state"


def test_calldata_and_msg_value_provenance(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        function run(uint256 amount) external payable {
            uint256 value = msg.value;
            uint256 local = amount;
            totalAssets = value + local;
        }
    }
    """
    flow = analyze_dataflow(_program(tmp_path, source))
    summary = flow._summary("run")
    assert summary is not None
    provenances = {value.provenance for value in summary.values if value.kind == "local"}
    assert "msg.value" in provenances
    assert "calldata" in provenances
    write = next(edge for edge in summary.edges if edge.kind == "local-to-state")
    assert flow.value_reaches(
        next(value.identity for value in summary.values if value.provenance == "msg.value"),
        write.sink,
    )


def test_accounting_distinguishes_observation_from_a_relation(tmp_path: Path) -> None:
    transfer = """
    pragma solidity ^0.8.20;
    contract Vault {
        function pay(address user, uint256 amount) external {
            user.call{value: amount}("");
        }
    }
    """
    paired = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        uint256 totalSupply;
        function sync(uint256 assets) external {
            totalSupply = totalSupply + assets;
            totalAssets = totalAssets + assets;
        }
    }
    """
    minted = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        uint256 totalSupply;
        function mint(uint256 shares) external {
            totalSupply = totalSupply + shares;
        }
    }
    """
    assert analyze_accounting_transition(_program(tmp_path, transfer), "pay").status == "observed"
    paired_result = analyze_accounting_transition(_program(tmp_path, paired), "sync")
    assert paired_result.status == "observed"
    assert paired_result.relation == "supply-and-assets-updated"
    minted_result = analyze_accounting_transition(_program(tmp_path, minted), "mint")
    assert minted_result.status == "potential"
    assert minted_result.relation == "supply-increased-without-asset-inflow"


def test_invariant_preserved_and_potential_are_not_proofs(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        uint256 totalSupply;
        function mint() external { totalSupply = totalSupply + 1; }
        function ping() external {}
    }
    """
    model = analyze_state_transitions(_program(tmp_path, source))
    assert model.invariants
    assert all(item.status == "candidate" for item in model.invariants)
    statuses = {item.status for item in model.checks}
    assert "potential" in statuses
    assert "non_interference_observed" in statuses
    assert "preserved" not in statuses
    assert "safe" not in statuses
    assert _FORBIDDEN.isdisjoint(statuses)
    assert _FORBIDDEN.isdisjoint({item.status for item in model.paths})


def test_multi_step_depth_and_bounds(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        uint256 totalSupply;
        function a() external { b(); }
        function b() internal { c(); }
        function c() internal { d(); }
        function d() internal { e(); }
        function e() internal { totalSupply = totalSupply + 1; }
    }
    """
    program = _program(tmp_path, source)
    model = analyze_state_transitions(program)
    assert MAX_DEPTH == 4
    truncated = [item for item in model.paths if item.completeness == "partial"]
    assert truncated
    assert all(item.status == "incomplete" for item in truncated)
    assert all("Vault.e" not in item.function_ids[-1] for item in truncated)
    assert model.status == "partial"
    assert "depth bound reached" in model.incomplete_reason
    reached = [
        item
        for item in model.paths
        if item.completeness == "complete"
        and item.depth == 3
        and "invokes" in {step.relation for step in item.steps}
    ]
    assert reached
    limited = analyze_state_transitions(program, limits={"paths": 1, "transitions": 1})
    assert limited.status == "partial"
    assert "transition limit reached" in limited.incomplete_reason
    assert len(limited.transitions) == 1


def test_cross_contract_resolution_is_not_guessed(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    interface Feed { function latestAnswer() external view returns (int256); }
    contract Token {
        uint256 totalAssets;
        uint256 totalSupply;
        function mint() external { totalSupply = totalSupply + 1; }
    }
    contract Vault {
        function deposit() external { Token.mint(); }
        function price(Feed feed) external view { feed.latestAnswer(); }
    }
    """
    model = analyze_state_transitions(_program(tmp_path, source))
    deposit = next(
        item for item in model.transitions if item.function_id.startswith("Vault.deposit")
    )
    assert any(part.endswith("resolved:contract-typed") for part in deposit.resolutions)
    price = next(item for item in model.transitions if item.function_id.startswith("Vault.price"))
    assert any("not a resolved function" in part or "unknown" in part for part in price.resolutions)
    chain = [
        item
        for item in model.paths
        if item.depth == 2 and any(step.relation == "invokes" for step in item.steps)
    ]
    assert chain
    assert {item.status for item in chain} == {"reachable"}
    assert all(item.relationship == "reachable" for item in chain)
    assert all("reachable only" in item.reason for item in chain)


def test_candidate_steps_are_structured_and_authority_is_operation_specific(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        address owner;
        uint256 totalAssets;
        function withdraw() external {
            uint256 owed = totalAssets;
            msg.sender.call("");
            totalAssets = owed;
        }
        function setOwner(address next) external {
            owner = next;
        }
        function guarded(address next) external {
            require(msg.sender == owner);
            owner = next;
        }
    }
    """
    program = _program(tmp_path, source)
    paths = find_candidate_exploit_paths(program)
    reentrancy = next(item for item in paths if item.path_id.startswith("reentrancy:"))
    assert [step.relation for step in reentrancy.steps] == [
        "reads-from",
        "depends-on",
        "externally-calls",
        "writes-to",
    ]
    assert reentrancy.depth == 1
    assert reentrancy.completeness == "complete"
    assert any(item.path_id.startswith("changes-authority:") for item in paths)
    assert not any(
        "guarded" in item.function_ids[0] and item.path_id.startswith("changes-authority:")
        for item in paths
    )


def test_branch_write_is_a_real_operation_and_exclusive_branches_are_not(tmp_path: Path) -> None:
    dependent = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        function run(bool flag) external {
            uint256 owed = totalAssets;
            if (flag) {
                msg.sender.call("");
                totalAssets = owed;
            }
        }
    }
    """
    exclusive = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        function run(bool flag) external {
            if (flag) {
                uint256 owed = totalAssets;
                msg.sender.call("");
            } else {
                totalAssets = 1;
            }
        }
    }
    """
    program = _program(tmp_path, dependent)
    function = program.functions_named("run", "Vault")[0]
    kinds = [item.kind for item in function.access_sites]
    assert kinds == ["read", "write"]
    assert analyze_reentrancy(program, "run").status == "potential"
    exclusive_program = _program(tmp_path, exclusive)
    exclusive_function = exclusive_program.functions_named("run", "Vault")[0]
    assert [item.kind for item in exclusive_function.access_sites] == ["read", "write"]
    assert analyze_reentrancy(exclusive_program, "run").status == "unknown"


def test_internal_call_does_not_preserve_a_callee_write(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        uint256 totalSupply;
        function ping() external { touch(); }
        function touch() internal { totalSupply = totalSupply + 1; }
        function idle() external {}
    }
    """
    model = analyze_state_transitions(_program(tmp_path, source))
    asset_checks = [
        item for item in model.checks if item.invariant_id.startswith("inv:asset-share:")
    ]
    by_function = {item.transition_id.split(":", 1)[0]: item.status for item in asset_checks}
    assert by_function["Vault.ping"] == "unknown"
    assert by_function["Vault.touch"] == "potential"
    assert by_function["Vault.idle"] == "non_interference_observed"


def test_comment_does_not_count_as_asset_inflow(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalAssets;
        uint256 totalSupply;
        function mint(uint256 shares) external {
            // msg.value is not received here
            totalSupply = totalSupply + shares;
        }
    }
    """
    result = analyze_accounting_transition(_program(tmp_path, source), "mint")
    assert result.status == "potential"
    assert result.relation == "supply-increased-without-asset-inflow"


def test_allowance_requires_a_value_reaching_transfer_from(tmp_path: Path) -> None:
    used = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint256) allowance;
        function pay(address token, address from, uint256 amount) external {
            uint256 current = allowance[from];
            token.transferFrom(from, address(this), current);
        }
    }
    """
    unused = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint256) allowance;
        function pay(address token, address from, uint256 amount) external {
            uint256 current = allowance[from];
            token.transferFrom(from, address(this), amount);
        }
    }
    """
    used_result = analyze_accounting_transition(_program(tmp_path, used), "pay")
    unused_result = analyze_accounting_transition(_program(tmp_path, unused), "pay")
    assert used_result.status == "potential"
    assert used_result.relation == "allowance-not-consumed"
    assert unused_result.status == "observed"
    assert unused_result.relation == "asset-transfer"


def test_external_target_is_not_the_same_named_local_function(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Token {
        function other() internal {}
    }
    contract Vault {
        uint256 totalSupply;
        uint256 totalAssets;
        function other() internal { totalSupply = totalSupply + 1; }
        function run(address token) external {
            other();
            token.other();
        }
    }
    """
    from app.parsing.solidity_value_flow import resolve_call

    program = _program(tmp_path, source)
    function = program.functions_named("run", "Vault")[0]
    resolutions = [resolve_call(program, function, site) for site in function.call_sites]
    assert resolutions[0].status == "resolved"
    assert resolutions[0].function_id.startswith("Vault.other")
    assert resolutions[1].status == "unknown"
    model = analyze_state_transitions(program)
    invokes = [
        step.sink_id
        for path in model.paths
        for step in path.steps
        if step.relation == "invokes" and step.function_id.startswith("Vault.run")
    ]
    assert any(item.startswith("Vault.other") for item in invokes)
    assert not any(item.startswith("Token.other") for item in invokes)


def test_supply_and_asset_invariants_are_judged_separately(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 balances;
        uint256 totalSupply;
        uint256 totalAssets;
        function mint() external { totalSupply = totalSupply + 1; }
        function sync(uint256 amount) external {
            balances = balances + amount;
            totalSupply = totalSupply + amount;
        }
    }
    """
    model = analyze_state_transitions(_program(tmp_path, source))

    def status(function: str, prefix: str) -> str:
        return next(
            item.status
            for item in model.checks
            if item.transition_id.startswith(function) and item.invariant_id.startswith(prefix)
        )

    assert status("Vault.mint", "inv:asset-share:") == "potential"
    assert status("Vault.mint", "inv:supply-balance:") == "potential"
    assert status("Vault.sync", "inv:supply-balance:") != "potential"


def test_later_unguarded_write_is_not_hidden_by_an_earlier_guard(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        address owner;
        function set(address next) external {
            if (msg.sender == owner) { owner = next; }
            owner = next;
        }
    }
    """
    program = _program(tmp_path, source)
    function = program.functions_named("set", "Vault")[0]
    writes = [item for item in function.access_sites if item.kind != "read"]
    assert [item.guard_status for item in writes] == ["guarded", "unguarded"]
    model = analyze_state_transitions(program)
    assert any(
        item.status == "potential" and item.invariant_id.startswith("inv:authorization:")
        for item in model.checks
        if item.transition_id == function.identity
    )


def test_paths_never_claim_verification(tmp_path: Path) -> None:
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
    model = analyze_state_transitions(_program(tmp_path, source))
    assert model.origin == "parser"
    assert model.compiler_ir_status != "available" or model.compiler_version
    assert _FORBIDDEN.isdisjoint({item.status for item in model.paths})
    assert _FORBIDDEN.isdisjoint({item.status for item in model.transitions})
    assert _FORBIDDEN.isdisjoint({item.status for item in model.checks})


def test_same_function_identity_in_two_sources_is_not_merged(tmp_path: Path) -> None:
    left_source = """pragma solidity ^0.8.20;
contract Vault {
    uint256 totalSupply;
    uint256 balances;
    function mint() external { totalSupply = totalSupply + 1; }
}
"""
    right_source = """pragma solidity ^0.8.19;
contract Vault {
    uint256 totalSupply;
    uint256 balances;
    function mint() external { totalSupply = totalSupply + 1; balances = balances + 1; }
}
"""
    reset_syntax_registry()
    reset_plugin_catalog()
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    left_path = one / "Vault.sol"
    right_path = two / "Vault.sol"
    left_path.write_text(left_source, encoding="utf-8")
    right_path.write_text(right_source, encoding="utf-8")
    left = build_semantic_program(parse_source("solidity", left_path, left_source))
    right = build_semantic_program(parse_source("solidity", right_path, right_source))
    left.compiler_version = "0.8.20"
    right.compiler_version = "0.8.19"
    model = analyze_state_transitions(left, also=(right,))
    assert model.status == "partial"
    assert "namespaced" in model.incomplete_reason
    assert model.compiler_version == "mixed"
    mints = [item for item in model.transitions if item.function_id.startswith("Vault.mint")]
    assert len(mints) == 2
    assert len({item.transition_id for item in mints}) == 2
    assert len({item.source_identity for item in mints}) == 2
    assert all(item.transition_id.startswith(item.source_identity) for item in mints)
    assert {item.source_file for item in mints} == {str(left_path), str(right_path)}
    left_mint = next(item for item in mints if item.source_file == str(left_path))
    right_mint = next(item for item in mints if item.source_file == str(right_path))
    assert any(
        item.transition_id == left_mint.transition_id
        and item.status == "potential"
        and item.invariant_id.startswith("inv:supply-balance:")
        for item in model.checks
    )
    assert not any(
        item.transition_id == right_mint.transition_id and item.status == "potential"
        for item in model.checks
    )
    assert len({item.invariant_id for item in model.invariants if item.category == "equality"}) == 2


def test_same_variable_name_does_not_merge_contracts(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Token {
        uint256 totalSupply;
        uint256 balances;
        function mint() external { totalSupply = totalSupply + 1; }
    }
    contract Unrelated {
        uint256 totalSupply;
        uint256 balances;
        function idle() external {}
    }
    """
    model = analyze_state_transitions(_program(tmp_path, source))
    equalities = [item for item in model.invariants if item.category == "equality"]
    assert len(equalities) == 2
    assert {item.contract for item in equalities} == {"Token", "Unrelated"}
    mint_checks = [
        item
        for item in model.checks
        if item.transition_id.startswith("Token.mint") and item.status == "potential"
    ]
    assert mint_checks
    assert all("Token" in item.invariant_id for item in mint_checks)
    assert not any("Unrelated" in item.invariant_id for item in mint_checks)
    assert len({item.declarations for item in equalities}) == 2
    assert all(
        item.predicate is not None and item.predicate.predicate_type == "equality"
        for item in equalities
    )
    assert all(item.representation_status == "exact" for item in equalities)


def test_mapping_sum_is_not_an_exact_formula(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Token {
        uint256 totalSupply;
        mapping(address => uint256) balances;
        function mint() external { totalSupply = totalSupply + 1; }
    }
    """
    model = analyze_state_transitions(_program(tmp_path, source))
    equality = next(item for item in model.invariants if item.category == "equality")
    assert equality.representation_status == "partial"
    assert equality.predicate is not None
    assert equality.predicate.rhs.startswith("sum(")
    assert "mapping aggregation is not encoded" in equality.unsupported


def test_shared_unrelated_state_is_not_a_security_path(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalSupply;
        uint256 balances;
        uint256 scratch;
        function a() external { scratch = 1; b(); }
        function b() external { uint256 seen = scratch; c(); }
        function c() external { totalSupply = totalSupply + 1; }
    }
    """
    model = analyze_state_transitions(_program(tmp_path, source))
    related = next(
        item
        for item in model.paths
        if item.path_id.startswith("chain:") and item.function_ids[0].startswith("Vault.a")
    )
    assert related.relationship == "semantically_related"
    assert related.status == "reachable"
    assert related.status != "candidate"


def test_multi_step_requires_shared_state(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalSupply;
        uint256 balances;
        uint256 other;
        function a() external { uint256 seen = totalSupply; b(); }
        function b() external { c(); }
        function c() external { totalSupply = totalSupply + 1; }
        function u() external { v(); }
        function v() external { w(); }
        function w() external { totalSupply = totalSupply + 1; }
    }
    """
    model = analyze_state_transitions(_program(tmp_path, source))
    connected = next(
        item
        for item in model.paths
        if item.path_id.startswith("chain:")
        and item.function_ids[0].startswith("Vault.a")
        and item.function_ids[-1].startswith("Vault.c")
    )
    unrelated = next(
        item
        for item in model.paths
        if item.path_id.startswith("chain:")
        and item.function_ids[0].startswith("Vault.u")
        and item.function_ids[-1].startswith("Vault.w")
    )
    assert connected.status == "candidate"
    assert connected.relationship == "candidate_security_path"
    assert connected.invariant_id.startswith("inv:")
    assert any(step.relation == "depends-on" for step in connected.steps)
    assert unrelated.status == "reachable"
    assert unrelated.relationship == "reachable"
    assert "reachable only" in unrelated.reason
    assert not any(step.relation == "violates-candidate-invariant" for step in unrelated.steps)


def test_depth_counts_transitions_and_the_bound_is_partial(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalSupply;
        uint256 balances;
        function a() external { uint256 seen = totalSupply; b(); }
        function b() external { c(); }
        function c() external { d(); }
        function d() external { totalSupply = totalSupply + 1; e(); }
        function e() external { totalSupply = totalSupply + 1; }
    }
    """
    model = analyze_state_transitions(_program(tmp_path, source))
    intra = [
        item for item in model.paths if item.depth == 1 and item.relationship == "intra-function"
    ]
    assert intra
    assert all(item.bound.startswith("depth=1;") for item in intra)
    depth2 = [
        item
        for item in model.paths
        if item.depth == 2
        and item.function_ids[0].startswith("Vault.c")
        and item.function_ids[-1].startswith("Vault.d")
    ]
    depth3 = [
        item
        for item in model.paths
        if item.depth == 3
        and item.function_ids[0].startswith("Vault.b")
        and item.function_ids[-1].startswith("Vault.d")
    ]
    depth4 = [
        item
        for item in model.paths
        if item.depth == 4
        and item.function_ids[0].startswith("Vault.a")
        and item.function_ids[-1].startswith("Vault.d")
    ]
    assert depth2 and depth2[0].status == "reachable"
    assert depth3 and depth3[0].status == "reachable"
    assert depth4 and depth4[0].status == "candidate"
    assert depth4[0].completeness == "complete"
    assert depth4[0].bound == f"depth=4; cap<={MAX_DEPTH}"
    truncated = [item for item in model.paths if item.path_id.startswith("explore:")]
    assert truncated
    assert all(item.status == "incomplete" for item in truncated)
    assert all(item.completeness == "partial" for item in truncated)
    assert all("truncated exploration" in item.reason for item in truncated)
    assert all(
        not any(step.relation == "violates-candidate-invariant" for step in item.steps)
        for item in truncated
    )
    assert model.status == "partial"
    assert "depth bound reached" in model.incomplete_reason


def test_allowance_write_covers_only_earlier_calls(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint256) allowance;
        function pay(address token, address from, uint256 amount) external {
            uint256 current = allowance[from];
            token.transferFrom(from, address(this), current);
            allowance[from] = 0;
            token.transferFrom(from, address(this), current);
        }
    }
    """
    result = analyze_accounting_transition(_program(tmp_path, source), "pay")
    assert result.status == "potential"
    assert result.relation == "allowance-not-consumed"


def test_loop_is_not_non_interference(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalSupply;
        uint256 balances;
        function idle() external {
            uint256 i = 0;
            while (i < 1) { i = i + 1; }
        }
    }
    """
    model = analyze_state_transitions(_program(tmp_path, source))
    checks = [item for item in model.checks if item.transition_id.startswith("Vault.idle")]
    assert checks
    assert {item.status for item in checks} == {"incomplete"}
    assert all("loop-carried" in item.reason for item in checks)


def test_external_call_is_not_non_interference(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint256 totalSupply;
        uint256 balances;
        function ping(address target) external { target.call(""); }
    }
    """
    model = analyze_state_transitions(_program(tmp_path, source))
    checks = [item for item in model.checks if item.transition_id.startswith("Vault.ping")]
    assert checks
    assert {item.status for item in checks} == {"unknown"}
    assert all("external call" in item.reason for item in checks)
