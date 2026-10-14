"""Phase 30: Solidity CFG and intra-procedural dataflow."""

from __future__ import annotations

from app.parsing.solidity_cfg import (
    auth_dominates_sensitive,
    build_function_cfg,
    operation_guarded,
    placeholder_is_guarded,
    write_reachable_after,
)
from app.parsing.solidity_flow import (
    analyze_flow,
    digest_contains,
    oracle_freshness_protects,
    signature_domain_gap,
    signature_replay_gap,
)


def test_nested_if_and_branch_local_checks() -> None:
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
    opposite = """
    function setOwner(address next, address other) external {
        if (msg.sender == owner) { owner = next; }
        else { owner = other; }
    }
    """
    revert_guard = """
    function setOwner(address next) external {
        if (msg.sender != owner) revert();
        owner = next;
    }
    """
    assert auth_dominates_sensitive(guarded) is True
    assert auth_dominates_sensitive(late) is False
    assert auth_dominates_sensitive(opposite) is False
    assert auth_dominates_sensitive(revert_guard) is True
    assert operation_guarded(opposite, "owner = other") is False
    assert operation_guarded(revert_guard, "owner = next") is True


def test_revert_and_return_terminate() -> None:
    source = """
    function f(address t) external {
        if (t == address(0)) { revert(); }
        t.call("");
        return;
        owner = next;
    }
    """
    assert write_reachable_after(source, 't.call("")', "owner = next") is False
    cfg = build_function_cfg(source)
    assert cfg.known is True
    kinds = {node.kind for node in cfg.nodes}
    assert "return" in kinds and "revert" in kinds


def test_loop_back_edge_does_not_reach_earlier_writes() -> None:
    source = """
    function f(address t) external {
        owner = next;
        for (uint i; i < n; i++) {
            t.call("");
            balances[msg.sender] = 0;
        }
    }
    """
    assert write_reachable_after(source, 't.call("")', "owner = next") is False
    assert write_reachable_after(source, 't.call("")', "balances[msg.sender] = 0") is True
    broken = "function f() external { for (uint i; i < n; i++) { "
    assert build_function_cfg(broken).known is False


def test_break_continue_and_try_paths() -> None:
    broken_loop = """
    function f(address t) external {
        for (uint i; i < 3; i++) {
            if (i == 1) { break; }
            t.call("");
        }
    }
    """
    assert write_reachable_after(broken_loop, "break;", 't.call("")') is False
    continued = """
    function f(address t) external {
        owner = next;
        while (i < n) {
            if (i == 1) { continue; }
            t.call("");
        }
    }
    """
    assert write_reachable_after(continued, "continue;", "owner = next") is False
    assert write_reachable_after(continued, "continue;", 't.call("")') is True
    tried = """
    function f(address token, address t) external {
        try token.foo() { t.call(""); }
        catch { owner = next; }
    }
    """
    cfg = build_function_cfg(tried)
    assert cfg.known is True
    assert write_reachable_after(tried, 't.call("")', "owner = next") is False


def test_unchecked_and_do_while_are_structured() -> None:
    source = """
    function f() external {
        unchecked { owner = next; }
        require(msg.sender == owner);
    }
    """
    assert auth_dominates_sensitive(source) is False
    loop = """
    function f(address t) external {
        do { t.call(""); } while (i < 2);
        owner = next;
    }
    """
    cfg = build_function_cfg(loop)
    assert cfg.known is True
    assert any(node.kind == "loop" for node in cfg.nodes)
    assert write_reachable_after(loop, 't.call("")', "owner = next") is True


def test_modifier_bodies_are_not_trusted_by_name() -> None:
    assert placeholder_is_guarded("{ _; }") is False
    assert placeholder_is_guarded("{ require(msg.sender == owner); _; }") is True
    assert placeholder_is_guarded("{ _; require(msg.sender == owner); }") is False
    assert placeholder_is_guarded("{ if (msg.sender != owner) revert(); _; }") is True
    assert placeholder_is_guarded("{ if (msg.sender == owner) { _; } }") is True


def test_digest_includes_only_values_that_reach_ecrecover() -> None:
    bound = """
    function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        nonce += 1;
        h = keccak256(abi.encode(DOMAIN_SEPARATOR, address(this), nonce));
        return ecrecover(h, v, r, s);
    }
    """
    missed = """
    function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        nonce += 1;
        bytes32 domain = DOMAIN_SEPARATOR;
        return ecrecover(h, v, r, s);
    }
    """
    assert digest_contains(bound, "nonce") is True
    assert digest_contains(bound, "DOMAIN_SEPARATOR") is True
    assert digest_contains(bound, "address(this)") is True
    assert signature_replay_gap(bound) is None
    assert signature_domain_gap(bound) is None
    assert digest_contains(missed, "nonce") is False
    assert signature_replay_gap(missed) is not None
    assert signature_domain_gap(missed) is not None
    flow = analyze_flow(bound)
    assert flow.known is True
    assert any(
        edge.source == "nonce" and edge.target == "h" and edge.kind == "hash" for edge in flow.edges
    )
    assert all(edge.end >= edge.start for edge in flow.edges)


def test_oracle_freshness_must_come_from_the_round() -> None:
    fresh = """
    function price() external view returns (int) {
        (uint80 roundId, int256 answer, uint startedAt, uint updatedAt, uint80 answeredInRound) =
            feed.latestRoundData();
        require(updatedAt != 0 && block.timestamp - updatedAt < 1 hours);
        return answer;
    }
    """
    unrelated = """
    function price() external view returns (int) {
        uint updatedAt = block.timestamp;
        (, int256 answer,,,) = feed.latestRoundData();
        require(updatedAt != 0);
        return answer;
    }
    """
    late = """
    function price() external view returns (int) {
        (, int256 answer,, uint updatedAt,) = feed.latestRoundData();
        int value = answer;
        require(block.timestamp - updatedAt < 1 hours);
        return value;
    }
    """
    assert oracle_freshness_protects(fresh) is True
    assert oracle_freshness_protects(unrelated) is False
    assert oracle_freshness_protects(late) is False
