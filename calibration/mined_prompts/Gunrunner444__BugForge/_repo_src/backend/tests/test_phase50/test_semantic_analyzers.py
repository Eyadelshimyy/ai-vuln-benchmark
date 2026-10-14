"""Positive and safe fixtures for every Phase 50 semantic family."""

from __future__ import annotations

import re

import pytest

from tests.test_phase50.phase50_support import (
    analyze,
    analyze_text,
    detectors,
    one,
    read,
    where,
)

SAFE_FIXTURES = (
    "caller_context_safe.sol",
    "oracle_safe.sol",
    "message_safe.sol",
    "accounting_safe.sol",
    "arithmetic_safe.sol",
    "aa_safe.sol",
    "no_aa.sol",
    "compiler_advisory_safe.sol",
)


# The ERC-4337 prefund in aa_safe.sol ignores its transfer result by design. The
# detector emits one low-confidence candidate. That emission is a possible false
# positive, not evidence that the safe fixture has a vulnerability, and not a
# precision hit.
def _without_labelled_prefund(candidates):
    rest, prefund = [], []
    for item in candidates:
        facts = dict(item.facts)
        if (
            item.detector == "caller_context.swallowed_call_failure"
            and facts.get("call_kind") == "eth_transfer_variable"
            and item.confidence == "low"
            and "best-effort" in facts.get("intent", "")
            and item.function.startswith("validateUserOp")
        ):
            assert facts.get("evidence_class") == "possible_false_positive"
            assert facts.get("pattern") == "prefund_style_transfer"
            assert facts.get("vulnerability_claim") == "none"
            assert item.verified is False
            prefund.append(item)
        else:
            rest.append(item)
    assert len(prefund) <= 1
    return tuple(rest)


@pytest.mark.parametrize("name", SAFE_FIXTURES)
def test_safe_fixtures_report_nothing(name: str) -> None:
    result = analyze(name)
    assert _without_labelled_prefund(result.candidates) == ()


def test_every_candidate_is_static_and_unverified() -> None:
    for name in (
        "caller_context_vulnerable.sol",
        "oracle_vulnerable.sol",
        "message_vulnerable.sol",
        "accounting_vulnerable.sol",
        "arithmetic_vulnerable.sol",
        "aa_vulnerable.sol",
    ):
        for item in analyze(name).candidates:
            assert item.verified is False
            assert item.status == "candidate"
            assert item.family


# ---- caller context -------------------------------------------------------------------------


def test_caller_context_confusion_variants_are_found() -> None:
    result = analyze("caller_context_vulnerable.sol", families=("caller_context",))
    assert where(result, "caller_context.self_call_elevation") == {
        ("SelfTrustMulticall", "multicall")
    }
    assert where(result, "caller_context.unrestricted_dispatch_with_approval_authority") == {
        ("ApprovedRouter", "execute")
    }
    assert where(result, "caller_context.trusted_intermediary_actor_parameter") == {
        ("UserRouter", "relayWithdraw")
    }
    assert where(result, "caller_context.forwarded_sender_delegatecall") == {
        ("ForwarderMulticall", "multicall")
    }


def test_caller_context_candidate_names_the_identity_change() -> None:
    result = analyze("caller_context_vulnerable.sol", families=("caller_context",))
    item = one(result, "caller_context.self_call_elevation", "SelfTrustMulticall")
    for fact in (
        "original_actor",
        "intermediate_callers",
        "eventual_callee",
        "authorization_check",
        "identity_difference",
    ):
        assert item.fact(fact), fact
    assert item.verified is False
    assert "msg.sender" in item.fact("identity_difference")


def test_safe_nested_calls_that_change_msg_sender_are_not_flagged() -> None:
    result = analyze("caller_context_safe.sol", families=("caller_context",))
    assert result.candidates == ()
    source = read("caller_context_safe.sol")
    # the safe fixtures contain nested calls and delegatecalls on purpose
    assert "address(this).call" in source and "delegatecall" in source


def test_caller_context_does_not_depend_on_function_names() -> None:
    source = read("caller_context_vulnerable.sol")
    renamed = (
        source.replace("multicall", "fnAlpha")
        .replace("setTreasury", "fnBeta")
        .replace("relayWithdraw", "fnGamma")
        .replace("withdrawFor", "fnDelta")
        .replace("setAdmin", "fnEpsilon")
    )
    before = analyze_text(source, ("caller_context",))
    after = analyze_text(renamed, ("caller_context",))
    assert detectors(before) == detectors(after)
    assert len(before.candidates) == len(after.candidates)


def test_an_actor_name_is_not_proof_of_authorization() -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Trusting {
        address public treasury;
        function setTreasury(address next) external {
            require(msg.sender == address(this), "self only");
            treasury = next;
        }
        function owner() external view returns (address) { return treasury; }
        function onlyOwnerBatch(bytes[] calldata calls) external {
            for (uint256 i = 0; i < calls.length; i++) {
                (bool ok, ) = address(this).call(calls[i]);
                require(ok, "failed");
            }
        }
    }
    """
    result = analyze_text(source, ("caller_context",))
    assert ("Trusting", "onlyOwnerBatch") in where(result, "caller_context.self_call_elevation")


def test_cross_contract_caller_context_across_files() -> None:
    callee = """
    pragma solidity ^0.8.20;
    contract Vault {
        address public router;
        mapping(address => uint256) public balances;
        function withdrawFor(address user, uint256 amount) external {
            require(msg.sender == router, "router only");
            balances[user] -= amount;
        }
    }
    """
    caller = """
    pragma solidity ^0.8.20;
    interface IVault { function withdrawFor(address user, uint256 amount) external; }
    contract Router {
        IVault public vault;
        function relay(address user, uint256 amount) external {
            vault.withdrawFor(user, amount);
        }
    }
    """
    from app.parsing.solidity_research_suite import analyze_sources

    result = analyze_sources({"Vault.sol": callee, "Router.sol": caller}, ("caller_context",))
    assert where(result, "caller_context.trusted_intermediary_actor_parameter") == {
        ("Router", "relay")
    }


# ---- oracle ---------------------------------------------------------------------------------


def test_oracle_quality_findings_and_safe_counterparts() -> None:
    result = analyze("oracle_vulnerable.sol", families=("oracle_quality",))
    assert where(result, "oracle.insufficient_quorum") == {
        ("WeakQuorumAggregator", "aggregatePrice")
    }
    assert ("StaleFeedConsumer", "collateralPrice") in where(result, "oracle.stale_source_accepted")
    assert ("WeakFallbackOracle", "getPrice") in where(result, "oracle.fallback_weaker")
    assert ("SpotPriceVault", "spotPrice") in where(result, "oracle.spot_price_sensitive_use")
    assert analyze("oracle_safe.sol", families=("oracle_quality",)).candidates == ()


def test_oracle_candidate_states_source_count_and_quorum() -> None:
    result = analyze("oracle_vulnerable.sol", families=("oracle_quality",))
    item = one(result, "oracle.insufficient_quorum", "WeakQuorumAggregator")
    assert item.fact("sources") == "1"
    assert item.fact("quorum_minimum") == "1"
    assert "collateral" in item.fact("use")


# ---- proof and message binding --------------------------------------------------------------


def test_message_and_proof_binding_findings() -> None:
    result = analyze("message_vulnerable.sol", families=("message_binding",))
    assert ("BridgeVault", "release") in where(result, "message.field_not_bound")
    assert "recipient" in one(result, "message.field_not_bound", "BridgeVault").fact(
        "omitted_fields"
    )
    assert ("LooseMerkleDistributor", "claim") in where(result, "message.field_not_bound")
    assert ("CrossChainReceiver", "receiveMessage") in where(result, "message.field_not_bound")
    assert ("ReplayableClaim", "claim") in where(result, "message.replay_no_consumption")
    assert where(result, "signature.typehash_arity_mismatch") == {("UnsignedNoncePermit", "permit")}
    assert analyze("message_safe.sol", families=("message_binding",)).candidates == ()


def test_signature_domain_separation_is_chain_and_contract_specific() -> None:
    result = analyze("message_vulnerable.sol", families=("message_binding",))
    item = one(result, "message.missing_domain_separation", "BridgeVault")
    assert item.fact("chain_bound") == "false"
    assert item.fact("contract_bound") == "false"
    assert "signature.domain_not_refreshed_on_fork" in detectors(result)


# ---- token accounting -----------------------------------------------------------------------


def test_balance_delta_and_token_behavior_candidates() -> None:
    result = analyze("accounting_vulnerable.sol", families=("balance_delta",))
    assert where(result, "accounting.fee_on_transfer_mismatch") >= {("RawAmountVault", "deposit")}
    assert where(result, "accounting.donation_share_price") == {("DonationVault", "deposit")}
    assert where(result, "accounting.balance_as_deposit") == {("BalanceAsDeposit", "creditDeposit")}
    assert where(result, "accounting.pull_all_balance") == {("SweepEverything", "withdrawAll")}
    assert where(result, "accounting.cached_balance") == {("CachedBalance", "sync")}
    assert analyze("accounting_safe.sol", families=("balance_delta",)).candidates == ()


def test_token_behavior_is_unknown_not_assumed() -> None:
    result = analyze("accounting_vulnerable.sol", families=("balance_delta",))
    item = one(result, "accounting.fee_on_transfer_mismatch", "RawAmountVault")
    assert item.fact("token_behavior") == "unknown"
    assert item.missing


# ---- arithmetic and batches -----------------------------------------------------------------


def test_arithmetic_and_batch_candidates() -> None:
    result = analyze("arithmetic_vulnerable.sol", families=("arithmetic",))
    assert where(result, "arithmetic.batch_accumulation_overflow") == {("BatchPayout", "payAll")}
    assert where(result, "arithmetic.unsafe_downcast_after_arithmetic") == {
        ("NarrowedStorage", "stake")
    }
    assert where(result, "arithmetic.division_before_multiplication") == {
        ("PrecisionLoss", "reward")
    }
    assert where(result, "arithmetic.rounding_burn_floor") == {("FloorVault", "withdraw")}
    assert analyze("arithmetic_safe.sol", families=("arithmetic",)).candidates == ()


def test_unchecked_arithmetic_changes_the_batch_conclusion() -> None:
    source = """
    pragma solidity ^0.7.6;
    contract Old {
        uint256 public limit;
        function pay(uint256[] calldata xs) external {
            uint256 total;
            for (uint256 i = 0; i < xs.length; i++) { total += xs[i]; }
            require(total <= limit, "over");
            payable(msg.sender).transfer(total);
        }
    }
    """
    modern = source.replace("^0.7.6", "^0.8.20")
    old = one(
        analyze_text(source, ("arithmetic",)), "arithmetic.batch_accumulation_overflow", "Old"
    )
    assert old.fact("arithmetic") == "wrapping"
    assert analyze_text(modern, ("arithmetic",)).candidates == ()


# ---- account abstraction --------------------------------------------------------------------


def test_account_abstraction_is_conditional_on_the_code() -> None:
    plain = analyze("no_aa.sol")
    assert "account_abstraction" in plain.families_skipped
    assert "account_abstraction" not in plain.families_run
    present = analyze("aa_vulnerable.sol")
    assert "account_abstraction" in present.families_run


def test_account_abstraction_mismatches_are_found_and_safe_code_is_clean() -> None:
    result = analyze("aa_vulnerable.sol", families=("account_abstraction",))
    expected = {
        "aa.unprotected_account_initializer",
        "aa.signature_not_bound_to_userophash",
        "aa.signature_result_ignored",
        "aa.validate_without_entrypoint_binding",
        "aa.unauthenticated_account_execution",
        "aa.paymaster_authorization_not_bound",
        "aa.factory_salt_not_bound_to_owner",
    }
    assert expected <= detectors(result)
    assert _without_labelled_prefund(analyze("aa_safe.sol").candidates) == ()


# ---- shared properties ----------------------------------------------------------------------


def test_a_missing_family_is_reported_as_not_run_and_never_as_clean() -> None:
    result = analyze("oracle_safe.sol", families=("oracle_quality", "no_such_family"))
    assert "no_such_family" in result.families_skipped
    assert "oracle_quality" in result.families_run


def test_unresolved_inputs_stay_unknown_rather_than_safe() -> None:
    source = re.sub(
        r"contract (\w+)", r"contract \1 is MissingBase", read("caller_context_vulnerable.sol")
    )
    result = analyze_text(source, ("caller_context",))
    assert all(item.verified is False for item in result.candidates)
