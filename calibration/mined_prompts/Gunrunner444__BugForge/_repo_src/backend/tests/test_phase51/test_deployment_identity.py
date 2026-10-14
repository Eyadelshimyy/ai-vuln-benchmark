"""Phase 51 deployment/scope identity reconciliation (owner Phase 3).

Source is never assumed to be the deployed code, and an ambiguous binding is never
promoted to confidently-in-scope.
"""

from __future__ import annotations

import copy

from app.discovery.bounty.campaign import BountyManifest
from app.discovery.bounty.deployment_identity import (
    BindingConfidence,
    ProxyKind,
    ScopeIdentityStatus,
    detect_proxy_topology,
    reconcile_deployment,
    resolve_scope_identity,
)
from app.parsing.solidity_research import build_research_model
from tests.test_phase50.phase50_support import MANIFEST_DATA

DIGEST = "ab" * 32
ADDR = "0x" + "11" * 20


def _model(text: str):
    return build_research_model({"C.sol": text})


def _manifest(**overrides) -> BountyManifest:
    data = copy.deepcopy(MANIFEST_DATA)
    data.update(overrides)
    return BountyManifest.from_dict(data)


# ---- proxy topology detection (evidence-based) ----------------------------------------------


def test_detect_uups() -> None:
    topo = detect_proxy_topology(
        _model("contract P is UUPSUpgradeable { function _authorizeUpgrade(address) internal {} }"),
        "P",
    )
    assert topo.kind is ProxyKind.UUPS and topo.is_proxy


def test_detect_transparent() -> None:
    topo = detect_proxy_topology(
        _model(
            "contract P is TransparentUpgradeableProxy { fallback() external { assembly "
            "{ let r := delegatecall(gas(), sload(0), 0, 0, 0, 0) } } }"
        ),
        "P",
    )
    assert topo.kind is ProxyKind.TRANSPARENT


def test_detect_beacon() -> None:
    topo = detect_proxy_topology(
        _model(
            "contract P { IBeacon beacon; fallback() external { address i = beacon.implementation(); "
            "assembly { let r := delegatecall(gas(), i, 0, 0, 0, 0) } } }"
        ),
        "P",
    )
    assert topo.kind is ProxyKind.BEACON


def test_detect_diamond() -> None:
    topo = detect_proxy_topology(
        _model(
            "contract D { function diamondCut() external {} fallback() external { address f; "
            "assembly { let r := delegatecall(gas(), f, 0, 0, 0, 0) } } }"
        ),
        "D",
    )
    assert topo.kind is ProxyKind.DIAMOND


def test_detect_clone() -> None:
    topo = detect_proxy_topology(
        _model("contract F { function make() external { Clones.clone(address(this)); } }"),
        "F",
    )
    assert topo.kind is ProxyKind.CLONE


def test_plain_contract_is_not_a_proxy() -> None:
    topo = detect_proxy_topology(_model("contract A { uint x; function f() external {} }"), "A")
    assert topo.kind is ProxyKind.NONE and not topo.is_proxy


def test_keyword_without_delegatecall_is_not_a_proxy() -> None:
    # "beacon" mentioned but no delegatecall / slot -> evidence-based detection says no
    topo = detect_proxy_topology(
        _model("contract A { string note = 'see the beacon docs'; function f() external {} }"),
        "A",
    )
    assert topo.kind is ProxyKind.NONE


# ---- binding reconciliation -----------------------------------------------------------------


def test_matching_digest_is_bound() -> None:
    mani = _manifest(
        compiler={"version": "0.8.25", "runtime_bytecode_digest": DIGEST},
        deployments=[
            {
                "address": ADDR,
                "chain_id": "1",
                "contract_name": "SelfTrustMulticall",
                "runtime_bytecode_digest": DIGEST,
            }
        ],
    )
    binding = reconcile_deployment(
        mani,
        contract="SelfTrustMulticall",
        address=ADDR,
        chain_id="1",
        model=_model("contract SelfTrustMulticall { function f() external {} }"),
    )
    assert binding.source_matches_deployed == "yes"
    assert binding.confidence is BindingConfidence.BOUND


def test_mismatching_digest_is_ambiguous() -> None:
    mani = _manifest(
        compiler={"version": "0.8.25", "runtime_bytecode_digest": DIGEST},
        deployments=[
            {
                "address": ADDR,
                "chain_id": "1",
                "contract_name": "SelfTrustMulticall",
                "runtime_bytecode_digest": "cd" * 32,
            }
        ],
    )
    binding = reconcile_deployment(
        mani,
        contract="SelfTrustMulticall",
        address=ADDR,
        chain_id="1",
        model=_model("contract SelfTrustMulticall { function f() external {} }"),
    )
    assert binding.source_matches_deployed == "no"
    assert binding.confidence is BindingConfidence.AMBIGUOUS


def test_unknown_digest_never_assumes_source_is_deployed() -> None:
    mani = _manifest()  # no runtime digests
    binding = reconcile_deployment(
        mani,
        contract="SelfTrustMulticall",
        address=ADDR,
        chain_id="1",
        model=_model("contract SelfTrustMulticall { function f() external {} }"),
    )
    assert binding.source_matches_deployed == "unknown"
    assert binding.confidence is BindingConfidence.AMBIGUOUS


def test_no_deployment_is_unbound() -> None:
    mani = _manifest(deployments=[])
    binding = reconcile_deployment(
        mani,
        contract="SelfTrustMulticall",
        model=_model("contract SelfTrustMulticall { function f() external {} }"),
    )
    assert binding.confidence is BindingConfidence.UNBOUND


def test_proxy_deployment_is_ambiguous_even_with_matching_digest() -> None:
    mani = _manifest(
        compiler={"version": "0.8.25", "runtime_bytecode_digest": DIGEST},
        deployments=[
            {
                "address": ADDR,
                "chain_id": "1",
                "contract_name": "SelfTrustMulticall",
                "runtime_bytecode_digest": DIGEST,
            }
        ],
    )
    binding = reconcile_deployment(
        mani,
        contract="SelfTrustMulticall",
        address=ADDR,
        chain_id="1",
        model=_model(
            "contract SelfTrustMulticall is UUPSUpgradeable { function _authorizeUpgrade(address) internal {} }"
        ),
    )
    assert binding.proxy.kind is ProxyKind.UUPS
    assert binding.confidence is BindingConfidence.AMBIGUOUS


# ---- scope identity -------------------------------------------------------------------------


def test_bound_and_in_scope_is_confidently_in_scope() -> None:
    mani = _manifest(
        compiler={"version": "0.8.25", "runtime_bytecode_digest": DIGEST},
        deployments=[
            {
                "address": ADDR,
                "chain_id": "1",
                "contract_name": "SelfTrustMulticall",
                "runtime_bytecode_digest": DIGEST,
            }
        ],
    )
    decision = resolve_scope_identity(
        mani,
        contract="SelfTrustMulticall",
        address=ADDR,
        chain_id="1",
        model=_model("contract SelfTrustMulticall { function f() external {} }"),
    )
    assert decision.status is ScopeIdentityStatus.CONFIDENTLY_IN_SCOPE


def test_in_scope_but_unknown_digest_is_ambiguous_not_in_scope() -> None:
    mani = _manifest()
    decision = resolve_scope_identity(
        mani,
        contract="SelfTrustMulticall",
        file="caller_context_vulnerable.sol",
        address=ADDR,
        chain_id="1",
        model=_model("contract SelfTrustMulticall { function f() external {} }"),
    )
    assert decision.status is ScopeIdentityStatus.AMBIGUOUS
    assert not decision.confidently_in_scope


def test_out_of_scope_target() -> None:
    mani = _manifest()
    decision = resolve_scope_identity(
        mani,
        contract="BalanceAsDeposit",
        file="accounting_vulnerable.sol",
        model=_model("contract BalanceAsDeposit {}"),
    )
    assert decision.status is ScopeIdentityStatus.OUT_OF_SCOPE


def test_unlisted_target_is_unknown_scope() -> None:
    mani = _manifest()
    decision = resolve_scope_identity(
        mani, contract="RandomThing", model=_model("contract RandomThing {}")
    )
    assert decision.status is ScopeIdentityStatus.UNKNOWN
