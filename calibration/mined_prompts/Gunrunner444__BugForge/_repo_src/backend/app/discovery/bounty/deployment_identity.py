"""Deployment and scope identity reconciliation (Phase 51, owner Phase 3).

A contract that exists in a repository is not the contract that is deployed, and a
deployed address may be a proxy that delegates to an implementation that is not the
source under review. This module reconciles what the manifest *claims* about a
deployment (chain id, address, runtime bytecode digest, source commit, compiler)
against what the source *shows* (proxy / beacon / diamond / clone topology), and
reports the binding honestly:

* ``source == deployed`` is never assumed. It is ``yes`` only when a runtime bytecode
  digest is known on both sides and matches, ``no`` when both are known and differ,
  and ``unknown`` otherwise (for example when no compiler is available to produce a
  digest).
* A proxy/beacon/diamond/clone topology makes the behavioural binding *ambiguous*:
  the deployed address runs an implementation the source may not contain.
* An ambiguous binding is never promoted to "confidently in scope". Explicit scope
  inclusion plus an unambiguous binding is required for that.

Nothing here compiles, downloads, or executes. Detection is evidence-based (it needs
a delegatecall and a recognised slot or base, not a bare keyword).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.discovery.bounty.campaign import UNKNOWN, BountyManifest, Deployment, ScopeStatus
from app.parsing.solidity_research import ResearchModel

# ERC-1967 standard storage slots.
_ERC1967_IMPL = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
_ERC1967_BEACON = "0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50"
_ERC1967_ADMIN = "0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103"
# EIP-1167 minimal-proxy runtime prefix.
_EIP1167 = "363d3d373d3d3d363d73"

_DELEGATECALL = re.compile(r"\.delegatecall\s*\(|\bdelegatecall\s*\(")
_CLONE_CALL = re.compile(r"\b(?:Clones\.)?clone(?:Deterministic)?\s*\(")
_DIAMOND = re.compile(r"\bdiamondCut\b|\bLibDiamond\b|\bfacetAddress\b")
_BEACON = re.compile(r"\bIBeacon\b|\bBeaconProxy\b|\bUpgradeableBeacon\b")
_UUPS_BASE = re.compile(r"\bUUPSUpgradeable\b")
_UUPS_HOOK = re.compile(r"\b_authorizeUpgrade\b")
_TRANSPARENT = re.compile(r"\bTransparentUpgradeableProxy\b|\bifAdmin\b|\bProxyAdmin\b")

MAX_CONTRACTS = 400


class ProxyKind(StrEnum):
    NONE = "none"
    UUPS = "uups"
    TRANSPARENT = "transparent"
    BEACON = "beacon"
    DIAMOND = "diamond"
    CLONE = "clone"
    GENERIC_PROXY = "generic_proxy"
    UNKNOWN = "unknown"


class BindingConfidence(StrEnum):
    BOUND = "bound"  # source unambiguously corresponds to the deployed runtime
    AMBIGUOUS = "ambiguous"  # a proxy, or an undecidable digest, breaks the binding
    UNBOUND = "unbound"  # no deployment information to bind to at all


class ScopeIdentityStatus(StrEnum):
    CONFIDENTLY_IN_SCOPE = "confidently_in_scope"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProxyTopology:
    kind: ProxyKind
    contract: str
    reasons: tuple[str, ...]

    @property
    def is_proxy(self) -> bool:
        return self.kind not in {ProxyKind.NONE, ProxyKind.UNKNOWN}


@dataclass(frozen=True)
class DeploymentBinding:
    chain_id: str
    address: str
    contract_name: str
    runtime_bytecode_digest: str
    source_commit: str
    compiler_fingerprint: str
    proxy: ProxyTopology
    source_matches_deployed: str  # yes | no | unknown
    confidence: BindingConfidence
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "address": self.address,
            "contract_name": self.contract_name,
            "runtime_bytecode_digest": self.runtime_bytecode_digest or UNKNOWN,
            "source_commit": self.source_commit or UNKNOWN,
            "compiler_fingerprint": self.compiler_fingerprint or UNKNOWN,
            "proxy_kind": self.proxy.kind.value,
            "proxy_reasons": list(self.proxy.reasons),
            "source_matches_deployed": self.source_matches_deployed,
            "confidence": self.confidence.value,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ScopeIdentityDecision:
    status: ScopeIdentityStatus
    reason: str
    binding: DeploymentBinding | None
    scope_status: str

    @property
    def confidently_in_scope(self) -> bool:
        return self.status is ScopeIdentityStatus.CONFIDENTLY_IN_SCOPE

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "scope_status": self.scope_status,
            "binding": self.binding.as_dict() if self.binding else None,
        }


def detect_proxy_topology(model: ResearchModel, contract: str = "") -> ProxyTopology:
    """Classify a contract's proxy topology from the source evidence.

    When ``contract`` is empty, the strongest proxy topology found across the model is
    returned so a single-file proxy is not missed.
    """

    best = ProxyTopology(ProxyKind.NONE, contract, ())
    order = {
        ProxyKind.NONE: 0,
        ProxyKind.UNKNOWN: 0,
        ProxyKind.GENERIC_PROXY: 1,
        ProxyKind.CLONE: 2,
        ProxyKind.TRANSPARENT: 3,
        ProxyKind.UUPS: 3,
        ProxyKind.BEACON: 3,
        ProxyKind.DIAMOND: 3,
    }
    names = [contract] if contract else sorted(model.contracts)
    for name in names[:MAX_CONTRACTS]:
        item = model.contracts.get(name)
        if item is None:
            continue
        found = _classify_contract(name, item.head, item.body, item.bases)
        if order[found.kind] > order[best.kind]:
            best = found
    return best


def _classify_contract(name: str, head: str, body: str, bases: tuple[str, ...]) -> ProxyTopology:
    text = f"{head}\n{body}"
    base_text = " ".join(bases)
    has_delegate = bool(_DELEGATECALL.search(text))
    reasons: list[str] = []

    if _DIAMOND.search(text) and (has_delegate or "facet" in text.lower()):
        reasons.append("diamond facet routing (diamondCut/LibDiamond/facetAddress)")
        return ProxyTopology(ProxyKind.DIAMOND, name, tuple(reasons))
    if (_BEACON.search(text) or _BEACON.search(base_text)) and (
        has_delegate or _ERC1967_BEACON in text.lower()
    ):
        reasons.append("beacon proxy (IBeacon/BeaconProxy or ERC-1967 beacon slot)")
        return ProxyTopology(ProxyKind.BEACON, name, tuple(reasons))
    if _UUPS_BASE.search(base_text) or (_UUPS_HOOK.search(text) and _ERC1967_IMPL in text.lower()):
        reasons.append("UUPS upgradeable (_authorizeUpgrade / UUPSUpgradeable + impl slot)")
        return ProxyTopology(ProxyKind.UUPS, name, tuple(reasons))
    if _TRANSPARENT.search(text) or _TRANSPARENT.search(base_text):
        if has_delegate or _ERC1967_ADMIN in text.lower():
            reasons.append("transparent proxy (ifAdmin/ProxyAdmin + ERC-1967 admin slot)")
            return ProxyTopology(ProxyKind.TRANSPARENT, name, tuple(reasons))
    if _CLONE_CALL.search(text) or _EIP1167 in text.lower().replace("0x", ""):
        reasons.append("minimal proxy / clone (EIP-1167 or Clones.clone)")
        return ProxyTopology(ProxyKind.CLONE, name, tuple(reasons))
    if has_delegate and _ERC1967_IMPL in text.lower():
        reasons.append("generic proxy (fallback delegatecall + ERC-1967 implementation slot)")
        return ProxyTopology(ProxyKind.GENERIC_PROXY, name, tuple(reasons))
    return ProxyTopology(ProxyKind.NONE, name, ())


def reconcile_deployment(
    manifest: BountyManifest,
    *,
    contract: str = "",
    address: str = "",
    chain_id: str = "",
    model: ResearchModel | None = None,
) -> DeploymentBinding:
    """Bind a (chain, address, contract) target to a deployment and judge the binding.

    A target that matches several deployments (the same address on several chains
    with no chain given, or a contract name deployed more than once) is reported as
    ambiguous; it is never bound to the first match.
    """

    deployment, ambiguity = find_deployment(
        manifest, contract=contract, address=address, chain_id=chain_id
    )
    proxy = (
        detect_proxy_topology(model, contract)
        if model is not None
        else ProxyTopology(ProxyKind.UNKNOWN, contract, ("no source model supplied",))
    )
    reasons: list[str] = []

    if deployment is None:
        reasons.append(ambiguity or "no manifest deployment matches this target")
        return DeploymentBinding(
            chain_id=chain_id,
            address=address,
            contract_name=contract,
            runtime_bytecode_digest="",
            source_commit=manifest.source_commit,
            compiler_fingerprint=manifest.compiler.fingerprint(),
            proxy=proxy,
            source_matches_deployed="unknown",
            confidence=BindingConfidence.AMBIGUOUS if ambiguity else BindingConfidence.UNBOUND,
            reasons=tuple(reasons),
        )

    declared = _declared_topology(deployment)
    if declared is not None and not proxy.is_proxy:
        proxy = declared
    matches = _digest_match(manifest, deployment)
    reasons.append(_digest_reason(manifest, deployment, matches))
    confidence = BindingConfidence.BOUND
    if proxy.is_proxy:
        confidence = BindingConfidence.AMBIGUOUS
        reasons.append(
            f"deployed address is a {proxy.kind.value}; the running implementation "
            "may not be this source"
        )
    if deployment.source_commit and manifest.source_commit:
        if deployment.source_commit != manifest.source_commit:
            confidence = BindingConfidence.AMBIGUOUS
            reasons.append("the deployment was built from a different source commit")
    if matches == "unknown":
        confidence = BindingConfidence.AMBIGUOUS
    elif matches == "no":
        confidence = BindingConfidence.AMBIGUOUS
        reasons.append("runtime bytecode digest does not match the source build")

    return DeploymentBinding(
        chain_id=deployment.chain_id,
        address=deployment.address,
        contract_name=deployment.contract_name or contract,
        runtime_bytecode_digest=deployment.runtime_bytecode_digest,
        source_commit=deployment.source_commit or manifest.source_commit,
        compiler_fingerprint=manifest.compiler.fingerprint(),
        proxy=proxy,
        source_matches_deployed=matches,
        confidence=confidence,
        reasons=tuple(reasons),
    )


def _declared_topology(deployment: Deployment) -> ProxyTopology | None:
    kind = deployment.proxy_kind.strip().lower()
    if kind in {"", "none"}:
        if deployment.implementation or deployment.beacon or deployment.facets:
            kind = "generic_proxy"
        elif deployment.clone_of:
            kind = "clone"
        else:
            return None
    try:
        parsed = ProxyKind(kind)
    except ValueError:
        parsed = ProxyKind.UNKNOWN
    detail = [f"declared by the manifest as {parsed.value}"]
    if deployment.implementation:
        detail.append(f"implementation {deployment.implementation}")
    if deployment.beacon:
        detail.append(f"beacon {deployment.beacon}")
    if deployment.facets:
        detail.append(f"{len(deployment.facets)} facets")
    if deployment.clone_of:
        detail.append(f"clone of {deployment.clone_of}")
    return ProxyTopology(parsed, deployment.contract_name, tuple(detail))


def resolve_scope_identity(
    manifest: BountyManifest,
    *,
    contract: str = "",
    file: str = "",
    address: str = "",
    chain_id: str = "",
    model: ResearchModel | None = None,
) -> ScopeIdentityDecision:
    """Combine manifest scope with the deployment binding. Ambiguity never confirms scope."""

    scope = manifest.scope_of(contract=contract, file=file, address=address, chain_id=chain_id)
    binding = reconcile_deployment(
        manifest, contract=contract, address=address, chain_id=chain_id, model=model
    )

    if scope.status is ScopeStatus.OUT_OF_SCOPE:
        return ScopeIdentityDecision(
            ScopeIdentityStatus.OUT_OF_SCOPE, scope.reason, binding, scope.status.value
        )
    if scope.status is ScopeStatus.UNKNOWN:
        return ScopeIdentityDecision(
            ScopeIdentityStatus.UNKNOWN, scope.reason, binding, scope.status.value
        )
    # scope is IN_SCOPE from the manifest; the binding decides confidence.
    if binding.confidence is BindingConfidence.BOUND:
        return ScopeIdentityDecision(
            ScopeIdentityStatus.CONFIDENTLY_IN_SCOPE,
            f"{scope.reason}; deployment binding is unambiguous",
            binding,
            scope.status.value,
        )
    detail = "; ".join(binding.reasons) or "the deployment binding is not established"
    return ScopeIdentityDecision(
        ScopeIdentityStatus.AMBIGUOUS,
        f"listed in scope, but the deployment binding is ambiguous: {detail}",
        binding,
        scope.status.value,
    )


def find_deployment(
    manifest: BountyManifest, *, contract: str, address: str, chain_id: str
) -> tuple[Deployment | None, str]:
    """The single deployment a target names, or ``(None, reason)`` when ambiguous."""
    if address:
        hits = [
            item
            for item in manifest.deployments
            if item.address.lower() == address.lower()
            and (not chain_id or not item.chain_id or item.chain_id == chain_id)
        ]
        if len(hits) == 1:
            return hits[0], ""
        if len(hits) > 1:
            chains = ",".join(sorted({item.chain_id or "?" for item in hits}))
            return None, f"address {address} is deployed on several chains ({chains}); name one"
        return None, ""
    if contract:
        named = [d for d in manifest.deployments if d.contract_name == contract]
        if len(named) == 1:
            return named[0], ""
        if len(named) > 1:
            return None, (
                f"contract {contract} has {len(named)} deployments; the target address is "
                "required to choose one"
            )
    return None, ""


def _find_deployment(
    manifest: BountyManifest, *, contract: str, address: str, chain_id: str
) -> Deployment | None:
    found, _reason = find_deployment(
        manifest, contract=contract, address=address, chain_id=chain_id
    )
    return found


def _digest_match(manifest: BountyManifest, deployment: Deployment) -> str:
    deployed = deployment.runtime_bytecode_digest.strip().lower()
    expected = manifest.compiler.runtime_bytecode_digest.strip().lower()
    if not deployed or not expected:
        return "unknown"
    return "yes" if deployed == expected else "no"


def _digest_reason(manifest: BountyManifest, deployment: Deployment, matches: str) -> str:
    if matches == "yes":
        return "runtime bytecode digest matches the source build"
    if matches == "no":
        return "runtime bytecode digest differs from the source build"
    return (
        "runtime bytecode digest unknown on at least one side; source is not assumed "
        "to be the deployed code"
    )


@dataclass(frozen=True)
class DeploymentIdentity:
    """The deployment truth carried end to end: scope, findings, sequences, evidence,
    report, dedup, and runtime replay all use this one record."""

    chain_id: str
    address: str
    contract: str
    source_commit: str
    runtime_digest: str
    compiler: str
    proxy_kind: str
    implementation: str
    beacon: str
    facets: tuple[str, ...]
    clone_of: str
    binding: str  # bound | ambiguous | unbound
    source_matches_deployed: str  # yes | no | unknown
    reasons: tuple[str, ...]

    @property
    def key(self) -> str:
        """A short identity for dedup and sequence binding. ``source`` when unbound."""
        if self.address:
            return f"{self.chain_id or '?'}:{self.address.lower()}"
        return "source"

    @property
    def known(self) -> bool:
        return bool(self.address)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id or UNKNOWN,
            "address": self.address or UNKNOWN,
            "contract": self.contract or UNKNOWN,
            "source_commit": self.source_commit or UNKNOWN,
            "runtime_digest": self.runtime_digest or UNKNOWN,
            "compiler": self.compiler or UNKNOWN,
            "proxy_kind": self.proxy_kind,
            "implementation": self.implementation or UNKNOWN,
            "beacon": self.beacon or UNKNOWN,
            "facets": list(self.facets),
            "clone_of": self.clone_of or UNKNOWN,
            "binding": self.binding,
            "source_matches_deployed": self.source_matches_deployed,
            "reasons": list(self.reasons),
            "key": self.key,
        }

    def request_extra(self) -> dict[str, str]:
        """Fields for runtime/replay requests. Empty values are left out."""
        extra = {
            "deployment_address": self.address.lower(),
            "deployment_chain_id": self.chain_id,
            "deployment_binding": self.binding,
            "deployment_proxy_kind": self.proxy_kind,
            "deployment_implementation": self.implementation.lower(),
            "deployment_runtime_digest": self.runtime_digest,
        }
        return {key: value for key, value in extra.items() if value}


def deployment_identity(
    manifest: BountyManifest,
    *,
    contract: str = "",
    address: str = "",
    chain_id: str = "",
    model: ResearchModel | None = None,
) -> DeploymentIdentity:
    """Build the end-to-end deployment identity for a target. Unknown stays unknown."""
    binding = reconcile_deployment(
        manifest, contract=contract, address=address, chain_id=chain_id, model=model
    )
    deployment, _ = find_deployment(manifest, contract=contract, address=address, chain_id=chain_id)
    compiler = manifest.compiler.fingerprint()
    if deployment is not None and deployment.compiler_version:
        compiler = f"declared:{deployment.compiler_version};{compiler}".rstrip(";")
    return DeploymentIdentity(
        chain_id=binding.chain_id,
        address=binding.address,
        contract=binding.contract_name,
        source_commit=binding.source_commit,
        runtime_digest=binding.runtime_bytecode_digest,
        compiler=compiler,
        proxy_kind=binding.proxy.kind.value,
        implementation=deployment.implementation if deployment else "",
        beacon=deployment.beacon if deployment else "",
        facets=tuple(deployment.facets) if deployment else (),
        clone_of=deployment.clone_of if deployment else "",
        binding=binding.confidence.value,
        source_matches_deployed=binding.source_matches_deployed,
        reasons=binding.reasons,
    )
