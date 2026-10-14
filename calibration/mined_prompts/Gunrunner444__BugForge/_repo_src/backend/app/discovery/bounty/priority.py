"""Program-aware, impact-first prioritization.

The result is a triage score. It orders what a researcher should read first and it
says nothing about whether anything is exploitable. Impact weights come from the
program's own policy; without a policy every signal weighs the same and the severity
stays unknown. Signals are structural facts read from the code with their evidence,
never function names.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from app.discovery.bounty.campaign import (
    BountyManifest,
    ImpactCategory,
    PocRequirement,
    ScopeStatus,
)
from app.parsing.solidity_research import (
    ResearchModel,
    RFunction,
    SemanticCandidate,
    member_calls,
    plain_calls,
)

DISCLAIMER = "triage score only; it is not a statement about exploitability"
MAX_ENTRIES = 64
MAX_SIGNAL_EVIDENCE = 3
MAX_CATEGORIES_SCORED = 3
UNIFORM_SIGNAL_WEIGHT = 10
POC_FEASIBLE_BONUS = 10
POC_BLOCKED_PENALTY = 20
EXPOSED_BONUS = 5
CANDIDATE_BONUS = 8

SIGNALS: tuple[str, ...] = (
    "value_transfer",
    "custody",
    "vault_accounting",
    "mint_burn",
    "privileged_operations",
    "upgrade",
    "arbitrary_calls",
    "router_dispatch",
    "oracle_dependent_value",
    "bridge_message_verification",
    "signature_authorization",
    "account_abstraction_validation",
    "liquidation",
    "collateral_valuation",
    "reserve_accounting",
    "governance",
    "permissions_roles",
    "token_approvals",
)

_TRANSFER_CALLS = frozenset(
    {"transfer", "safeTransfer", "transferFrom", "safeTransferFrom", "send", "sendValue"}
)
_MINT_CALLS = frozenset({"mint", "_mint", "burn", "_burn", "burnFrom"})
_ORACLE_CALLS = frozenset(
    {"latestRoundData", "latestAnswer", "getPrice", "consult", "slot0", "getReserves", "peek"}
)
_APPROVAL_CALLS = frozenset(
    {"approve", "safeApprove", "forceApprove", "increaseAllowance", "permit", "allowance"}
)
_SIGNATURE_CALLS = frozenset({"recover", "tryRecover", "isValidSignature", "ecrecover"})
_PROOF_CALLS = frozenset({"verify", "verifyCalldata", "processProof", "verifyProof"})
_GOVERNANCE_CALLS = frozenset({"getPastVotes", "getVotes", "quorum", "propose", "castVote"})
_ROLE_CALLS = frozenset({"hasRole", "grantRole", "revokeRole", "_grantRole", "_setupRole"})
_UPGRADE_CALLS = frozenset({"upgradeTo", "upgradeToAndCall", "_authorizeUpgrade", "_upgradeTo"})
_LIQUIDATION_MARKERS = re.compile(r"\b(?:healthFactor|isLiquidatable|seize\w*|liquidat\w+)\s*\(")
_SENDER_GUARD = re.compile(r"\bmsg\.sender\s*(?:==|!=)|(?:==|!=)\s*msg\.sender\b")
_STATE_ASSIGN = re.compile(r"\b(?:owner|admin|governance|guardian|operator)\w*\s*=(?!=)")


@dataclass(frozen=True)
class Signal:
    name: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class PocFeasibility:
    status: str  # feasible | unknown | blocked
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PriorityEntry:
    identity: str
    contract: str
    function: str
    file: str
    line: int
    score: int
    components: tuple[tuple[str, int], ...]
    signals: tuple[Signal, ...]
    categories: tuple[str, ...]
    detectors: tuple[str, ...]
    scope: str
    poc: PocFeasibility
    explanation: str
    disclaimer: str = DISCLAIMER


@dataclass(frozen=True)
class PriorityResult:
    ranked: tuple[PriorityEntry, ...]
    excluded_out_of_scope: tuple[str, ...]
    policy: str  # program | uniform
    truncated: bool
    disclaimer: str = DISCLAIMER


def signals_of(model: ResearchModel, function: RFunction) -> tuple[Signal, ...]:
    """Structural impact signals with the evidence that produced each one."""
    found: dict[str, list[str]] = {}

    def add(name: str, evidence: str) -> None:
        bucket = found.setdefault(name, [])
        text = " ".join(evidence.split())[:120]
        if text not in bucket and len(bucket) < MAX_SIGNAL_EVIDENCE:
            bucket.append(text)

    body = function.body
    calls = member_calls(body)
    plains = plain_calls(body)
    names = {call.name for call in calls} | {name for name, _args, _off in plains}
    param_names = set(function.param_names())

    for call in calls:
        if call.name in _TRANSFER_CALLS or (call.name == "call" and "value" in call.options):
            add("value_transfer", call.text)
        if call.name in _MINT_CALLS:
            add("mint_burn", call.text)
        if call.name in _ORACLE_CALLS:
            add("oracle_dependent_value", call.text)
        if call.name in _APPROVAL_CALLS:
            add("token_approvals", call.text)
        if call.name in _SIGNATURE_CALLS:
            add("signature_authorization", call.text)
        if call.name in _PROOF_CALLS:
            add("bridge_message_verification", call.text)
        if call.name in _GOVERNANCE_CALLS:
            add("governance", call.text)
        if call.name in _ROLE_CALLS:
            add("permissions_roles", call.text)
        if call.name == "delegatecall":
            add("upgrade", call.text)
            add("arbitrary_calls", call.text)
        if call.name in {"call", "delegatecall", "staticcall"} and call.receiver:
            if call.receiver in param_names or any(
                call.receiver.startswith(f"{p}.") for p in param_names
            ):
                add("arbitrary_calls", call.text)
            if any(arg.split("[")[0].strip() in param_names for arg in call.arguments):
                add("router_dispatch", call.text)
        if call.name == "balanceOf":
            add("custody", call.text)
    for name, _args, _offset in plains:
        if name in _MINT_CALLS:
            add("mint_burn", f"{name}(...)")
        if name in _UPGRADE_CALLS:
            add("upgrade", f"{name}(...)")
        if name in _ROLE_CALLS:
            add("permissions_roles", f"{name}(...)")
        if name == "ecrecover":
            add("signature_authorization", "ecrecover(...)")
        if name in _SIGNATURE_CALLS - {"ecrecover"}:
            add("signature_authorization", f"{name}(...)")
    if names & _UPGRADE_CALLS:
        add("upgrade", "upgrade entry point")
    if any("UserOperation" in item.type_name for item in function.params):
        add("account_abstraction_validation", f"parameter {function.params[0].type_name}")
    if liquidation := _LIQUIDATION_MARKERS.search(body):
        add("liquidation", liquidation.group(0))
    if "value_transfer" in found and "custody" in found:
        add("vault_accounting", "transfer and balance read in one function")
    if "oracle_dependent_value" in found and (
        "liquidation" in found or re.search(r"\b(?:collateral|borrow|debt)\w*", body)
    ):
        add("collateral_valuation", "oracle value used beside collateral or debt")
    if re.search(r"\b(?:reserve\w*|totalAssets|totalSupply|totalShares)\b", body) and (
        "value_transfer" in found or "mint_burn" in found
    ):
        add("reserve_accounting", "reserve or supply read beside a value movement")
    if "mint_burn" in found and re.search(r"\b(?:totalAssets|totalSupply|totalShares)\b", body):
        add("vault_accounting", "mint or burn priced by a supply or asset total")
    if _SENDER_GUARD.search(body) or _unresolved_or_guarded(model, function):
        add("privileged_operations", _guard_evidence(model, function))
    if assignment := _STATE_ASSIGN.search(body):
        add("permissions_roles", assignment.group(0))
    if "transferFrom" in names or "safeTransferFrom" in names:
        add("custody", "pulls tokens from a holder")
    return tuple(Signal(name, tuple(found[name])) for name in SIGNALS if name in found)


def _unresolved_or_guarded(model: ResearchModel, function: RFunction) -> bool:
    return any(model.modifier(function.contract, name) is not None for name in function.modifiers)


def _guard_evidence(model: ResearchModel, function: RFunction) -> str:
    for name in function.modifiers:
        modifier = model.modifier(function.contract, name)
        if modifier is not None and _SENDER_GUARD.search(modifier.body):
            return f"modifier {name} checks msg.sender"
    match = _SENDER_GUARD.search(function.body)
    return match.group(0) if match else "guarded entry point"


def poc_feasibility(model: ResearchModel, function: RFunction) -> PocFeasibility:
    """Whether a runnable proof of concept could even reach this function from the model."""
    contract = model.contracts.get(function.contract)
    reasons: list[str] = []
    if not function.has_body or contract is None or contract.kind in {"interface", "library"}:
        return PocFeasibility("blocked", ("no concrete body to execute",))
    if function.contract in model.ambiguous:
        return PocFeasibility("unknown", ("the contract name is ambiguous in the sources",))
    if not function.exposed:
        reasons.append("not directly reachable from outside the contract")
    unresolved = [
        name for name in function.modifiers if model.modifier(function.contract, name) is None
    ]
    if unresolved:
        reasons.append(f"unresolved modifier {unresolved[0]}")
    if reasons:
        return PocFeasibility("unknown", tuple(reasons))
    return PocFeasibility("feasible", ("externally reachable with a resolved body",))


def prioritize(
    model: ResearchModel,
    *,
    manifest: BountyManifest | None = None,
    candidates: Iterable[SemanticCandidate] = (),
) -> PriorityResult:
    categories = manifest.impact_categories if manifest is not None else ()
    policy = "program" if categories else "uniform"
    by_function: dict[str, list[SemanticCandidate]] = {}
    for item in candidates:
        by_function.setdefault(f"{item.contract}.{item.function}", []).append(item)

    entries: list[PriorityEntry] = []
    excluded: list[str] = []
    for function in model.all_functions():
        if not function.has_body or function.kind not in {"function", "receive", "fallback"}:
            continue
        signals = signals_of(model, function)
        related = by_function.get(function.identity, [])
        tags = {s.name for s in signals} | {tag for item in related for tag in item.impact_tags}
        if not tags and not related:
            continue
        decision = (
            manifest.scope_of(contract=function.contract, file=function.file)
            if manifest is not None
            else None
        )
        scope = decision.status.value if decision is not None else ScopeStatus.UNKNOWN.value
        if decision is not None and decision.status is ScopeStatus.OUT_OF_SCOPE:
            excluded.append(function.identity)
            continue
        entries.append(_entry(model, function, signals, related, tags, categories, scope, manifest))

    entries.sort(key=lambda item: (-item.score, item.identity, item.line))
    return PriorityResult(
        ranked=tuple(entries[:MAX_ENTRIES]),
        excluded_out_of_scope=tuple(sorted(excluded)),
        policy=policy,
        truncated=len(entries) > MAX_ENTRIES or model.truncated,
    )


def _entry(
    model: ResearchModel,
    function: RFunction,
    signals: tuple[Signal, ...],
    related: list[SemanticCandidate],
    tags: set[str],
    categories: tuple[ImpactCategory, ...],
    scope: str,
    manifest: BountyManifest | None,
) -> PriorityEntry:
    matched = _matched_categories(categories, tags)
    components: dict[str, int] = {}
    if categories:
        top = sorted(matched, key=lambda item: (-item.weight, item.name))[:MAX_CATEGORIES_SCORED]
        components["program_impact"] = min(100, sum(item.weight for item in top))
    else:
        components["signals"] = min(100, UNIFORM_SIGNAL_WEIGHT * len(tags))
    if related:
        components["candidate_evidence"] = CANDIDATE_BONUS * min(len(related), 3)
    if function.exposed:
        components["externally_reachable"] = EXPOSED_BONUS
    poc = poc_feasibility(model, function)
    if manifest is not None and manifest.poc_requirement is PocRequirement.REQUIRED:
        if poc.status == "feasible":
            components["poc_feasible"] = POC_FEASIBLE_BONUS
        elif poc.status == "blocked":
            components["poc_blocked"] = -POC_BLOCKED_PENALTY
    score = max(0, sum(components.values()))
    explanation = _explain(matched, signals, related, categories)
    return PriorityEntry(
        identity=function.identity,
        contract=function.contract,
        function=function.signature,
        file=function.file,
        line=function.line,
        score=score,
        components=tuple(sorted(components.items())),
        signals=signals,
        categories=tuple(sorted(item.name for item in matched)),
        detectors=tuple(sorted({item.detector for item in related})),
        scope=scope,
        poc=poc,
        explanation=explanation,
    )


def _matched_categories(
    categories: tuple[ImpactCategory, ...], tags: set[str]
) -> list[ImpactCategory]:
    return [item for item in categories if tags & set(item.tags)]


def _explain(
    matched: list[ImpactCategory],
    signals: tuple[Signal, ...],
    related: list[SemanticCandidate],
    categories: tuple[ImpactCategory, ...],
) -> str:
    parts: list[str] = []
    if signals:
        parts.append("signals: " + ", ".join(item.name for item in signals[:6]))
    if matched:
        parts.append("program impact: " + ", ".join(sorted(item.name for item in matched)[:4]))
    elif categories:
        parts.append("no program impact category matches these signals")
    else:
        parts.append("the program supplied no impact policy, so signals weigh equally")
    if related:
        parts.append(f"{len(related)} static candidate(s)")
    return "; ".join(parts)[:300]
