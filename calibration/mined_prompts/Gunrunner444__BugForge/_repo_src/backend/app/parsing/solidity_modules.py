"""Evidence-triggered protocol modules (Phase 52, Slice D).

A module runs only when the analyzed sources contain evidence for it, and records
the evidence (file, line, construct) that triggered it. A module that is triggered
but has no dedicated detector says so (``triggered_no_detector``) instead of
reporting a clean result.

* ``erc4337``   -- ``validateUserOp`` / ``validatePaymasterUserOp`` / user-op types
  (detectors: the Phase 50 account-abstraction family);
* ``eip7702``   -- an EOA assumption used as a gate, or the 0xef0100 delegation
  designator (detector: ``eip7702.eoa_assumption`` in the high-value family);
* ``uniswap_v4_hooks`` -- hook callbacks (``beforeSwap``...) or ``IPoolManager``;
  detector ``v4_hooks.callback_without_pool_manager_check``;
* ``bridge``    -- cross-chain message receivers (``lzReceive``, ``ccipReceive``,
  ``receiveMessage``, ``handle``...); detector
  ``bridge.receiver_without_endpoint_check``;
* ``governance`` -- ``propose``/``castVote``/``queue``/``execute`` shapes or Governor /
  Timelock types; no dedicated detector yet (reported as such).

Every result is a static candidate, never verification. A receiver whose sender
check cannot be resolved (an unresolved modifier or base) is not reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.parsing.solidity_account_abstraction import account_abstraction_present
from app.parsing.solidity_caller_context import has_sender_authorization
from app.parsing.solidity_research import (
    MAX_CANDIDATES_PER_FAMILY,
    ResearchModel,
    RFunction,
    SemanticCandidate,
    skeleton,
)

FAMILY = "protocol_modules"
MAX_EVIDENCE = 6

_V4_CALLBACKS = frozenset(
    {
        "beforeInitialize",
        "afterInitialize",
        "beforeAddLiquidity",
        "afterAddLiquidity",
        "beforeRemoveLiquidity",
        "afterRemoveLiquidity",
        "beforeSwap",
        "afterSwap",
        "beforeDonate",
        "afterDonate",
        "unlockCallback",
    }
)
_BRIDGE_RECEIVERS = frozenset(
    {
        "lzReceive",
        "lzCompose",
        "ccipReceive",
        "receiveMessage",
        "handle",
        "xReceive",
        "onMessageReceived",
        "receiveWormholeMessages",
        "executeMessage",
        "finalizeInboundTransfer",
    }
)
_GOVERNANCE_NAMES = frozenset({"propose", "castVote", "queue", "execute", "cancel"})
_GOVERNANCE_TYPES = re.compile(r"\b(Governor\w*|TimelockController|ITimelock\w*)\b")
_V4_TYPES = re.compile(r"\b(IPoolManager|IHooks|BaseHook|PoolKey)\b")
_DELEGATION = re.compile(r"0xef0100", re.I)
_EOA_GATE = re.compile(
    r"msg\.sender\s*==\s*tx\.origin|tx\.origin\s*==\s*msg\.sender|\.code\.length\s*==\s*0"
    r"|extcodesize\s*\("
)
_STATE_WRITE = re.compile(r"(?<![=!<>])=(?!=)|\+=|-=|\bdelete\b|\.transfer\s*\(|\.call\s*\{")


@dataclass(frozen=True)
class ModuleStatus:
    module: str
    triggered: bool
    status: str  # ran | triggered_no_detector | not_triggered
    detectors: tuple[str, ...]
    evidence: tuple[tuple[str, int, str], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "triggered": self.triggered,
            "status": self.status,
            "detectors": list(self.detectors),
            "evidence": [
                {"file": file, "line": line, "construct": construct}
                for file, line, construct in self.evidence
            ],
        }


def _functions(model: ResearchModel) -> list[RFunction]:
    return [f for f in model.all_functions() if f.has_body or f.kind == "function"]


def _evidence_by_name(model: ResearchModel, names: frozenset[str]) -> list[tuple[str, int, str]]:
    found = []
    for function in _functions(model):
        if function.name in names:
            found.append((function.file, function.line, f"function {function.name}"))
    return found[:MAX_EVIDENCE]


def _evidence_by_text(
    model: ResearchModel, pattern: re.Pattern[str], what: str
) -> list[tuple[str, int, str]]:
    found = []
    for name in sorted(model.contracts):
        contract = model.contracts[name]
        match = pattern.search(skeleton(contract.body)) or pattern.search(contract.head)
        if match is not None:
            found.append((contract.file, contract.line, f"{what} {match.group(0)}"))
    return found[:MAX_EVIDENCE]


def module_triggers(model: ResearchModel) -> tuple[ModuleStatus, ...]:
    """Which protocol modules the evidence triggers, with the evidence."""
    statuses: list[ModuleStatus] = []

    aa = account_abstraction_present(model)
    aa_evidence = _evidence_by_name(model, frozenset({"validateUserOp", "validatePaymasterUserOp"}))
    statuses.append(
        ModuleStatus(
            "erc4337",
            aa,
            "ran" if aa else "not_triggered",
            ("aa.*",),
            tuple(aa_evidence),
        )
    )
    eip7702 = _evidence_by_text(model, _EOA_GATE, "EOA check") + _evidence_by_text(
        model, _DELEGATION, "delegation designator"
    )
    statuses.append(
        ModuleStatus(
            "eip7702",
            bool(eip7702),
            "ran" if eip7702 else "not_triggered",
            ("eip7702.eoa_assumption",),
            tuple(eip7702[:MAX_EVIDENCE]),
        )
    )
    v4 = _evidence_by_name(model, _V4_CALLBACKS) + _evidence_by_text(model, _V4_TYPES, "type")
    statuses.append(
        ModuleStatus(
            "uniswap_v4_hooks",
            bool(v4),
            "ran" if v4 else "not_triggered",
            ("v4_hooks.callback_without_pool_manager_check",),
            tuple(v4[:MAX_EVIDENCE]),
        )
    )
    bridge = _evidence_by_name(model, _BRIDGE_RECEIVERS)
    statuses.append(
        ModuleStatus(
            "bridge",
            bool(bridge),
            "ran" if bridge else "not_triggered",
            ("bridge.receiver_without_endpoint_check",),
            tuple(bridge),
        )
    )
    names = {f.name for f in _functions(model)}
    governance = (
        _evidence_by_name(model, _GOVERNANCE_NAMES) if len(names & _GOVERNANCE_NAMES) >= 2 else []
    ) + _evidence_by_text(model, _GOVERNANCE_TYPES, "type")
    statuses.append(
        ModuleStatus(
            "governance",
            bool(governance),
            "triggered_no_detector" if governance else "not_triggered",
            (),
            tuple(governance[:MAX_EVIDENCE]),
        )
    )
    return tuple(statuses)


def modules_present(model: ResearchModel) -> bool:
    return any(
        s.triggered and s.module in {"uniswap_v4_hooks", "bridge"} for s in module_triggers(model)
    )


def _unauthenticated_callbacks(
    model: ResearchModel,
    names: frozenset[str],
    detector: str,
    title: str,
    trusted: str,
    tags: tuple[str, ...],
) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for name in sorted(model.contracts):
        contract = model.contracts[name]
        if contract.kind != "contract":
            continue
        for function in model.functions_of(name, inherited=False):
            if function.name not in names or not function.exposed or not function.has_body:
                continue
            if function.mutability in {"view", "pure"}:
                continue
            if not _STATE_WRITE.search(skeleton(function.body)):
                continue  # nothing to protect: no state write, transfer or call
            authorized = has_sender_authorization(model, function)
            if authorized is not False:
                continue  # authorized, or unknowable (unresolved modifier/base): no claim
            found.append(
                SemanticCandidate(
                    detector=detector,
                    family=FAMILY,
                    title=title,
                    summary=(
                        f"{function.name} acts on its input but never checks that msg.sender "
                        f"is the {trusted}; any caller can invoke it directly"
                    ),
                    file=function.file,
                    line=function.line,
                    contract=name,
                    function=function.signature,
                    facts=(("callback", function.name), ("trusted_caller", trusted)),
                    observed=("exposed_callback", "no_sender_check", "state_effect"),
                    missing=(f"confirm the {trusted} is the only intended caller",),
                    confidence="medium",
                    impact_tags=tags,
                )
            )
            if len(found) >= MAX_CANDIDATES_PER_FAMILY:
                return found
    return found


def analyze_protocol_modules(model: ResearchModel) -> list[SemanticCandidate]:
    triggered = {s.module for s in module_triggers(model) if s.triggered}
    found: list[SemanticCandidate] = []
    if "uniswap_v4_hooks" in triggered:
        found.extend(
            _unauthenticated_callbacks(
                model,
                _V4_CALLBACKS,
                "v4_hooks.callback_without_pool_manager_check",
                "hook callback callable by anyone",
                "PoolManager",
                ("access_control", "custody"),
            )
        )
    if "bridge" in triggered:
        found.extend(
            _unauthenticated_callbacks(
                model,
                _BRIDGE_RECEIVERS,
                "bridge.receiver_without_endpoint_check",
                "cross-chain receiver callable by anyone",
                "bridge endpoint/router",
                ("bridge", "access_control", "custody"),
            )
        )
    return found[:MAX_CANDIDATES_PER_FAMILY]
