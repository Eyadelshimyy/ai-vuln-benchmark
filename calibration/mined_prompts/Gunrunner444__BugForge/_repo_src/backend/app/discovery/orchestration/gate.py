"""Scope, safety, and approval gate.

The orchestrator asks; the gate answers. A suggestion, a planner score, or a
restored state can never grant what the gate refuses. A gate that raises is a
refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.discovery.engine import AnalysisRequest
from app.discovery.orchestration.model import Candidate, ResearchState, StopReason

_NO_NETWORK = frozenset({"", "none", "off", "disabled", "no"})


@dataclass(frozen=True)
class GateVerdict:
    allowed: bool
    stop: StopReason | None = None
    detail: str = ""


class ExecutionGate(Protocol):
    def check(
        self, *, candidate: Candidate, request: AnalysisRequest, state: ResearchState
    ) -> GateVerdict: ...


@dataclass(frozen=True)
class DefaultGate:
    """Approvals come only from the caller and are never stored in campaign state."""

    scope_allowed: bool = True
    approvals: frozenset[str] = frozenset()
    approval_required: frozenset[str] = frozenset({"fork_validation"})
    network_capabilities: frozenset[str] = frozenset({"fork_validation"})

    def check(
        self, *, candidate: Candidate, request: AnalysisRequest, state: ResearchState
    ) -> GateVerdict:
        if not self.scope_allowed or request.extra.get("scope") == "out_of_scope":
            return GateVerdict(False, StopReason.SCOPE_BLOCKED, "target is outside the scope")
        network = request.extra.get("network", "").strip().lower()
        if network not in _NO_NETWORK and candidate.capability not in self.network_capabilities:
            return GateVerdict(
                False,
                StopReason.SAFETY_BLOCKED,
                "network access is only possible for approved fork validation",
            )
        if candidate.capability in self.approval_required and (
            candidate.capability not in self.approvals
        ):
            return GateVerdict(
                False, StopReason.APPROVAL_REQUIRED, f"{candidate.capability} needs approval"
            )
        return GateVerdict(True)


def check_gate(
    gate: ExecutionGate,
    *,
    candidate: Candidate,
    request: AnalysisRequest,
    state: ResearchState,
) -> GateVerdict:
    try:
        verdict = gate.check(candidate=candidate, request=request, state=state)
    except Exception:
        return GateVerdict(False, StopReason.SAFETY_BLOCKED, "the gate failed, so it refuses")
    if not isinstance(verdict, GateVerdict):
        return GateVerdict(False, StopReason.SAFETY_BLOCKED, "the gate gave no verdict")
    if not verdict.allowed and verdict.stop is None:
        return GateVerdict(False, StopReason.SAFETY_BLOCKED, verdict.detail or "refused")
    return verdict
