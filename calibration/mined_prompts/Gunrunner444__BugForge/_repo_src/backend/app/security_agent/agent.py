"""Guided SecurityResearchAgent. Planner only — not scope/verify/approve authority."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.ai.external_controller import ExternalControllerProvider
from app.ai.provider import CompletionRequest, LLMProvider
from app.domain.findings import SecurityFinding
from app.security_agent.budget import SessionBudget
from app.security_agent.context import ContextManager
from app.security_agent.correlation import correlate_all, finding_fingerprint
from app.security_agent.correlation import prioritize as rank_hypotheses
from app.security_agent.evidence_graph import EvidenceGraph
from app.security_agent.executors import ToolContext, bind_engine_tools
from app.security_agent.injection import (
    channel,
    contains_injection_attempt,
    strip_instruction_attempts,
    untrusted_observation,
)
from app.security_agent.privilege import PrivilegeSnapshot, capture_privileges
from app.security_agent.promotion import apply_reproduction, promote_hypothesis
from app.security_agent.schemas import (
    PlannerOutput,
    ResearchHypothesis,
    ResearchPlan,
    ResearchPlanStep,
    ToolCallRequest,
)
from app.security_agent.states import (
    RESUME_BLOCKED_STATES,
    TERMINAL_RESEARCH_STATES,
    EvidenceGraphKind,
    HypothesisStatus,
    ReproductionOutcome,
    ResearchController,
    ResearchMode,
    ResearchState,
    TerminationReason,
    ToolAuthorization,
    ToolCapability,
    ToolResultQuality,
    ToolRiskLevel,
)
from app.security_agent.tools import ToolRegistry, default_registry
from app.security_testing.approvals import ApprovalKind
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.errors import (
    ApprovalRequiredError,
    AuthorizationDeniedError,
    RestrictedActivityError,
    SafetyLimitExceededError,
)
from app.security_testing.secrets import redact_text

Planner = Callable[["ResearchSession"], Awaitable["AgentDecision"]]

_FORBIDDEN_AI_ACTIONS = frozenset(
    {
        "mark_verified",
        "verify_finding",
        "approve_report",
        "submit_hackerone",
        "grant_budget",
        "enable_tool",
        "change_scope",
        "declare_in_scope",
        "enable_active_testing",
        "grant_approval",
        "increase_budget",
        "submit_report",
    }
)

_RETRYABLE_QUALITY = frozenset(
    {
        ToolResultQuality.FAILED.value,
        ToolResultQuality.TIMEOUT.value,
        ToolResultQuality.BLOCKED.value,
        ToolResultQuality.UNAVAILABLE.value,
    }
)


@dataclass
class AgentDecision:
    kind: str
    tool: ToolCallRequest | None = None
    hypothesis: ResearchHypothesis | None = None
    plan: ResearchPlan | None = None
    note: str = ""
    thinking: str | None = None


@dataclass
class TimelineEvent:
    event_type: str
    decision: str = ""
    tool: str = ""
    target: str = ""
    authorization: str = ""
    result: str = ""
    evidence_id: str = ""
    finding_id: str = ""
    strategy: str = ""
    state: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def snapshot(self) -> dict[str, Any]:
        return {
            "time": self.created_at.isoformat(),
            "timestamp": self.created_at.isoformat(),
            "event_type": self.event_type,
            "decision": self.decision,
            "tool": self.tool,
            "target": self.target,
            "authorization": self.authorization,
            "result": self.result,
            "evidence_id": self.evidence_id,
            "finding_id": self.finding_id,
            "strategy": self.strategy,
            "state": self.state,
        }


@dataclass
class FingerprintRecord:
    fingerprint: str
    at: datetime
    quality: str


@dataclass
class ToolCallRecord:
    tool: str
    arguments: dict[str, Any]
    reason: str = ""
    authorization: str = ""
    authorization_reason: str = ""
    execution_state: str = ""
    result_quality: str = ""
    result_summary: str = ""
    evidence_ids: tuple[str, ...] = ()
    error: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "reason": self.reason,
            "authorization": self.authorization,
            "authorization_reason": self.authorization_reason,
            "execution_state": self.execution_state,
            "result_quality": self.result_quality,
            "result_summary": self.result_summary,
            "evidence_ids": list(self.evidence_ids),
            "error": self.error,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ResearchSession:
    project_id: str
    target: str
    mode: ResearchMode
    engine: SecurityTestEngine
    provider: LLMProvider
    program_handle: str = ""
    state: ResearchState = ResearchState.CREATED
    model_name: str = ""
    thinking_enabled: bool = True
    budget: SessionBudget = field(default_factory=SessionBudget.from_settings)
    hypotheses: list[ResearchHypothesis] = field(default_factory=list)
    plan: ResearchPlan | None = None
    graph: EvidenceGraph = field(default_factory=EvidenceGraph)
    timeline: list[TimelineEvent] = field(default_factory=list)
    tool_history: list[str] = field(default_factory=list)
    fingerprints: list[str] = field(default_factory=list)
    fingerprint_records: list[FingerprintRecord] = field(default_factory=list)
    tool_call_records: list[ToolCallRecord] = field(default_factory=list)
    findings: list[SecurityFinding] = field(default_factory=list)
    disabled_tools: set[str] = field(default_factory=set)
    paused: bool = False
    stopped: bool = False
    error: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    termination_reason: TerminationReason | None = None
    repo_root: str = "."
    exchanges: dict[str, dict[str, Any]] = field(default_factory=dict)
    privilege: PrivilegeSnapshot | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    step_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    zap_runner: Any = None
    nuclei_runner: Any = None
    zap_binary: str | None = None
    nuclei_binary: str | None = None
    next_action: dict[str, Any] | None = None
    identities: Any = None
    strategy: str = "passive_recon"
    memory: Any = None
    replay_mode: bool = False
    operator_identity: str = ""
    research_project_id: str = ""
    controller: ResearchController = ResearchController.INTERNAL_LLM
    exploratory_attempts: list[Any] = field(default_factory=list)
    leads: list[Any] = field(default_factory=list)
    chains: list[Any] = field(default_factory=list)
    research_route: str = "wide"

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "program_handle": self.program_handle,
            "target": self.target,
            "mode": self.mode.value,
            "controller": self.controller.value,
            "ai_controller": (
                "cursor" if self.controller is ResearchController.CURSOR else "bugforge_llm"
            ),
            "ai_execution": (
                "none"
                if self.controller is ResearchController.CURSOR
                else self.provider.provider_name
            ),
            "provider": self.provider.provider_name,
            "state": self.state.value,
            "termination_reason": self.termination_reason.value
            if self.termination_reason
            else None,
            "model": self.model_name or self.provider.model_name,
            "budget": self.budget.snapshot(),
            "hypotheses": [item.snapshot() for item in self.hypotheses],
            "plan": self.plan.snapshot() if self.plan else None,
            "timeline": [item.snapshot() for item in self.timeline],
            "error": self.error,
            "paused": self.paused,
            "stopped": self.stopped,
            "graph": self.graph.snapshot(),
            "disabled_tools": sorted(self.disabled_tools),
            "next_action": dict(self.next_action) if self.next_action else None,
            "findings": [
                {
                    "id": str(item.id),
                    "title": item.title,
                    "status": item.status.value,
                    "target": item.target,
                    "vulnerability_class": item.vulnerability_class,
                }
                for item in self.findings
            ],
            "tool_history": [item.snapshot() for item in self.tool_call_records],
            "strategy": self.strategy,
            "operator_identity": self.operator_identity,
            "replay_mode": self.replay_mode,
            "research_project_id": self.research_project_id,
            "research_route": self.research_route,
            "leads": [item.snapshot() for item in self.leads if hasattr(item, "snapshot")],
            "chains": [item.snapshot() for item in self.chains if hasattr(item, "snapshot")],
        }

    def cancelled(self) -> bool:
        return self.cancel_event.is_set() or self.stopped or self.paused


class SecurityResearchAgent:
    """Observe → analyze → hypothesize → plan → request tool → authorize → execute."""

    def __init__(
        self,
        session: ResearchSession,
        *,
        tools: ToolRegistry | None = None,
        planner: Planner | None = None,
    ) -> None:
        self.session = session
        session.graph.session_id = session.id
        session.graph.project_id = session.project_id
        self.tools = tools or bind_engine_tools(_tool_context(session), default_registry())
        self.planner = planner or self._llm_planner
        if (
            session.controller is ResearchController.CURSOR
            or session.provider.provider_name == "cursor_external"
        ):
            session.controller = ResearchController.CURSOR
            if not isinstance(session.provider, ExternalControllerProvider):
                session.provider = ExternalControllerProvider(
                    model_name=session.model_name or "cursor-selected-model"
                )
            session.model_name = session.provider.model_name
            self.planner = self._refuse_internal_planner
        self.context = ContextManager()
        self._refresh_privilege()

    def pause(self, reason: str = "operator") -> None:
        self.session.paused = True
        self.session.cancel_event.set()
        self.session.state = ResearchState.USER_PAUSED
        self.session.termination_reason = TerminationReason.USER_PAUSED
        self._timeline("pause", decision=f"PAUSE:{reason}")
        self._timeline("human_override", decision=f"PAUSE:{reason}")

    def stop(self, reason: str = "operator") -> None:
        self.session.stopped = True
        self.session.cancel_event.set()
        self.session.state = ResearchState.USER_STOPPED
        self.session.termination_reason = TerminationReason.USER_STOPPED
        self._timeline("stop", decision=f"STOP:{reason}")
        self._timeline("human_override", decision=f"STOP:{reason}")
        from app.security_testing.exploratory import release_exploratory_engine

        release_exploratory_engine(self.session.id)

    def reject_action(self, reason: str) -> None:
        self._timeline("human_override", decision=f"REJECT_ACTION:{reason}")

    def disable_tool(self, name: str) -> None:
        self.session.disabled_tools.add(name)
        self.tools.disable(name)
        self._timeline("human_override", decision=f"DISABLE_TOOL:{name}")

    def reject_finding(self, finding_id: str) -> None:
        from app.domain.findings import SecurityFinding as DomainFinding

        updated: list[Any] = []
        for item in self.session.findings:
            if str(item.id) == str(finding_id):
                updated.append(
                    DomainFinding.rejected(
                        item.title,
                        evidence=item.evidence,
                        vulnerability_class=item.vulnerability_class,
                        target=item.target,
                        hypothesis=item.hypothesis,
                        id=item.id,
                    )
                )
            else:
                updated.append(item)
        self.session.findings = updated
        self._timeline(
            "human_override", decision=f"REJECT_FINDING:{finding_id}", finding_id=finding_id
        )

    def change_limits(self, *, operator: str, extra_tool_calls: int = 0) -> None:
        self.session.budget.grant_more(operator=operator, extra_tool_calls=extra_tool_calls)
        self._timeline("approval", decision=f"CHANGE_LIMITS:{operator}")
        self._timeline("human_override", decision=f"CHANGE_LIMITS:{operator}")

    def enable_active_testing(self, *, operator: str, note: str = "") -> None:
        from app.security_testing.approvals import is_ai_operator

        if is_ai_operator(operator):
            raise RestrictedActivityError("The AI cannot enable active testing")
        self.session.engine.enable_active_testing(operator=operator, note=note)
        self._refresh_privilege()
        self._timeline("approval", decision=f"ENABLE_ACTIVE_TESTING:{operator}")

    def grant_tool_approval(self, kind: ApprovalKind, *, operator: str, note: str = "") -> None:
        from app.security_testing.approvals import is_ai_operator

        if is_ai_operator(operator):
            raise RestrictedActivityError("The AI cannot grant tool approvals")
        self.session.engine.grant(kind, operator=operator, note=note)
        self._refresh_privilege()
        self._timeline("approval", decision=f"{kind.value}:{operator}")

    def resume(self, *, operator: str) -> ResearchSession:
        from app.security_testing.approvals import is_ai_operator

        if is_ai_operator(operator):
            raise RestrictedActivityError("The AI cannot resume a research session")
        if self.session.state in RESUME_BLOCKED_STATES or self.session.stopped:
            raise RestrictedActivityError("session_resume_requires_operator_reopen")
        self.session.paused = False
        self.session.cancel_event.clear()
        self.session.termination_reason = None
        self.session.state = ResearchState.ANALYZING
        self._timeline("session_state", decision=f"RESUME:{operator}")
        return self.session

    async def run(self, *, max_steps: int | None = None) -> ResearchSession:
        limit = max_steps if max_steps is not None else self.session.budget.max_iterations
        steps = 0
        for _ in range(limit):
            if self.session.stopped or self.session.paused:
                break
            if self.session.state in TERMINAL_RESEARCH_STATES:
                break
            await self.step()
            steps += 1
        self._finalize_run(steps=steps, limit=limit)
        return self.session

    def _finalize_run(self, *, steps: int, limit: int) -> None:
        if self.session.termination_reason is not None:
            return
        if self.session.stopped:
            self.session.termination_reason = TerminationReason.USER_STOPPED
            self.session.state = ResearchState.USER_STOPPED
            return
        if self.session.paused:
            self.session.termination_reason = TerminationReason.USER_PAUSED
            self.session.state = ResearchState.USER_PAUSED
            return
        if self.session.state is ResearchState.FAILED:
            self.session.termination_reason = TerminationReason.FAILED
            return
        if self.session.budget.iterations >= self.session.budget.max_iterations or steps >= limit:
            self.session.termination_reason = TerminationReason.MAX_ITERATIONS
            self.session.state = ResearchState.MAX_ITERATIONS
            return
        verified = any(
            item.is_verified
            and item.evidence.verifying_items()
            and not any(
                getattr(ev.provenance, "value", str(ev.provenance)) == "replay"
                for ev in item.evidence.items
            )
            for item in self.session.findings
        )
        if verified:
            self.session.termination_reason = TerminationReason.COMPLETED_SUCCESS
            self.session.state = ResearchState.COMPLETED_SUCCESS
            return
        if self.session.findings or self.session.hypotheses:
            self.session.termination_reason = TerminationReason.INCONCLUSIVE
            self.session.state = ResearchState.INCONCLUSIVE
            return
        self.session.termination_reason = TerminationReason.COMPLETED_NO_FINDINGS
        self.session.state = ResearchState.COMPLETED_NO_FINDINGS

    async def step(self) -> AgentDecision:
        async with self.session.step_lock:
            return await self._step_locked()

    async def _step_locked(self) -> AgentDecision:
        self._refuse_if_cursor_controlled()
        if self.session.stopped:
            return AgentDecision(kind="stopped")
        if self.session.paused:
            return AgentDecision(kind="paused")
        try:
            self.session.budget.consume("iteration")
        except SafetyLimitExceededError as exc:
            self.session.state = ResearchState.BUDGET_EXHAUSTED
            self.session.termination_reason = TerminationReason.BUDGET_EXHAUSTED
            self.session.error = str(exc)
            self._timeline("session_state", decision="BUDGET_EXHAUSTED", result=str(exc))
            return AgentDecision(kind="budget_exhausted", note=str(exc))
        decision = await self.planner(self.session)
        return await self._apply_decision(decision)

    def _refuse_if_cursor_controlled(self) -> None:
        if self.session.controller is not ResearchController.CURSOR:
            return
        self._timeline(
            "blocked",
            decision="cursor_mode_refuses_internal_planner",
            authorization="blocked",
        )
        raise RestrictedActivityError("cursor_mode_refuses_internal_planner")

    async def _refuse_internal_planner(self, session: ResearchSession) -> AgentDecision:
        del session
        raise RestrictedActivityError("cursor_mode_refuses_internal_planner")

    async def apply_external_decision(self, raw: dict[str, Any]) -> AgentDecision:
        """Apply a Cursor decision. The payload is untrusted planner output.

        This path never calls ``provider.complete``. Forbidden kinds are
        rejected by ``PlannerOutput`` before any state change.
        """

        if self.session.controller is not ResearchController.CURSOR:
            raise RestrictedActivityError("external_decision_requires_cursor_controller")
        if not isinstance(self.session.provider, ExternalControllerProvider):
            raise RestrictedActivityError("cursor_session_provider_mismatch")
        async with self.session.step_lock:
            if self.session.stopped:
                return AgentDecision(kind="stopped")
            if self.session.paused:
                return AgentDecision(kind="paused")
            try:
                self.session.budget.consume("iteration")
            except SafetyLimitExceededError as exc:
                self.session.state = ResearchState.BUDGET_EXHAUSTED
                self.session.termination_reason = TerminationReason.BUDGET_EXHAUSTED
                self.session.error = str(exc)
                self._timeline("session_state", decision="BUDGET_EXHAUSTED", result=str(exc))
                return AgentDecision(kind="budget_exhausted", note=str(exc))
            try:
                payload = PlannerOutput.model_validate(raw)
            except ValidationError as exc:
                self._timeline("external_decision", decision=f"rejected_malformed:{exc}")
                raise RestrictedActivityError("rejected_external_decision") from exc
            self._timeline(
                "external_decision",
                decision=payload.kind,
                result="cursor_external",
            )
            try:
                if payload.kind == "reproduce":
                    note = strip_instruction_attempts(payload.reason or payload.note)
                    decision = AgentDecision(
                        kind="tool",
                        tool=ToolCallRequest(
                            tool="reproduce",
                            arguments=dict(payload.arguments or {}),
                            reason=note,
                        ),
                        note=note,
                    )
                else:
                    decision = self._decision_from_planner(payload, thinking=None)
            except ValueError as exc:
                self._timeline("external_decision", decision=f"rejected_malformed:{exc}")
                raise RestrictedActivityError("rejected_external_decision") from exc
            return await self._apply_decision(decision)

    async def _apply_decision(self, decision: AgentDecision) -> AgentDecision:
        if decision.thinking:
            self._timeline("thinking", decision="thinking discarded as non-evidence")
        if decision.kind in _FORBIDDEN_AI_ACTIONS:
            self._timeline("blocked", decision=decision.kind, authorization="blocked")
            raise RestrictedActivityError(decision.kind)
        if decision.kind == "hypothesis" and decision.hypothesis:
            self._record_hypothesis(decision.hypothesis)
            return decision
        if decision.kind == "update_hypothesis" and decision.hypothesis:
            self._record_hypothesis(decision.hypothesis, update=True)
            return decision
        if decision.kind == "plan" and decision.plan:
            self.session.plan = decision.plan
            self.session.state = ResearchState.PLAN_READY
            self._timeline("plan", decision=f"{len(decision.plan.steps)} steps")
            self._timeline("model_decision", decision="plan")
            return decision
        if decision.kind == "tool" and decision.tool:
            await self._execute_tool(decision.tool, reason=decision.note)
            return decision
        if decision.kind == "reproduce":
            self.session.state = ResearchState.REPRODUCING
            self._timeline("reproduction", decision=decision.note or "reproduce")
            return decision
        if decision.kind == "complete":
            self._finalize_run(
                steps=self.session.budget.iterations, limit=self.session.budget.max_iterations
            )
            return decision
        self.session.state = ResearchState.ANALYZING
        self._timeline("model_decision", decision=decision.kind or "analyze")
        return decision

    async def request_tool(self, request: ToolCallRequest) -> dict[str, Any]:
        async with self.session.step_lock:
            return await self._execute_tool(request, reason=request.reason)

    async def _execute_tool(self, request: ToolCallRequest, *, reason: str) -> dict[str, Any]:
        fingerprint = f"{request.tool}:{json.dumps(request.arguments, sort_keys=True, default=str)}"
        if request.tool != "exploratory_test":
            self._check_duplicate(fingerprint)
        if request.tool in self.session.disabled_tools:
            raise RestrictedActivityError(f"disabled_tool:{request.tool}")
        parsed = self.tools.validate(request)
        spec = self.tools.spec(request.tool)
        dumped = parsed.model_dump()
        target = str(dumped.get("url") or dumped.get("target") or self.session.target)
        self._timeline("tool_request", decision=reason, tool=request.tool, target=target)
        request_node = self.session.graph.add(
            kind=EvidenceGraphKind.TOOL_REQUEST.value,
            provenance="tool_request",
            summary=f"{request.tool} {target}",
            source=request.tool,
            extra={"reason": reason},
        )
        try:
            self.tools.validate_capability(request.tool, mode=self.session.mode)
        except RestrictedActivityError as exc:
            if spec.capability is ToolCapability.UNAVAILABLE:
                unavailable = ToolResultQuality.UNAVAILABLE.value
                payload = {"quality": unavailable, "reason": str(exc), "executed": False}
                self._remember_fingerprint(fingerprint, unavailable)
                self._timeline(
                    "authorization",
                    tool=request.tool,
                    target=target,
                    authorization="blocked",
                    result=unavailable,
                    evidence_id=request_node.id,
                )
                return payload
            raise
        if spec.capability is ToolCapability.PLANNING_ONLY:
            payload = {
                "quality": ToolResultQuality.NO_RESULT.value,
                "capability": ToolCapability.PLANNING_ONLY.value,
                "executed": False,
                "reason": "planning_only",
            }
            self._remember_fingerprint(fingerprint, ToolResultQuality.NO_RESULT.value)
            return payload
        auth = self._authorize_target(target, tool=request.tool, spec=spec, arguments=dumped)
        if auth is ToolAuthorization.BLOCKED:
            self._timeline(
                "authorization",
                decision=reason,
                tool=request.tool,
                target=target,
                authorization="blocked",
                result="BLOCKED",
            )
            self._remember_fingerprint(fingerprint, ToolResultQuality.BLOCKED.value)
            return {
                "authorization": "BLOCKED",
                "quality": ToolResultQuality.BLOCKED.value,
                "reason": "ScopeGuard denied the target",
                "executed": False,
            }
        if auth is ToolAuthorization.APPROVAL_REQUIRED:
            self.session.state = ResearchState.WAITING_FOR_APPROVAL
            self._timeline(
                "authorization",
                tool=request.tool,
                target=target,
                authorization="approval_required",
                result="APPROVAL_REQUIRED",
            )
            self._remember_fingerprint(fingerprint, ToolResultQuality.APPROVAL_REQUIRED.value)
            return {
                "authorization": "APPROVAL_REQUIRED",
                "quality": ToolResultQuality.APPROVAL_REQUIRED.value,
                "reason": spec.approval_kind or "human approval required",
                "executed": False,
            }
        try:
            from app.security_agent.cost import consume_units_for, estimate_operation_cost

            remaining = int(self.session.budget.remaining().get("requests") or 0)
            estimate = estimate_operation_cost(
                request.tool, dumped, spec, remaining_requests=remaining
            )
            self.session.budget.consume("tool")
            kind, amount = consume_units_for(estimate)
            if kind != "tool" and amount:
                self.session.budget.consume(kind, amount=amount)
        except SafetyLimitExceededError as exc:
            self.session.state = ResearchState.BUDGET_EXHAUSTED
            self.session.termination_reason = TerminationReason.BUDGET_EXHAUSTED
            self._timeline("tool", tool=request.tool, authorization="blocked", result=str(exc))
            raise
        self.session.state = ResearchState.EXECUTING
        self.session.next_action = {
            "tool": request.tool,
            "target": target,
            "reason": reason,
            "expected_evidence": spec.description,
            "estimated_requests": estimate.estimated_requests,
            "estimated_units": estimate.estimated_units,
            "estimate_unit": estimate.unit,
            "is_estimate": True,
            "risk_level": spec.risk_level.value,
            "approval_required": bool(spec.requires_human_approval),
        }
        self._timeline(
            "execution_start", tool=request.tool, target=target, authorization="authorized"
        )
        exec_node = self.session.graph.add(
            kind=EvidenceGraphKind.TOOL_EXECUTION.value,
            provenance="execution",
            summary=f"execute {request.tool}",
            source=request.tool,
            parent_id=request_node.id,
            relation="executes",
        )
        try:
            permit = self.tools.issue_permit(
                request.tool, session_id=self.session.id, mode=self.session.mode
            )
            result = await self.tools.execute(request, permit=permit)
        except AuthorizationDeniedError as exc:
            self._timeline(
                "execution_result",
                tool=request.tool,
                target=target,
                authorization="blocked",
                result=str(exc),
            )
            self._remember_fingerprint(fingerprint, ToolResultQuality.BLOCKED.value)
            return {
                "authorization": "BLOCKED",
                "quality": ToolResultQuality.BLOCKED.value,
                "reason": str(exc),
                "executed": False,
            }
        except ApprovalRequiredError as exc:
            self.session.state = ResearchState.WAITING_FOR_APPROVAL
            self._remember_fingerprint(fingerprint, ToolResultQuality.APPROVAL_REQUIRED.value)
            return {
                "authorization": "APPROVAL_REQUIRED",
                "quality": ToolResultQuality.APPROVAL_REQUIRED.value,
                "reason": str(exc),
                "executed": False,
            }
        if result.get("scan_seconds"):
            try:
                self.session.budget.consume_scan(float(result["scan_seconds"]))
            except SafetyLimitExceededError:
                self.session.termination_reason = TerminationReason.BUDGET_EXHAUSTED
                self.session.state = ResearchState.BUDGET_EXHAUSTED
        if result.get("duration_seconds") and request.tool == "exploratory_test":
            try:
                self.session.budget.note_exploratory_seconds(float(result["duration_seconds"]))
            except SafetyLimitExceededError:
                self.session.termination_reason = TerminationReason.BUDGET_EXHAUSTED
                self.session.state = ResearchState.BUDGET_EXHAUSTED
        quality = str(result.get("quality") or ToolResultQuality.SUCCESS.value)
        summary = json.dumps(result, default=str)[:500]
        if contains_injection_attempt(summary):
            summary = untrusted_observation(request.tool, summary)
        observation = None
        if (
            quality
            not in {
                ToolResultQuality.BLOCKED.value,
                ToolResultQuality.UNAVAILABLE.value,
                ToolResultQuality.APPROVAL_REQUIRED.value,
            }
            and result.get("executed", True) is not False
        ):
            observation = self.session.graph.add(
                kind=EvidenceGraphKind.OBSERVATION.value,
                provenance="replay" if self.session.replay_mode else "execution",
                summary=redact_text(summary),
                source=request.tool,
                parent_id=exec_node.id,
                relation="observes",
            )
            for evidence_id in result.get("evidence_ids") or []:
                if evidence_id in self.session.graph.nodes:
                    edge = (exec_node.id, str(evidence_id), "produced")
                    if edge not in self.session.graph.edges:
                        self.session.graph.link(exec_node.id, str(evidence_id), "produced")
            if result.get("evidence_id") and result["evidence_id"] in self.session.graph.nodes:
                edge = (exec_node.id, str(result["evidence_id"]), "produced")
                if edge not in self.session.graph.edges:
                    self.session.graph.link(exec_node.id, str(result["evidence_id"]), "produced")
        self.session.state = ResearchState.OBSERVING
        evidence_id = (
            observation.id if observation else str(result.get("evidence_id") or exec_node.id)
        )
        self._timeline(
            "execution_result",
            decision=reason,
            tool=request.tool,
            target=target,
            authorization="authorized",
            result=summary[:200],
            evidence_id=evidence_id,
        )
        self._timeline(
            "evidence_created", tool=request.tool, evidence_id=evidence_id, result=quality
        )
        self._remember_fingerprint(fingerprint, quality)
        self.session.tool_history.append(request.tool)
        evidence_ids = tuple(
            str(item)
            for item in (result.get("evidence_ids") or ([evidence_id] if evidence_id else []))
        )
        self.session.tool_call_records.append(
            ToolCallRecord(
                tool=request.tool,
                arguments=dict(dumped),
                reason=reason,
                authorization="AUTHORIZED",
                authorization_reason="",
                execution_state=str(result.get("state") or "completed"),
                result_quality=quality,
                result_summary=summary,
                evidence_ids=evidence_ids,
            )
        )
        return {
            "authorization": "AUTHORIZED",
            "result": result,
            "evidence_id": evidence_id,
            "quality": quality,
        }

    def _authorize_target(
        self,
        target: str,
        *,
        tool: str,
        spec: Any | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> ToolAuthorization:
        arguments = arguments or {}
        if spec is None:
            try:
                spec = self.tools.spec(tool)
            except RestrictedActivityError:
                spec = None
        if spec is None:
            return ToolAuthorization.BLOCKED
        if self.session.mode is ResearchMode.LIVE_HACKERONE:
            if not self.session.engine.session.scope.includes:
                return ToolAuthorization.BLOCKED
            if (
                spec.requires_active_testing
                and not self.session.engine.session.active_testing_enabled
            ):
                return ToolAuthorization.BLOCKED
            if self.session.program_handle and (
                self.session.engine.session.scope.program_name
                not in {self.session.program_handle, None}
                and self.session.engine.session.scope.program_name != self.session.program_handle
            ):
                return ToolAuthorization.BLOCKED
        if spec.requires_human_approval and spec.approval_kind:
            if self.session.mode is ResearchMode.LIVE_HACKERONE or spec.risk_level in {
                ToolRiskLevel.ACTIVE,
                ToolRiskLevel.HIGH_RISK,
            }:
                kind = ApprovalKind(spec.approval_kind)
                if (
                    self.session.mode is ResearchMode.LIVE_HACKERONE
                    and not self.session.engine.approvals.is_granted(kind)
                ):
                    return ToolAuthorization.APPROVAL_REQUIRED
                if (
                    spec.risk_level is ToolRiskLevel.HIGH_RISK
                    and not self.session.engine.approvals.is_granted(ApprovalKind.HIGH_RISK_SCANNER)
                    and self.session.mode is ResearchMode.LIVE_HACKERONE
                ):
                    return ToolAuthorization.APPROVAL_REQUIRED
                if (
                    tool == "fuzz"
                    and not self.session.engine.session.fuzzing_enabled
                    and not self.session.engine.approvals.is_granted(ApprovalKind.ENABLE_FUZZING)
                ):
                    return ToolAuthorization.APPROVAL_REQUIRED
        high_risk = spec.risk_level is ToolRiskLevel.HIGH_RISK
        live_scan = tool in {"zap_scan", "nuclei_scan"}
        poc = tool in {"reproduce"} or (
            tool == "http_request"
            and bool(arguments.get("active", True))
            and spec.approval_kind == "send_poc_request"
        )
        content = arguments.get("content")
        payload_bytes = len(content.encode("utf-8")) if isinstance(content, str) else 0
        decision = self.session.engine.authorize(
            target,
            method=str(arguments.get("method") or "GET"),
            tool=tool,
            active=spec.requires_active_testing,
            payload_bytes=payload_bytes,
            require_live_scan_approval=live_scan,
            require_poc_approval=poc and self.session.mode is ResearchMode.LIVE_HACKERONE,
            high_risk=high_risk and self.session.mode is ResearchMode.LIVE_HACKERONE,
        )
        if not decision.allowed:
            if decision.approval_required:
                return ToolAuthorization.APPROVAL_REQUIRED
            return ToolAuthorization.BLOCKED
        return ToolAuthorization.AUTHORIZED

    def correlate(self) -> list[ResearchHypothesis]:
        """Strengthen hypotheses with independent sources. Never auto-verify."""
        self.session.state = ResearchState.CORRELATING
        updated = correlate_all(self.session.hypotheses, self.session.graph)
        for item in updated:
            self._timeline("hypothesis_update", decision=f"{item.id}:{item.status.value}")
        return updated

    def investigate_conflicts(self, hypothesis: ResearchHypothesis) -> HypothesisStatus:
        from app.security_agent.correlation import correlate_hypothesis

        status = correlate_hypothesis(hypothesis, self.session.graph)
        self._timeline("hypothesis_update", decision=f"conflict:{hypothesis.id}:{status.value}")
        return status

    def prioritize(self) -> list[ResearchHypothesis]:
        return rank_hypotheses(self.session.hypotheses)

    def update_hypothesis(
        self,
        hypothesis_id: str,
        *,
        status: HypothesisStatus,
        evidence_ids: tuple[str, ...] = (),
        contradicting: tuple[str, ...] = (),
    ) -> ResearchHypothesis:
        if status is HypothesisStatus.VERIFIED:
            raise RestrictedActivityError("verify_finding")
        found = next((item for item in self.session.hypotheses if item.id == hypothesis_id), None)
        if found is None:
            raise RestrictedActivityError("unknown_hypothesis")
        if status in {
            HypothesisStatus.SUPPORTED,
            HypothesisStatus.WEAKENED,
            HypothesisStatus.DISPROVED,
            HypothesisStatus.REQUIRES_REPRODUCTION,
            HypothesisStatus.REJECTED,
        } and not (evidence_ids or contradicting):
            raise RestrictedActivityError("hypothesis_status_requires_evidence")
        found.status = status
        if evidence_ids:
            found.supporting_evidence_ids = tuple(
                dict.fromkeys(found.supporting_evidence_ids + evidence_ids)
            )
        if contradicting:
            found.contradicting_evidence_ids = tuple(
                dict.fromkeys(found.contradicting_evidence_ids + contradicting)
            )
        self._timeline("hypothesis_update", decision=f"{found.id}:{status.value}")
        return found

    def draft_report(self, hypothesis: ResearchHypothesis) -> dict[str, Any]:
        """Summarize validated graph evidence. Cannot verify, approve, or submit."""
        why = []
        for evidence_id in hypothesis.supporting_evidence_ids:
            why.extend(self.session.graph.why(evidence_id))
        return {
            "title": hypothesis.title,
            "vulnerability_class": hypothesis.vulnerability_class,
            "target": hypothesis.target,
            "impact": "Impact is described from corroborated evidence only.",
            "evidence": why,
            "not_verified": True,
            "cannot_approve": True,
            "cannot_submit": True,
            "thinking_excluded": True,
        }

    def draft_report_candidate(self, finding: SecurityFinding) -> dict[str, Any]:
        if not finding.is_verified:
            raise RestrictedActivityError("draft_report_requires_verified_finding")
        evidence_ids = [str(item.id) for item in finding.evidence.items]
        return {
            "kind": "DRAFT_REPORT_CANDIDATE",
            "verified_finding_id": str(finding.id),
            "evidence_ids": evidence_ids,
            "reproduction_ids": [
                node.id
                for node in self.session.graph.nodes.values()
                if node.kind == EvidenceGraphKind.REPRODUCTION.value
            ],
            "target": finding.target or self.session.target,
            "scope_snapshot": {
                "program": self.session.program_handle,
                "includes": [
                    rule.identifier for rule in self.session.engine.session.scope.includes
                ],
                "hash": self.session.privilege.scope_hash if self.session.privilege else "",
            },
            "candidate_weakness": finding.vulnerability_class or "",
            "candidate_severity": finding.impact or "unspecified",
            "cannot_approve": True,
            "cannot_submit": True,
            "cannot_verify": True,
            "cannot_change_scope": True,
        }

    def promote(self, hypothesis: ResearchHypothesis) -> SecurityFinding | None:
        finding = promote_hypothesis(self.session, hypothesis)
        if finding:
            self._timeline(
                "verification", decision="promoted_potential", finding_id=str(finding.id)
            )
        return finding

    def record_reproduction(
        self,
        hypothesis: ResearchHypothesis,
        *,
        outcome: ReproductionOutcome,
        evidence: list[Any] | None = None,
    ) -> SecurityFinding | None:
        finding = apply_reproduction(self.session, hypothesis, outcome=outcome, evidence=evidence)
        self._timeline(
            "reproduction", decision=outcome.value, finding_id=str(finding.id) if finding else ""
        )
        return finding

    def context_window(self) -> dict[str, Any]:
        return self.context.build(self.session)

    async def _llm_planner(self, session: ResearchSession) -> AgentDecision:
        schemas = json.dumps(self.tools.llm_tools(), default=str)
        self.context.estimate(session, tool_schemas=schemas)
        advertised = session.provider.capabilities()
        limit = int(advertised.max_context_tokens or 8192)
        context = self.context.for_model(session, max_context_tokens=limit)
        request = CompletionRequest(
            system_prompt=channel(
                "TRUSTED_INSTRUCTIONS",
                "You are a BugForge research planner. You never decide scope, "
                "never mark findings verified, never approve reports, never grant "
                "approvals, and never submit to HackerOne. Return JSON with keys "
                "kind, tool, arguments, reason. Thinking is not evidence.",
                trusted=True,
            ),
            user_message=context,
            json_mode=True,
            thinking=session.thinking_enabled,
            tools=self.tools.llm_tools(),
        )
        response = await session.provider.complete(request)
        usage = getattr(response, "usage", None)
        if usage is not None:
            total = int(getattr(usage, "prompt_tokens", 0) or 0) + int(
                getattr(usage, "completion_tokens", 0) or 0
            )
            if total:
                try:
                    session.budget.consume("token", amount=total)
                except SafetyLimitExceededError as exc:
                    session.state = ResearchState.BUDGET_EXHAUSTED
                    session.termination_reason = TerminationReason.BUDGET_EXHAUSTED
                    return AgentDecision(kind="budget_exhausted", note=str(exc))
        if response.thinking:
            self._timeline("thinking", decision="model thinking ignored as evidence")
        try:
            if response.tool_calls:
                first = response.tool_calls[0]
                payload = PlannerOutput.model_validate(
                    {
                        "kind": "tool",
                        "tool": str(first.get("tool") or first.get("name") or ""),
                        "arguments": dict(first.get("arguments") or first.get("args") or {}),
                        "reason": str(first.get("reason") or ""),
                    }
                )
            else:
                raw = json.loads(response.content or "{}")
                if not isinstance(raw, dict):
                    raise ValueError("model output is not an object")
                payload = PlannerOutput.model_validate(raw)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            self._timeline("model_decision", decision=f"rejected_malformed:{exc}")
            return AgentDecision(
                kind="analyze", note="unparseable model output", thinking=response.thinking
            )
        if payload.kind in _FORBIDDEN_AI_ACTIONS:
            raise RestrictedActivityError(payload.kind)
        self._timeline("model_decision", decision=payload.kind)
        return self._decision_from_planner(payload, thinking=response.thinking)

    def _decision_from_planner(
        self, payload: PlannerOutput, *, thinking: str | None
    ) -> AgentDecision:
        if payload.kind in _FORBIDDEN_AI_ACTIONS:
            raise RestrictedActivityError(payload.kind)
        reason = strip_instruction_attempts(payload.reason or payload.note or "")
        if payload.kind == "tool" and payload.tool:
            tool = ToolCallRequest(
                tool=payload.tool,
                arguments=dict(payload.arguments or {}),
                reason=reason,
            )
            return AgentDecision(kind="tool", tool=tool, thinking=thinking, note=reason)
        if payload.kind == "hypothesis":
            hyp = ResearchHypothesis(
                title=strip_instruction_attempts(str(payload.title or "Untitled hypothesis")),
                vulnerability_class=strip_instruction_attempts(
                    str(payload.vulnerability_class or "unknown")
                ),
                target=str(payload.target or self.session.target),
                reason=reason,
                confidence=str(payload.confidence or "low"),
                severity=str(payload.severity or "medium"),
                impact=strip_instruction_attempts(str(payload.impact or "")),
            )
            return AgentDecision(kind="hypothesis", hypothesis=hyp, thinking=thinking)
        if payload.kind == "update_hypothesis" and payload.hypothesis_id:
            status = HypothesisStatus(payload.status or HypothesisStatus.OPEN.value)
            hyp = self.update_hypothesis(
                payload.hypothesis_id,
                status=status,
                evidence_ids=tuple(payload.evidence_ids),
            )
            if reason:
                hyp.reason = reason
            return AgentDecision(kind="update_hypothesis", hypothesis=hyp, thinking=thinking)
        if payload.kind == "plan":
            steps = tuple(
                ResearchPlanStep(
                    order=int(item.get("order") or index),
                    action=strip_instruction_attempts(str(item.get("action") or "")),
                    tool=item.get("tool"),
                    target=str(item.get("target") or ""),
                    note=strip_instruction_attempts(str(item.get("note") or "")),
                )
                for index, item in enumerate(payload.plan_steps or [])
            )
            return AgentDecision(kind="plan", plan=ResearchPlan(steps=steps), thinking=thinking)
        return AgentDecision(kind=payload.kind, note=reason, thinking=thinking)

    def _record_hypothesis(self, hypothesis: ResearchHypothesis, *, update: bool = False) -> None:
        if hypothesis.status is HypothesisStatus.VERIFIED:
            raise RestrictedActivityError("verify_finding")
        if not update:
            fingerprint = finding_fingerprint(
                vulnerability_class=hypothesis.vulnerability_class,
                target=hypothesis.target,
                endpoint=hypothesis.target,
            )
            for existing in self.session.hypotheses:
                other = finding_fingerprint(
                    vulnerability_class=existing.vulnerability_class,
                    target=existing.target,
                    endpoint=existing.target,
                )
                if other == fingerprint:
                    existing.reason = hypothesis.reason or existing.reason
                    existing.confidence = hypothesis.confidence
                    self._timeline("hypothesis_update", decision=f"dedup:{existing.id}")
                    self.session.state = ResearchState.HYPOTHESIS_CREATED
                    return
            if hypothesis.id in self.session.graph.nodes:
                node = self.session.graph.nodes[hypothesis.id]
            else:
                node = self.session.graph.add(
                    kind=EvidenceGraphKind.HYPOTHESIS.value,
                    provenance="ai_hypothesis",
                    summary=hypothesis.title,
                    source="planner",
                    node_id=hypothesis.id,
                    extra={"hypothesis_id": hypothesis.id},
                )
            hypothesis.supporting_evidence_ids = hypothesis.supporting_evidence_ids
            self.session.hypotheses.append(hypothesis)
            from app.security_agent.leads import lead_for_hypothesis

            lead_for_hypothesis(self.session, hypothesis)
            for evidence_id in hypothesis.supporting_evidence_ids:
                if evidence_id in self.session.graph.nodes and evidence_id != node.id:
                    self.session.graph.link(node.id, evidence_id, "supports")
            for evidence_id in hypothesis.contradicting_evidence_ids:
                if evidence_id in self.session.graph.nodes and evidence_id != node.id:
                    self.session.graph.link(node.id, evidence_id, "contradicted_by")
        self.session.state = ResearchState.HYPOTHESIS_CREATED
        self._timeline("hypothesis", decision=hypothesis.title)

    def _check_duplicate(self, fingerprint: str) -> None:
        window = timedelta(seconds=self.session.budget.identical_call_window_seconds)
        now = datetime.now(UTC)
        recent = [
            item
            for item in self.session.fingerprint_records
            if item.fingerprint == fingerprint and now - item.at <= window
        ]
        successful = [item for item in recent if item.quality not in _RETRYABLE_QUALITY]
        if len(successful) >= self.session.budget.max_identical_calls:
            raise RestrictedActivityError("repeated_identical_tool_call")
        # Legacy fingerprints list keeps the previous count semantics for identical successes.
        self.session.fingerprints.append(fingerprint)

    def _remember_fingerprint(self, fingerprint: str, quality: str) -> None:
        self.session.fingerprint_records.append(
            FingerprintRecord(fingerprint=fingerprint, at=datetime.now(UTC), quality=quality)
        )

    def _refresh_privilege(self) -> None:
        self.session.privilege = capture_privileges(
            self.session.engine,
            mode=self.session.mode,
            program_handle=self.session.program_handle,
            disabled_tools=self.session.disabled_tools,
        )

    def _timeline(self, event_type: str, **fields: Any) -> None:
        fields.setdefault("strategy", self.session.strategy)
        fields.setdefault("state", self.session.state.value)
        self.session.timeline.append(TimelineEvent(event_type=event_type, **fields))


def _tool_context(session: ResearchSession) -> ToolContext:
    return ToolContext(
        engine=session.engine,
        graph=session.graph,
        project_id=session.project_id,
        session_id=session.id,
        mode=session.mode,
        program_handle=session.program_handle,
        repo_root=Path(session.repo_root).resolve()
        if session.repo_root
        else Path("/nonexistent-bugforge-no-repo"),
        exchanges=session.exchanges,
        cancelled=session.cancelled,
        zap_runner=session.zap_runner,
        nuclei_runner=session.nuclei_runner,
        zap_binary=session.zap_binary,
        nuclei_binary=session.nuclei_binary,
        hypotheses=lambda: {item.id: item.target for item in session.hypotheses},
        exploratory_records=session.exploratory_attempts,
    )


def bind_engine_tools_for_session(
    session: ResearchSession, registry: ToolRegistry | None = None
) -> ToolRegistry:
    return bind_engine_tools(_tool_context(session), registry or default_registry())
