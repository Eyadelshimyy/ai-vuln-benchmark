"""Guided security-research agent API. The AI cannot grant itself authority."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai import get_provider
from app.ai.external_controller import ExternalControllerProvider, cursor_model_label
from app.ai.provider import LLMProvider
from app.core.config import settings
from app.cursor_control.analysis import analyze_session_repository
from app.cursor_control.status import gdk_connectivity, git_identity
from app.database import get_db
from app.models.project import Project
from app.repositories.hackerone_repo import HackerOneRepository
from app.repositories.security_agent_repo import SecurityAgentRepository
from app.security_agent.agent import ResearchSession, SecurityResearchAgent
from app.security_agent.export import FindingNotFoundError, export_package
from app.security_agent.handoff import prepare_hackerone_handoff
from app.security_agent.identities import IdentityPair, ResearchIdentity, compare_identities
from app.security_agent.memory import ResearchMemory
from app.security_agent.orchestrator import AdvancedResearchOrchestrator
from app.security_agent.privilege import program_scope_from_hackerone
from app.security_agent.project import SecurityResearchProject
from app.security_agent.replay import SessionReplay
from app.security_agent.repo_lock import resolve_repo_root
from app.security_agent.schemas import PlannerOutput, ToolCallRequest
from app.security_agent.secrets import secrets_for
from app.security_agent.states import (
    RESUME_BLOCKED_STATES,
    ResearchController,
    ResearchMode,
    ResearchProjectState,
)
from app.security_agent.workbench import (
    dashboard as workbench_dashboard,
)
from app.security_agent.workbench import (
    evidence_explorer,
    evidence_graph_view,
    findings_workbench,
    identity_workbench,
    next_action_review,
)
from app.security_agent.workbench import (
    timeline as workbench_timeline,
)
from app.security_testing.approvals import ApprovalKind, is_ai_operator
from app.security_testing.engine import SecurityTestEngine, SecurityTestSession, TestingMode
from app.security_testing.errors import (
    ApprovalRequiredError,
    RestrictedActivityError,
    SafetyLimitExceededError,
)
from app.security_testing.operator_auth import OperatorSession, require_operator
from app.security_testing.safety import SafetyLimits

router = APIRouter(prefix="/security-agent", tags=["Security Agent"])

_SESSIONS: dict[str, SecurityResearchAgent] = {}
_ORCHESTRATORS: dict[str, AdvancedResearchOrchestrator] = {}
_LOCKS: dict[str, asyncio.Lock] = {}
_PROJECTS: dict[str, SecurityResearchProject] = {}
_LIVE_DISABLED = frozenset({"zap_scan", "nuclei_scan", "fuzz"})
_REPO_ROOT = Path(__file__).resolve().parents[5]


class CreateAgentSessionRequest(BaseModel):
    project_id: str
    target: str
    mode: str = "lab"
    program_handle: str = ""
    thinking: bool = True
    repo_root: str | None = None
    controller: str = "internal_llm"
    cursor_model: str | None = None


class ExternalDecisionRequest(BaseModel):
    kind: str
    tool: str | None = None
    arguments: dict[str, object] = {}
    reason: str = ""
    title: str | None = None
    vulnerability_class: str | None = None
    target: str | None = None
    confidence: str | None = None
    severity: str | None = None
    impact: str | None = None
    hypothesis_id: str | None = None
    status: str | None = None
    evidence_ids: list[str] = []
    plan_steps: list[dict[str, object]] = []
    note: str = ""


class AnalyzeRepositoryRequest(BaseModel):
    max_files: int = 200


class ControlRequest(BaseModel):
    reason: str = "operator"


class ToolActionRequest(BaseModel):
    tool: str
    arguments: dict[str, object]
    reason: str = ""


class OverrideRequest(BaseModel):
    action: str
    reason: str = ""
    extra_tool_calls: int = 0
    tool: str | None = None
    finding_id: str | None = None
    strategy: str | None = None
    source: str | None = None
    destination: str | None = None
    amount: int = 0


class ApprovalActionRequest(BaseModel):
    kind: str
    note: str = ""


class StrategyRequest(BaseModel):
    strategy: str


class FalsePositiveRequest(BaseModel):
    why: str
    evidence: str = ""
    source: str = "operator"


class IdentityRequest(BaseModel):
    label: str
    cookies: dict[str, str] = {}
    storage: dict[str, str] = {}
    headers: dict[str, str] = {}


class ReplayRequest(BaseModel):
    events: list[dict[str, object]]
    live_network: bool = False


class HandoffRequest(BaseModel):
    finding_id: str


def _controller_from(value: str) -> ResearchController:
    cleaned = (value or "").strip().lower()
    if cleaned in {"", "internal_llm", "internal"}:
        return ResearchController.INTERNAL_LLM
    if cleaned in {"cursor", "cursor_external", "external_ai"}:
        return ResearchController.CURSOR
    raise ValueError("controller must be internal_llm or cursor")


def _lock_for(session_id: str) -> asyncio.Lock:
    lock = _LOCKS.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _LOCKS[session_id] = lock
    return lock


def _orchestrator_for(agent: SecurityResearchAgent) -> AdvancedResearchOrchestrator:
    existing = _ORCHESTRATORS.get(agent.session.id)
    if existing is not None and existing.agent is agent:
        return existing
    orch = AdvancedResearchOrchestrator(agent)
    _ORCHESTRATORS[agent.session.id] = orch
    return orch


@router.post("/sessions")
async def create_session(
    payload: CreateAgentSessionRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    try:
        mode = ResearchMode(payload.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="mode must be lab or live_hackerone") from exc
    try:
        controller = _controller_from(payload.controller)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if controller is ResearchController.CURSOR and mode is not ResearchMode.LAB:
        raise HTTPException(status_code=400, detail="cursor control sessions are lab-only")
    project: Project | None = None
    if _is_uuid(payload.project_id):
        project = await db.get(Project, UUID(payload.project_id))
        if project is None:
            raise HTTPException(status_code=404, detail="Unknown project")
    try:
        repo_root = resolve_repo_root(
            project_repository_path=project.repository_path if project else None,
            client_repo_root=payload.repo_root,
            mode=mode.value,
            project_id=payload.project_id,
        )
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    disabled: set[str] = set()
    if mode is ResearchMode.LIVE_HACKERONE:
        handle = payload.program_handle.strip()
        if not handle:
            raise HTTPException(
                status_code=400, detail="program_handle is required for live sessions"
            )
        program = await HackerOneRepository(db).get_program(handle)
        if program is None or not program.structured_scopes:
            raise HTTPException(
                status_code=409,
                detail="LIVE_SCOPE_MISSING: synchronize structured scope before creating a live research session",
            )
        scope = program_scope_from_hackerone(program)
        if not scope.includes:
            raise HTTPException(
                status_code=409,
                detail="LIVE_SCOPE_EMPTY: refusing a permissive live session without includes",
            )
        disabled = set(_LIVE_DISABLED)
        engine = SecurityTestEngine(
            SecurityTestSession(
                project_id=payload.project_id,
                mode=TestingMode.LIVE,
                scope=scope,
                limits=SafetyLimits.conservative(),
                dry_run=True,
                active_testing_enabled=False,
                fuzzing_enabled=False,
            )
        )
    else:
        engine = SecurityTestEngine.lab(
            payload.project_id,
            allow_active_testing=True,
            limits=SafetyLimits.lab(),
        )
    provider: LLMProvider
    if controller is ResearchController.CURSOR:
        provider = ExternalControllerProvider(model_name=cursor_model_label(payload.cursor_model))
        thinking = False
    else:
        provider = get_provider()
        thinking = payload.thinking
    research = ResearchSession(
        project_id=payload.project_id,
        target=payload.target,
        mode=mode,
        engine=engine,
        provider=provider,
        program_handle=payload.program_handle,
        thinking_enabled=thinking,
        repo_root=repo_root,
        disabled_tools=disabled,
        identities=IdentityPair(),
        memory=ResearchMemory(project_id=payload.project_id),
        operator_identity=session.identity,
        controller=controller,
        model_name=provider.model_name,
    )
    agent = SecurityResearchAgent(research)
    agent.tools.restore_disabled(sorted(disabled))
    _SESSIONS[research.id] = agent
    _orchestrator_for(agent)
    await SecurityAgentRepository(db).save_session(research)
    await db.commit()
    return research.snapshot()


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    return agent.session.snapshot()


@router.post("/sessions/{session_id}/step")
async def step_session(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    if agent.session.controller is ResearchController.CURSOR:
        raise HTTPException(
            status_code=409,
            detail="cursor_mode_refuses_internal_planner",
        )
    orch = _orchestrator_for(agent)
    async with _lock_for(session_id):
        try:
            decision = await orch.step()
        except RestrictedActivityError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except SafetyLimitExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        await SecurityAgentRepository(db).save_session(agent.session)
        await db.commit()
    return {
        "decision": decision.kind,
        "session": agent.session.snapshot(),
        "next_action": agent.session.next_action,
        "dashboard": orch.dashboard(),
    }


@router.post("/sessions/{session_id}/tools")
async def request_tool(
    session_id: str,
    payload: ToolActionRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    orch = _orchestrator_for(agent)
    async with _lock_for(session_id):
        try:
            orch.explain_next(
                payload.tool, reason=payload.reason, arguments=dict(payload.arguments)
            )
            result = await agent.request_tool(
                ToolCallRequest(
                    tool=payload.tool, arguments=dict(payload.arguments), reason=payload.reason
                )
            )
        except RestrictedActivityError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except SafetyLimitExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ApprovalRequiredError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        await SecurityAgentRepository(db).save_session(agent.session)
        await db.commit()
    return {**result, "next_action": agent.session.next_action}


@router.post("/sessions/{session_id}/override")
async def override_session(
    session_id: str,
    payload: OverrideRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    action = payload.action.lower()
    if action == "pause":
        agent.pause(payload.reason)
    elif action == "stop":
        agent.stop(payload.reason)
    elif action == "resume":
        try:
            agent.resume(operator=session.identity)
        except RestrictedActivityError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
    elif action == "reject_action":
        agent.reject_action(payload.reason)
    elif action.startswith("disable_tool:"):
        agent.disable_tool(action.split(":", 1)[1])
    elif action == "disable_tool" and payload.tool:
        agent.disable_tool(payload.tool)
    elif action.startswith("enable_tool:"):
        name = action.split(":", 1)[1]
        agent.session.disabled_tools.discard(name)
        agent.tools.enable(name)
    elif action == "enable_tool" and payload.tool:
        agent.session.disabled_tools.discard(payload.tool)
        agent.tools.enable(payload.tool)
    elif action == "change_limits":
        agent.change_limits(operator=session.identity, extra_tool_calls=payload.extra_tool_calls)
    elif action == "reallocate_budget":
        try:
            agent.session.budget.reallocate(
                source=str(payload.source or ""),
                destination=str(payload.destination or ""),
                amount=payload.amount,
            )
        except SafetyLimitExceededError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    elif action == "reject_finding":
        ident = payload.finding_id or payload.tool
        if not ident:
            raise HTTPException(status_code=400, detail="finding_id is required")
        agent.reject_finding(ident)
    elif action == "change_strategy":
        strategy = payload.strategy or payload.reason
        if not strategy:
            raise HTTPException(status_code=400, detail="strategy is required")
        orch = _orchestrator_for(agent)
        orch.recommend_strategy(strategy)
        agent.session.strategy = orch.strategy.value
        agent._timeline("human_override", decision=f"CHANGE_STRATEGY:{strategy}")
    else:
        raise HTTPException(status_code=400, detail="Unknown override")
    await SecurityAgentRepository(db).save_session(agent.session)
    await db.commit()
    return agent.session.snapshot()


@router.post("/sessions/{session_id}/approvals")
async def grant_session_approval(
    session_id: str,
    payload: ApprovalActionRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    if is_ai_operator(session.identity):
        raise HTTPException(status_code=403, detail="The AI cannot grant approvals")
    agent = await _require(session_id, db, operator=session)
    try:
        kind = ApprovalKind(payload.kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="unknown approval kind") from exc
    try:
        if kind is ApprovalKind.ENABLE_ACTIVE_TESTING:
            agent.enable_active_testing(operator=session.identity, note=payload.note)
        else:
            agent.grant_tool_approval(kind, operator=session.identity, note=payload.note)
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    await SecurityAgentRepository(db).save_session(agent.session)
    await db.commit()
    return {
        "approvals": agent.session.engine.approvals.snapshot(),
        "active_testing": agent.session.engine.session.active_testing_enabled,
        "fuzzing": agent.session.engine.session.fuzzing_enabled,
    }


@router.post("/sessions/{session_id}/resume")
async def resume_session(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    if agent.session.state in RESUME_BLOCKED_STATES or agent.session.stopped:
        raise HTTPException(
            status_code=409,
            detail="Terminal sessions require an explicit operator reopen; automatic resume is blocked",
        )
    try:
        agent.resume(operator=session.identity)
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    await SecurityAgentRepository(db).save_session(agent.session)
    await db.commit()
    return agent.session.snapshot()


@router.get("/status")
async def research_status(
    session: OperatorSession = Depends(require_operator),
) -> dict[str, object]:
    """Session inventory for the local operator. Does not call a language model."""

    visible = []
    for agent in _SESSIONS.values():
        owner = agent.session.operator_identity
        if owner and owner != session.identity:
            continue
        snap = agent.session.snapshot()
        visible.append(
            {
                "id": snap["id"],
                "project_id": snap["project_id"],
                "target": snap["target"],
                "mode": snap["mode"],
                "state": snap["state"],
                "controller": snap["controller"],
                "ai_controller": snap["ai_controller"],
                "ai_execution": snap["ai_execution"],
                "provider": snap["provider"],
                "model": snap["model"],
            }
        )
    return {
        "version": settings.version,
        "git": git_identity(_REPO_ROOT),
        "configured_legacy_ai_provider": settings.ai_provider,
        "cursor_mode_calls_llm": False,
        "ai_execution_in_cursor_mode": "none",
        "sessions": visible,
        "gdk": gdk_connectivity(),
    }


@router.post("/sessions/{session_id}/decision")
async def external_decision(
    session_id: str,
    payload: ExternalDecisionRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Accept one Cursor decision. Never invokes the internal planner."""

    agent = await _require(session_id, db, operator=session)
    if agent.session.controller is not ResearchController.CURSOR:
        raise HTTPException(status_code=409, detail="external_decision_requires_cursor_controller")
    try:
        parsed = PlannerOutput.model_validate(payload.model_dump())
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="rejected_external_decision") from exc
    async with _lock_for(session_id):
        try:
            decision = await agent.apply_external_decision(parsed.model_dump())
        except RestrictedActivityError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except SafetyLimitExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ApprovalRequiredError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        await SecurityAgentRepository(db).save_session(agent.session)
        await db.commit()
    tool_result = None
    if agent.session.tool_call_records:
        tool_result = agent.session.tool_call_records[-1].snapshot()
    return {
        "decision": decision.kind,
        "session": agent.session.snapshot(),
        "tool_result": tool_result,
        "llm_invoked": False,
    }


@router.post("/sessions/{session_id}/analyze")
async def analyze_repository(
    session_id: str,
    payload: AnalyzeRepositoryRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    if agent.session.controller is not ResearchController.CURSOR:
        raise HTTPException(status_code=409, detail="cursor_analysis_requires_cursor_controller")
    try:
        report = analyze_session_repository(agent, max_files=payload.max_files)
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    await SecurityAgentRepository(db).save_session(agent.session)
    await db.commit()
    return report


@router.post("/sessions/{session_id}/pause")
async def pause_cursor_session(
    session_id: str,
    payload: ControlRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    agent.pause(payload.reason)
    await SecurityAgentRepository(db).save_session(agent.session)
    await db.commit()
    return agent.session.snapshot()


@router.post("/sessions/{session_id}/stop")
async def stop_cursor_session(
    session_id: str,
    payload: ControlRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    agent.stop(payload.reason)
    await SecurityAgentRepository(db).save_session(agent.session)
    await db.commit()
    return agent.session.snapshot()


@router.get("/sessions/{session_id}/timeline")
async def timeline(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
    tool: str | None = None,
    finding: str | None = None,
    strategy: str | None = None,
    state: str | None = None,
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    return workbench_timeline(
        _orchestrator_for(agent), tool=tool, finding=finding, strategy=strategy, state=state
    )


@router.get("/sessions/{session_id}/tools")
async def list_tools(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    return {"tools": agent.tools.catalog()}


@router.get("/sessions/{session_id}/dashboard")
async def dashboard(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    return workbench_dashboard(_orchestrator_for(agent))


@router.get("/sessions/{session_id}/next-action")
async def next_action(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    return {"next_action": agent.session.next_action}


@router.post("/sessions/{session_id}/strategy")
async def set_strategy(
    session_id: str,
    payload: StrategyRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    orch = _orchestrator_for(agent)
    strategy = orch.recommend_strategy(payload.strategy)
    agent.session.strategy = strategy.value
    await SecurityAgentRepository(db).save_session(agent.session)
    await db.commit()
    return {"strategy": strategy.value, "permitted_tools": list(orch.permitted_tools())}


@router.post("/sessions/{session_id}/false-positive")
async def mark_false_positive(
    session_id: str,
    payload: FalsePositiveRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    orch = _orchestrator_for(agent)
    orch.record_false_positive(why=payload.why, evidence=payload.evidence, source=payload.source)
    memory = agent.session.memory or ResearchMemory(project_id=agent.session.project_id)
    memory.remember(
        "false_positive",
        payload.why,
        {"evidence": payload.evidence, "source": payload.source},
    )
    agent.session.memory = memory
    await SecurityAgentRepository(db).save_session(agent.session)
    await db.commit()
    return {"recorded": True, "findings_retained": len(agent.session.findings)}


@router.post("/sessions/{session_id}/identities")
async def set_identity(
    session_id: str,
    payload: IdentityRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    pair = agent.session.identities or IdentityPair()
    ident = ResearchIdentity(
        label=payload.label,
        cookies=dict(payload.cookies),
        storage=dict(payload.storage),
        headers=dict(payload.headers),
    )
    ident.ingest_secrets(secrets_for(agent.session.id))
    if payload.label.upper() == "B":
        pair.context_b = ident
    else:
        pair.context_a = ident
    agent.session.identities = pair
    await SecurityAgentRepository(db).save_session(agent.session)
    await db.commit()
    return {"a": pair.context_a.snapshot(), "b": pair.context_b.snapshot()}


@router.post("/sessions/{session_id}/checkpoint")
async def create_checkpoint(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    ident = await SecurityAgentRepository(db).save_checkpoint(agent.session)
    await db.commit()
    return {"checkpoint_id": ident}


@router.get("/sessions/{session_id}/export")
async def export_session(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
    finding_id: str | None = Query(default=None),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    try:
        if finding_id:
            return export_package(agent.session, finding_id=finding_id)
        return export_package(agent.session)
    except FindingNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Unknown finding") from exc


@router.post("/sessions/{session_id}/handoff")
async def handoff_finding(
    session_id: str,
    payload: HandoffRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    finding = next(
        (item for item in agent.session.findings if str(item.id) == payload.finding_id), None
    )
    if finding is None:
        raise HTTPException(status_code=404, detail="Unknown finding")
    try:
        return prepare_hackerone_handoff(agent.session, finding)
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/sessions/{session_id}/replay")
async def replay_session(
    session_id: str,
    payload: ReplayRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    if payload.live_network:
        raise HTTPException(status_code=403, detail="replay_default_is_offline")
    agent = await _require(session_id, db, operator=session)
    replay = SessionReplay(live_network=False)
    for event in payload.events:
        replay.record(
            str(event.get("tool") or ""),
            _as_dict(event.get("arguments")),
            _as_dict(event.get("result")),
        )
    async with _lock_for(session_id):
        decisions = await replay.play(agent)
    return {"decisions": [item.kind for item in decisions], "session": agent.session.snapshot()}


class ResearchProjectRequest(BaseModel):
    name: str
    project_id: str
    target: str
    mode: str = "lab"
    program_handle: str = ""
    strategy: str = "passive_recon"


class IdentityCompareRequest(BaseModel):
    method: str = "GET"
    url: str
    expectation: str
    oracle: str = "owner_only"


class WorkflowPlanRequest(BaseModel):
    tool: str
    arguments: dict[str, object] = {}
    reason: str = ""


class ProjectTransitionRequest(BaseModel):
    state: str


@router.post("/research-projects")
async def create_research_project(
    payload: ResearchProjectRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    project = SecurityResearchProject(
        name=payload.name,
        project_id=payload.project_id,
        target=payload.target,
        mode=payload.mode,
        program_handle=payload.program_handle,
        strategy=payload.strategy,
        operator_identity=session.identity,
    )
    _PROJECTS[project.id] = project
    await SecurityAgentRepository(db).save_project(project)
    await db.commit()
    return project.snapshot()


@router.get("/research-projects/{project_id}")
async def get_research_project(
    project_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    project = await _require_research_project(project_id, db, operator=session)
    return project.snapshot()


@router.post("/research-projects/{project_id}/transition")
async def transition_research_project(
    project_id: str,
    payload: ProjectTransitionRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    project = await _require_research_project(project_id, db, operator=session)
    try:
        project.transition(ResearchProjectState(payload.state))
    except (ValueError, RestrictedActivityError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await SecurityAgentRepository(db).save_project(project)
    await db.commit()
    return project.snapshot()


@router.get("/sessions/{session_id}/findings")
async def session_findings(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    return findings_workbench(_orchestrator_for(agent))


@router.get("/sessions/{session_id}/evidence")
async def session_evidence(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
    evidence_id: str | None = Query(default=None),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    if evidence_id:
        node = agent.session.graph.inspect(
            evidence_id,
            session_id=agent.session.id,
            project_id=agent.session.project_id,
        )
        if node is None:
            raise HTTPException(status_code=404, detail="Unknown evidence")
        return {"evidence": node, "session_id": agent.session.id}
    explorer = evidence_explorer(_orchestrator_for(agent))
    explorer["session_id"] = agent.session.id
    explorer["graph"] = agent.session.graph.snapshot()
    return explorer


@router.get("/sessions/{session_id}/graph")
async def session_graph(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    return evidence_graph_view(_orchestrator_for(agent))


@router.get("/sessions/{session_id}/identities")
async def session_identities(
    session_id: str,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    return identity_workbench(_orchestrator_for(agent))


@router.post("/sessions/{session_id}/identities/compare")
async def compare_session_identities(
    session_id: str,
    payload: IdentityCompareRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    pair = agent.session.identities or IdentityPair()
    try:
        result = await compare_identities(
            agent.session.engine,
            pair,
            method=payload.method,
            url=payload.url,
            expectation=payload.expectation,
            oracle=payload.oracle,
            session_id=agent.session.id,
        )
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    await SecurityAgentRepository(db).save_session(agent.session)
    await db.commit()
    return dict(result.snapshot())


@router.post("/sessions/{session_id}/next-action/review")
async def review_next_action(
    session_id: str,
    payload: WorkflowPlanRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    return next_action_review(
        _orchestrator_for(agent),
        tool=payload.tool,
        arguments=dict(payload.arguments),
        reason=payload.reason,
    )


@router.post("/sessions/{session_id}/workflows/plan")
async def plan_workflow(
    session_id: str,
    payload: WorkflowPlanRequest,
    session: OperatorSession = Depends(require_operator),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    agent = await _require(session_id, db, operator=session)
    review = next_action_review(
        _orchestrator_for(agent),
        tool=payload.tool,
        arguments=dict(payload.arguments),
        reason=payload.reason or f"plan {payload.tool}",
    )
    review["execute"] = False
    review["authorized_scan_is_not_successful_scan"] = True
    review["alert_is_not_verified_vulnerability"] = True
    return review


async def _require(
    session_id: str,
    db: AsyncSession,
    *,
    operator: OperatorSession,
) -> SecurityResearchAgent:
    agent = _SESSIONS.get(session_id)
    if agent is not None:
        await _assert_project(agent.session.project_id, db)
        _assert_operator_owns(agent.session.operator_identity, operator)
        return agent
    row_preview = await SecurityAgentRepository(db).load_row(session_id)
    if row_preview is None:
        raise HTTPException(status_code=404, detail="Unknown research session")
    await _assert_project(str(row_preview.project_id or ""), db)
    _assert_operator_owns(row_preview.operator_identity, operator)
    program = None
    if row_preview.mode == ResearchMode.LIVE_HACKERONE.value and row_preview.program_handle:
        program = await HackerOneRepository(db).get_program(row_preview.program_handle)
        if program is None:
            raise HTTPException(
                status_code=409,
                detail="LIVE_SCOPE_MISSING: current HackerOne program record is required to resume",
            )
    restored = await SecurityAgentRepository(db).reconstruct(session_id, current_program=program)
    if restored is None:
        raise HTTPException(status_code=404, detail="Unknown research session")
    _SESSIONS[session_id] = restored
    _orchestrator_for(restored)
    return restored


async def _require_research_project(
    project_id: str,
    db: AsyncSession,
    *,
    operator: OperatorSession,
) -> SecurityResearchProject:
    project = _PROJECTS.get(project_id)
    if project is None:
        row = await SecurityAgentRepository(db).load_project(project_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Unknown research project")
        project = SecurityResearchProject.from_row(row)
        _PROJECTS[project.id] = project
    _assert_operator_owns(project.operator_identity, operator)
    return project


def _assert_operator_owns(owner: str, operator: OperatorSession) -> None:
    if not owner:
        return
    if owner != operator.identity:
        raise HTTPException(status_code=403, detail="session_operator_mismatch")


async def _assert_project(project_id: str, db: AsyncSession) -> None:
    if not project_id or not _is_uuid(project_id):
        return
    project = await db.get(Project, UUID(project_id))
    if project is None:
        raise HTTPException(status_code=404, detail="Unknown project for research session")


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except ValueError:
        return False
