"""Authorized security testing API. No unrestricted scan action exists."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.plugins import get_plugin_catalog
from app.schemas.security_testing import (
    ApprovalRequest,
    AuditLogResponse,
    AuthorizationResponse,
    AuthorizeRequest,
    CreateTestingSessionRequest,
    TestingSessionResponse,
)
from app.security_testing.approvals import ApprovalKind
from app.security_testing.engine import SecurityTestEngine, SecurityTestSession, TestingMode
from app.security_testing.errors import (
    ApprovalRequiredError,
    RestrictedActivityError,
)
from app.security_testing.operator_auth import OperatorSession, require_operator
from app.security_testing.safety import SafetyLimits
from app.security_testing.scope_model import ProgramScope, ScopeRule
from app.security_testing.target import AssetType

router = APIRouter(prefix="/security-testing", tags=["Security Testing"])

_SESSIONS: dict[str, SecurityTestEngine] = {}


def _engine(project_id: str) -> SecurityTestEngine:
    engine = _SESSIONS.get(project_id)
    if engine is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No testing session")
    return engine


@router.post("/sessions", response_model=TestingSessionResponse)
async def create_session(payload: CreateTestingSessionRequest) -> TestingSessionResponse:
    try:
        mode = TestingMode(payload.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="mode must be lab or live") from exc
    includes = tuple(_rule(item, exclusion=False) for item in payload.includes)
    excludes = tuple(_rule(item, exclusion=True) for item in payload.excludes)
    scope = ProgramScope(
        program_name=payload.program_name,
        includes=includes,
        excludes=excludes,
        allow_active_testing=payload.allow_active_testing and mode is TestingMode.LAB,
        allowed_methods=tuple(payload.allowed_methods),
        lab_mode=mode is TestingMode.LAB,
    )
    limits = SafetyLimits(
        max_requests=payload.max_requests,
        requests_per_second=payload.requests_per_second,
        allowed_http_methods=tuple(payload.allowed_methods),
    )
    engine = SecurityTestEngine(
        SecurityTestSession(
            project_id=payload.project_id,
            mode=mode,
            scope=scope,
            limits=limits,
            dry_run=payload.dry_run if mode is TestingMode.LIVE else payload.dry_run,
            active_testing_enabled=payload.allow_active_testing and mode is TestingMode.LAB,
        )
    )
    _SESSIONS[payload.project_id] = engine
    return _session_response(engine)


@router.get("/sessions/{project_id}", response_model=TestingSessionResponse)
async def get_session(project_id: str) -> TestingSessionResponse:
    return _session_response(_engine(project_id))


@router.post("/sessions/{project_id}/approvals")
async def grant_approval(
    project_id: str,
    payload: ApprovalRequest,
    session: OperatorSession = Depends(require_operator),
) -> dict[str, str]:
    engine = _engine(project_id)
    try:
        kind = ApprovalKind(payload.kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="unknown approval kind") from exc
    try:
        engine.grant(kind, operator=session.identity, note=payload.note)
    except RestrictedActivityError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ApprovalRequiredError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return engine.approvals.snapshot()


@router.post("/sessions/{project_id}/authorize", response_model=AuthorizationResponse)
async def authorize(project_id: str, payload: AuthorizeRequest) -> AuthorizationResponse:
    engine = _engine(project_id)
    decision = engine.authorize(
        payload.target, method=payload.method, tool=payload.tool, active=payload.active
    )
    return AuthorizationResponse(
        allowed=decision.allowed,
        reason=decision.reason,
        target=decision.target_original,
        method=decision.method,
        tool=decision.tool,
        program=decision.program,
        dry_run=decision.dry_run,
        approval_required=decision.approval_required,
        approval_state=decision.approval_state,
        matched_rule=decision.matched_rule.identifier if decision.matched_rule else None,
        request_limit=engine.safety.limits.max_requests,
    )


@router.get("/sessions/{project_id}/audit", response_model=AuditLogResponse)
async def audit_log(project_id: str) -> AuditLogResponse:
    engine = _engine(project_id)
    return AuditLogResponse(
        entries=[event.to_mapping() for event in engine.audit.entries()],
        chain_valid=engine.audit.verify_chain(),
    )


@router.get("/tools")
async def list_tools() -> dict[str, list[str]]:
    catalog = get_plugin_catalog()
    return {
        "browsers": catalog.browsers.available_ids(),
        "proxies": catalog.proxies.available_ids(),
        "security_tools": catalog.security_tools.available_ids(),
        "fuzzers": catalog.fuzzers.available_ids(),
    }


def _session_response(engine: SecurityTestEngine) -> TestingSessionResponse:
    catalog = get_plugin_catalog()
    snap = engine.snapshot()
    snap["tools"] = {
        "browser": catalog.browsers.available_ids(),
        "proxy": catalog.proxies.available_ids(),
        "zap": "zap" in catalog.security_tools.available_ids(),
        "nuclei": "nuclei" in catalog.security_tools.available_ids(),
        "api": True,
        "fuzzing": catalog.fuzzers.available_ids(),
    }
    return TestingSessionResponse(session=snap, tools=catalog.security_tools.available_ids())


def _rule(item: object, *, exclusion: bool) -> ScopeRule:
    from app.schemas.security_testing import ScopeRulePayload

    assert isinstance(item, ScopeRulePayload)
    try:
        asset = AssetType(item.asset_type)
    except ValueError:
        asset = AssetType.DOMAIN
    return ScopeRule(
        identifier=item.identifier,
        asset_type=asset,
        allow_active_testing=item.allow_active_testing,
        allowed_methods=tuple(item.allowed_methods),
        path_prefix=item.path_prefix,
        instructions=item.instructions,
        is_exclusion=exclusion or item.is_exclusion,
    )
