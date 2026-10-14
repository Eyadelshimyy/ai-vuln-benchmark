"""API schemas for authorized security testing sessions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScopeRulePayload(BaseModel):
    identifier: str
    asset_type: str = "domain"
    allow_active_testing: bool = False
    allowed_methods: list[str] = Field(default_factory=list)
    path_prefix: str = ""
    is_exclusion: bool = False
    instructions: str = ""


class CreateTestingSessionRequest(BaseModel):
    project_id: str
    mode: str = "lab"
    program_name: str | None = "local-lab"
    includes: list[ScopeRulePayload] = Field(default_factory=list)
    excludes: list[ScopeRulePayload] = Field(default_factory=list)
    allow_active_testing: bool = False
    dry_run: bool = True
    allowed_methods: list[str] = Field(default_factory=lambda: ["GET", "HEAD", "OPTIONS"])
    max_requests: int = 50
    requests_per_second: float = 1.0


class ApprovalRequest(BaseModel):
    kind: str
    operator: str
    note: str = ""


class AuthorizeRequest(BaseModel):
    target: str
    method: str = "GET"
    tool: str = "manual"
    active: bool = True


class AuthorizationResponse(BaseModel):
    allowed: bool
    reason: str
    target: str
    method: str
    tool: str
    program: str | None
    dry_run: bool
    approval_required: bool
    approval_state: str
    matched_rule: str | None = None
    request_limit: int | None = None


class TestingSessionResponse(BaseModel):
    session: dict[str, Any]
    tools: list[str]
    notes: str = (
        "There is no unrestricted scan action. Every active operation shows "
        "its target, scope decision, tool, request limit, and approval state. "
        "HackerOne submission is gated: dry-run never creates a report, and "
        "real submission requires HUMAN_APPROVED plus a human operator."
    )


class AuditLogResponse(BaseModel):
    entries: list[dict[str, Any]]
    chain_valid: bool
