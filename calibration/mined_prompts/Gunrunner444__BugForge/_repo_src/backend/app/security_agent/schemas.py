"""Structured tool calls, hypotheses, and research plans. Typed before execution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from app.security_agent.states import HypothesisStatus


class ToolCallRequest(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    @field_validator("tool")
    @classmethod
    def tool_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tool is required")
        return cleaned


class HttpRequestArgs(BaseModel):
    method: str = "GET"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    content: str | None = None
    active: bool = True


class BrowserNavigateArgs(BaseModel):
    url: str
    capture_screenshot: bool = True
    capture_console: bool = True


class SourceInspectArgs(BaseModel):
    path: str
    query: str = ""
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def line_range(self) -> SourceInspectArgs:
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError("end_line must be >= start_line")
        return self


class EvidenceInspectArgs(BaseModel):
    evidence_id: str


class ScanArgs(BaseModel):
    target: str
    template: str | None = None
    ingest_only: bool = False
    alerts_json: str | None = None


class FuzzArgs(BaseModel):
    url: str
    method: str = "GET"
    param: str = "q"
    count: int = Field(default=3, ge=1, le=50)
    kinds: list[str] = Field(default_factory=lambda: ["query"])
    headers: dict[str, str] = Field(default_factory=dict)
    content: str | None = None
    requests_per_second: float | None = Field(default=None, gt=0)
    concurrency: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_payload_size: int | None = Field(default=None, ge=1)


class ReproduceActionArgs(BaseModel):
    method: str = "GET"
    url: str
    content: str | None = None
    expected_status: int | None = None
    expected_body_contains: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class ReproduceArgs(BaseModel):
    plan_id: str | None = None
    url: str | None = None
    method: str = "GET"
    expected_result: str = ""
    expected_body_contains: str | None = None
    expected_status: int | None = None
    preconditions: list[str] = Field(default_factory=list)
    setup: str = ""
    cleanup: str = ""
    reproducibility_count: int = Field(default=1, ge=1, le=5)
    actions: list[ReproduceActionArgs] = Field(default_factory=list)


class ProxyEvidenceArgs(BaseModel):
    exchange_id: str = ""


class ApiTestArgs(BaseModel):
    url: str | None = None
    method: str = "GET"
    spec_path: str | None = None
    spec_format: str | None = None
    execute: bool = True
    max_tests: int = Field(default=8, ge=1, le=40)


class ExploratoryTestArgs(BaseModel):
    """What to test. BugForge chooses the command, network, and sandbox."""

    model_config = {"extra": "forbid"}

    hypothesis_id: str
    language: str
    framework: str
    target_file: str = ""
    target_symbol: str
    test_code: str
    expected_behavior: str
    oracle: str
    reason: str = ""
    parent_attempt_id: str = ""
    project_id: str = ""
    session_id: str = ""
    confidence: str = "low"
    follow_up: str = ""

    @field_validator("language", "framework", "oracle")
    @classmethod
    def short_token(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned or len(cleaned) > 40 or not re.fullmatch(r"[a-z0-9_-]+", cleaned):
            raise ValueError("invalid exploratory token")
        return cleaned

    @field_validator("confidence")
    @classmethod
    def confidence_value(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"low", "medium", "high"}:
            raise ValueError("confidence must be low, medium, or high")
        return cleaned

    @field_validator("follow_up")
    @classmethod
    def follow_up_kind(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            return ""
        allowed = {"confirmation", "falsification", "boundary", "negative_control", "alternative"}
        if cleaned not in allowed:
            raise ValueError("unknown follow-up kind")
        return cleaned


class HypothesisUpdateArgs(BaseModel):
    hypothesis_id: str
    status: str
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""


TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "http_request": HttpRequestArgs,
    "browser_navigate": BrowserNavigateArgs,
    "source_inspect": SourceInspectArgs,
    "evidence_inspect": EvidenceInspectArgs,
    "zap_scan": ScanArgs,
    "nuclei_scan": ScanArgs,
    "fuzz": FuzzArgs,
    "reproduce": ReproduceArgs,
    "proxy_evidence": ProxyEvidenceArgs,
    "api_test": ApiTestArgs,
    "exploratory_test": ExploratoryTestArgs,
}

_ALLOWED_PLANNER_KINDS = frozenset(
    {
        "analyze",
        "hypothesis",
        "plan",
        "tool",
        "reproduce",
        "complete",
        "update_hypothesis",
        "draft_report",
        "paused",
        "stopped",
    }
)

_FORBIDDEN_PLANNER_KINDS = frozenset(
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
    }
)

_CONFIDENCE_VALUES = frozenset({"low", "medium", "high"})


class PlannerOutput(BaseModel):
    """Untrusted model output. Validated before any agent state change."""

    kind: str
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    title: str | None = None
    vulnerability_class: str | None = None
    target: str | None = None
    confidence: str | None = None
    severity: str | None = None
    impact: str | None = None
    hypothesis_id: str | None = None
    status: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    plan_steps: list[dict[str, Any]] = Field(default_factory=list)
    note: str = ""

    @field_validator("kind")
    @classmethod
    def kind_allowed(cls, value: str) -> str:
        cleaned = (value or "").strip().lower()
        if not cleaned:
            raise ValueError("kind is required")
        if cleaned in _FORBIDDEN_PLANNER_KINDS:
            raise ValueError(f"forbidden_kind:{cleaned}")
        if cleaned not in _ALLOWED_PLANNER_KINDS:
            raise ValueError(f"unknown_kind:{cleaned}")
        return cleaned

    @field_validator("confidence")
    @classmethod
    def confidence_allowed(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        cleaned = value.strip().lower()
        if cleaned not in _CONFIDENCE_VALUES:
            raise ValueError("confidence must be low, medium, or high")
        return cleaned

    @field_validator("status")
    @classmethod
    def status_not_verified(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        cleaned = value.strip().lower()
        if cleaned == HypothesisStatus.VERIFIED.value:
            raise ValueError("AI cannot set VERIFIED")
        return cleaned

    @model_validator(mode="after")
    def consistent(self) -> PlannerOutput:
        if self.kind == "tool" and not self.tool:
            raise ValueError("tool kind requires a tool name")
        if self.kind == "hypothesis" and not (self.title or self.vulnerability_class):
            raise ValueError("hypothesis requires title or vulnerability_class")
        if self.kind == "update_hypothesis" and not self.hypothesis_id:
            raise ValueError("update_hypothesis requires hypothesis_id")
        return self


class DraftReportCandidate(BaseModel):
    kind: Literal["DRAFT_REPORT_CANDIDATE"] = "DRAFT_REPORT_CANDIDATE"
    verified_finding_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    reproduction_ids: list[str] = Field(default_factory=list)
    target: str
    scope_snapshot: dict[str, Any] = Field(default_factory=dict)
    candidate_weakness: str = ""
    candidate_severity: str = ""
    cannot_approve: bool = True
    cannot_submit: bool = True
    cannot_verify: bool = True
    cannot_change_scope: bool = True


@dataclass
class ResearchHypothesis:
    title: str
    vulnerability_class: str
    target: str
    reason: str
    confidence: str = "low"
    severity: str = "medium"
    impact: str = ""
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    suggested_next_action: str = ""
    status: HypothesisStatus = HypothesisStatus.OPEN
    reproducibility: str = "unknown"
    evidence_strength: int = 0
    id: str = field(default_factory=lambda: uuid4().hex)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "vulnerability_class": self.vulnerability_class,
            "target": self.target,
            "reason": self.reason,
            "confidence": self.confidence,
            "severity": self.severity,
            "impact": self.impact,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "suggested_next_action": self.suggested_next_action,
            "status": self.status.value,
            "reproducibility": self.reproducibility,
            "evidence_strength": self.evidence_strength,
        }


@dataclass
class ResearchPlanStep:
    order: int
    action: str
    tool: str | None = None
    target: str = ""
    note: str = ""


@dataclass
class ResearchPlan:
    steps: tuple[ResearchPlanStep, ...] = ()
    current: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "current": self.current,
            "steps": [
                {
                    "order": step.order,
                    "action": step.action,
                    "tool": step.tool,
                    "target": step.target,
                    "note": step.note,
                }
                for step in self.steps
            ],
        }
