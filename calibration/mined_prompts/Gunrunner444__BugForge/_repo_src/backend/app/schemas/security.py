from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SecurityFindingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID | None
    analysis_id: UUID | None
    title: str
    status: str
    vulnerability_class: str | None
    evidence_tier: str
    confidence: str
    description: str
    hypothesis: str | None
    ai_analysis: str | None
    impact: str | None
    file_path: str | None
    line: int | None
    analyzer: str | None
    rule_ids: str
    observation_refs: str
    asset: str | None
    created_at: datetime
    parser_backend: str | None = None
    node_id: str | None = None
    taint_path: str | None = None
    language: str | None = None
    finding_key: str = ""
    flow_summary: str = ""
    flow_source: str = ""
    flow_sink: str = ""
    field_path: str = ""
    files_crossed: str = ""
    analysis_incomplete: str = ""
    parser_completeness: str = ""
    evidence_summary: str = ""
    related_group: str = ""
    human_review_state: str = "unreviewed"


class SecurityLifecycleRequest(BaseModel):
    """Operator transition against evidence the server already stored.

    ``extra="forbid"`` rejects client fields ``status``, ``evidence``,
    ``finding_key``, ``finding_id``, ``execution_id``, ``verified``, and
    ``reproduced``. The path identifies the finding. The server loads it.
    """

    model_config = ConfigDict(extra="forbid")

    operation: Literal["corroborate", "reproduce", "verify", "human_accept", "reject"]


class PaginatedSecurityFindingsResponse(BaseModel):
    items: list[SecurityFindingResponse]
    total: int
    offset: int
    limit: int


class SecurityAnalyzerInfo(BaseModel):
    language_id: str
    display_name: str
    capabilities: list[str]
    extensions: list[str]
    parser_tier: str = "detection_only"
    parser_backend: str = "none"
    native_available: bool = False
    parser_status: str = "native_parser_unavailable"


class SecurityStatusResponse(BaseModel):
    status: str
    language_analyzers: list[SecurityAnalyzerInfo]
    rule_ids: list[str]
    ai_provider: str
    ai_model: str
    ai_local: bool
    thinking_enabled: bool
    notes: str = (
        "Static and AI hypotheses remain potential or corroborated. "
        "They are not verified vulnerabilities. Active testing uses ScopeGuard "
        "and requires human approval. HackerOne submission is gated behind "
        "human review, dry-run validation, and HUMAN_APPROVED."
    )


class RunSecurityAnalysisRequest(BaseModel):
    use_ai: bool = Field(default=False)
