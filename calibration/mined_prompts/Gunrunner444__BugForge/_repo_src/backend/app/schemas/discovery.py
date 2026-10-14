"""Pydantic schemas for discovery / autonomous analysis API (v1.1.0)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# DiscoveryRun schemas
# ---------------------------------------------------------------------------


class DiscoveryRunResponse(_Base):
    id: UUID
    status: str
    search_criteria: str | None
    discovered_count: int
    eligible_count: int
    rejected_count: int
    github_api_requests: int
    started_at: datetime
    completed_at: datetime | None
    duration_seconds: float | None
    error_message: str | None
    created_at: datetime


class DiscoveryRunsListResponse(_Base):
    items: list[DiscoveryRunResponse]
    total: int
    offset: int
    limit: int


class TriggerDiscoveryRequest(BaseModel):
    """Body for POST /api/v1/discovery/run."""

    max_pages: int = Field(default=5, ge=1, le=20)


# ---------------------------------------------------------------------------
# RepositoryCandidate schemas
# ---------------------------------------------------------------------------


class RepositoryCandidateResponse(_Base):
    id: UUID
    github_repo_id: int
    owner: str
    name: str
    full_name: str
    html_url: str
    stars: int
    is_fork: bool
    is_archived: bool
    default_branch: str
    primary_language: str | None
    license_key: str | None
    size_kb: int
    open_issues: int
    topics: str | None  # JSON array string
    description: str | None
    last_updated_at: datetime | None
    last_pushed_at: datetime | None
    eligibility_status: str
    eligibility_score: float | None
    rejection_reason: str | None
    safety_classification: str | None
    safety_detail: str | None
    analysis_status: str | None
    last_analyzed_commit: str | None
    last_analyzed_at: datetime | None
    discovered_at: datetime
    created_at: datetime
    updated_at: datetime


class RepositoryCandidatesListResponse(_Base):
    items: list[RepositoryCandidateResponse]
    total: int
    offset: int
    limit: int


class CandidateStatusCounts(_Base):
    counts: dict[str, int]


class TriggerAnalysisRequest(BaseModel):
    """Body for POST /api/v1/repositories/discovered/{id}/analyze."""

    force_rescan: bool = False


# ---------------------------------------------------------------------------
# AutonomousAnalysisRun schemas
# ---------------------------------------------------------------------------


class AutonomousRunResponse(_Base):
    id: UUID
    candidate_id: UUID
    project_id: UUID | None
    status: str
    current_stage: str | None
    commit_sha: str | None
    ai_provider: str | None
    ai_model: str | None
    is_local_ai: bool
    static_findings_count: int
    ai_hypotheses_count: int
    validated_findings_count: int
    rejected_findings_count: int
    tests_generated: int
    tests_executed: int
    repairs_generated: int
    repairs_verified: int
    started_at: datetime | None
    completed_at: datetime | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class AutonomousRunsListResponse(_Base):
    items: list[AutonomousRunResponse]
    total: int
    offset: int
    limit: int


# ---------------------------------------------------------------------------
# AI health schemas
# ---------------------------------------------------------------------------


class AIStatusResponse(BaseModel):
    provider: str
    model: str
    is_local: bool
    reachable: bool
    configured: bool
    model_available: bool | None
    error: str | None
    capabilities: list[str] | None
    thinking_enabled: bool = False
    language_analyzers: list[str] = []
    security_analysis_status: str = "available"


class AITestRequest(BaseModel):
    prompt: str = Field(
        default="Reply with only the word: ok",
        max_length=500,
    )


class AITestResponse(BaseModel):
    provider: str
    model: str
    response: str
    duration_seconds: float
    error: str | None


# ---------------------------------------------------------------------------
# Discovery settings schemas
# ---------------------------------------------------------------------------


class DiscoverySettingsResponse(BaseModel):
    discovery_mode: str
    discovery_interval_hours: int
    discovery_min_stars: int
    discovery_max_stars: int
    discovery_languages: str
    discovery_require_license: bool
    discovery_skip_forks: bool
    discovery_skip_archived: bool
    discovery_max_staleness_days: int
    discovery_daily_repo_limit: int
    discovery_max_concurrent: int
    discovery_max_size_kb: int
    discovery_excluded_topics: str
    discovery_excluded_owners: str
    safety_max_repo_size_kb: int
    safety_max_file_count: int
    safety_allow_docker_exec: bool
    safety_allow_sandbox_network: bool
    safety_allow_dep_install: bool
