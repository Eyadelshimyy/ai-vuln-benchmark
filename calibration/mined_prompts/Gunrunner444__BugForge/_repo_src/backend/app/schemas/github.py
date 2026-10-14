"""Pydantic schemas for GitHub Integration — Phase 9."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, field_validator


class GitHubConnectRequest(BaseModel):
    """Request to connect a GitHub repository to a project."""

    owner: str
    repo: str

    @field_validator("owner", "repo")
    @classmethod
    def _no_slashes(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("owner and repo must not contain path separators")
        v = v.strip()
        if not v:
            raise ValueError("owner and repo must not be empty")
        return v


class GitHubRepositoryResponse(BaseModel):
    """Connected GitHub repository information."""

    id: UUID
    project_id: UUID
    owner: str
    repo: str
    github_id: int | None
    default_branch: str
    html_url: str
    connected: bool
    created_at: datetime
    updated_at: datetime


class GitHubDeliveryResponse(BaseModel):
    """Persistent delivery record returned from the API."""

    id: UUID
    project_id: UUID
    candidate_id: UUID
    verification_id: UUID
    owner: str
    repo: str
    base_branch: str
    delivery_branch: str
    commit_sha: str | None
    pull_request_number: int | None
    pull_request_url: str | None
    # Hash verification fields — confirms delivered == verified
    verified_patch_hash: str
    delivered_patch_hash: str | None
    status: str
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
