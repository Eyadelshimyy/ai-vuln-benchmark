from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Human-readable project name")
    description: str | None = Field(None, max_length=2048)
    repository_path: str = Field(..., description="Absolute local path to the repository directory")


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    repository_path: str
    created_at: datetime
    updated_at: datetime
    latest_analysis_id: UUID | None = None
    latest_analysis_status: str | None = None


class ProjectListResponse(BaseModel):
    items: list[ProjectResponse]
    total: int
