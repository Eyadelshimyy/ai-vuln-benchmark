from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class GeneratedTestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    project_id: UUID
    target_file: str
    target_symbol: str
    category: str
    rationale: str
    generated_code: str
    confidence: float
    validation_status: str
    validation_error: str | None
    execution_status: str
    execution_output: str | None
    quality_score: float | None
    quality_notes: str | None
    hypothesis_id: UUID | None
    created_at: datetime


class TestGenerationSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    analysis_id: UUID | None
    test_run_id: UUID | None
    debugging_session_id: UUID | None
    status: str
    error_message: str | None
    candidate_count: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class PaginatedGeneratedTestsResponse(BaseModel):
    items: list[GeneratedTestResponse]
    total: int
    offset: int
    limit: int


class PaginatedTestGenSessionsResponse(BaseModel):
    items: list[TestGenerationSessionResponse]
    total: int
    offset: int
    limit: int


class StartTestGenerationRequest(BaseModel):
    analysis_id: UUID | None = None
    test_run_id: UUID | None = None
    debugging_session_id: UUID | None = None
