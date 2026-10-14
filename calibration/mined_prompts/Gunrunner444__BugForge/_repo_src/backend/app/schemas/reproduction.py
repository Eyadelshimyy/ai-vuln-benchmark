from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ReproductionAttemptResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    attempt_number: int
    command: str | None
    input_description: str | None
    reproducer_code: str | None
    exit_code: int | None
    stdout: str | None
    stderr: str | None
    traceback: str | None
    duration_seconds: float | None
    timed_out: bool
    reproduced: bool
    classification: str
    evidence_matched: str | None
    created_at: datetime


class BugReproductionSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    debugging_session_id: UUID | None
    hypothesis_id: UUID | None
    generated_test_id: UUID | None
    status: str
    attempt_count: int
    successful_attempts: int
    total_attempts: int
    reproducibility_rate: float | None
    final_classification: str | None
    strategy_summary: str | None
    error_message: str | None
    target_behavior: str | None
    expected_failure_pattern: str | None
    observable_evidence: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class BugReproductionDetailResponse(BugReproductionSessionResponse):
    attempts: list[ReproductionAttemptResponse] = []


class PaginatedReproductionSessionsResponse(BaseModel):
    items: list[BugReproductionSessionResponse]
    total: int
    offset: int
    limit: int


class StartReproductionRequest(BaseModel):
    hypothesis_id: UUID | None = None
    generated_test_id: UUID | None = None
    debugging_session_id: UUID | None = None
    total_attempts: int = 3
