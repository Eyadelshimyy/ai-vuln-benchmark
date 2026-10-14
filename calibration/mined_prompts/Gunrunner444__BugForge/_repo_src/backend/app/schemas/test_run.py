from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TestResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    test_run_id: UUID
    node_id: str
    test_file: str | None
    test_name: str
    status: str
    duration_seconds: float | None
    traceback: str | None
    stdout: str | None
    stderr: str | None
    skip_reason: str | None


class TestRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    status: str
    framework: str
    repository_path: str
    command: str | None
    exit_code: int | None
    duration_seconds: float | None
    error_message: str | None
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    # stdout/stderr omitted from list responses; available via detail endpoint
    stdout: str | None = None
    stderr: str | None = None


class PaginatedTestRunsResponse(BaseModel):
    items: list[TestRunResponse]
    total: int
    offset: int
    limit: int


class PaginatedTestResultsResponse(BaseModel):
    items: list[TestResultResponse]
    total: int
    offset: int
    limit: int
