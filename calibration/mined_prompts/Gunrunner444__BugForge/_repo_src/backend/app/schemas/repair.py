from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class PatchCandidateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    rank: int
    patch_provider: str | None
    patch_model: str | None
    patch_plan: str | None
    patch_diff: str | None
    changed_files: list[str]
    status: str
    validation_status: str | None
    validation_error: str | None
    pre_patch_reproduced: bool | None
    post_patch_reproduced: bool | None
    bug_fixed: bool | None
    existing_tests_total: int | None
    existing_tests_passed: int | None
    existing_tests_failed: int | None
    no_regressions: bool | None
    regression_count: int
    new_static_findings: int
    score: float | None
    disposition: str
    created_at: datetime
    completed_at: datetime | None

    @classmethod
    def from_orm_with_files(cls, candidate: object) -> PatchCandidateResponse:
        """Build from ORM object, parsing changed_files_json."""
        from app.models.repair import PatchCandidate

        assert isinstance(candidate, PatchCandidate)
        files: list[str] = []
        if candidate.changed_files_json:
            try:
                files = json.loads(candidate.changed_files_json)
            except (json.JSONDecodeError, TypeError):
                pass
        base = cls.model_validate(candidate)
        return base.model_copy(update={"changed_files": files})


class RepairSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    debugging_session_id: UUID | None
    hypothesis_id: UUID | None
    reproduction_session_id: UUID | None
    status: str
    total_candidates: int
    best_candidate_id: UUID | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class RepairSessionDetailResponse(RepairSessionResponse):
    candidates: list[PatchCandidateResponse] = []


class PaginatedRepairSessionsResponse(BaseModel):
    items: list[RepairSessionResponse]
    total: int
    offset: int
    limit: int


class StartRepairRequest(BaseModel):
    hypothesis_id: UUID | None = None
    reproduction_session_id: UUID | None = None
    debugging_session_id: UUID | None = None
    max_candidates: int = 1
