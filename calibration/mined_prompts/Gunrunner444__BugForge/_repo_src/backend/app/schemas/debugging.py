from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AIModelCallResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    duration_seconds: float
    success: bool
    error_message: str | None
    created_at: datetime


class DebuggingHypothesisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    root_cause: str
    confidence: float
    confidence_label: str
    # Stored as JSON strings in DB; parsed here for clean API responses
    affected_files: list[str]
    affected_symbols: list[str]
    evidence_summary: list[str]
    contradictory_evidence: list[str]
    reproduction_strategy: str
    recommended_tests: list[str]
    explanation: str
    ai_provider: str
    ai_model: str
    created_at: datetime

    @classmethod
    def from_orm_row(cls, row: object) -> DebuggingHypothesisResponse:
        """Parse JSON list fields from DB row."""
        from app.models.debugging import DebuggingHypothesis

        r: DebuggingHypothesis = row  # type: ignore[assignment]

        def _parse(v: str) -> list[str]:
            try:
                result = json.loads(v)
                return result if isinstance(result, list) else []
            except (json.JSONDecodeError, TypeError):
                return []

        return cls(
            id=r.id,
            session_id=r.session_id,
            root_cause=r.root_cause,
            confidence=r.confidence,
            confidence_label=r.confidence_label,
            affected_files=_parse(r.affected_files),
            affected_symbols=_parse(r.affected_symbols),
            evidence_summary=_parse(r.evidence_summary),
            contradictory_evidence=_parse(r.contradictory_evidence),
            reproduction_strategy=r.reproduction_strategy,
            recommended_tests=_parse(r.recommended_tests),
            explanation=r.explanation,
            ai_provider=r.ai_provider,
            ai_model=r.ai_model,
            created_at=r.created_at,
        )


class DebuggingSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    analysis_id: UUID | None
    test_run_id: UUID | None
    status: str
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    hypothesis_count: int = 0


class DebuggingSessionDetailResponse(DebuggingSessionResponse):
    hypotheses: list[DebuggingHypothesisResponse] = []
    ai_calls: list[AIModelCallResponse] = []


class PaginatedDebuggingSessionsResponse(BaseModel):
    items: list[DebuggingSessionResponse]
    total: int
    offset: int
    limit: int


class StartDebuggingRequest(BaseModel):
    analysis_id: UUID | None = None
    test_run_id: UUID | None = None
