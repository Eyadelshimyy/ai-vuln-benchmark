from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class FindingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    analysis_id: UUID
    category: str
    severity: str
    confidence: str
    file_path: str
    line: int
    end_line: int
    column: int | None
    message: str
    explanation: str
    analyzer: str
    evidence: str
    suggested_fix: str
    created_at: datetime
    catalog: str = "code_quality"
    language: str | None = None
    parser_backend: str | None = None
    node_id: str | None = None


class PaginatedFindingsResponse(BaseModel):
    items: list[FindingResponse]
    total: int
    offset: int
    limit: int
