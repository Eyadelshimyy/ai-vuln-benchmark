from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class LanguageStatsSchema(BaseModel):
    language: str
    file_count: int
    percentage: float


class FrameworkDetectionSchema(BaseModel):
    name: str
    language: str
    confidence: float
    evidence: list[str]


class AnalysisSummarySchema(BaseModel):
    total_files: int
    source_files: int
    test_files: int
    ignored_files: int
    total_entities: int
    total_imports: int
    total_findings: int = 0
    security_findings: int = 0
    languages: list[LanguageStatsSchema]
    frameworks: list[FrameworkDetectionSchema]
    analysis_duration_seconds: float | None = None
    language_capabilities: list[dict[str, Any]] = []


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    status: str
    repository_path: str
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    summary: AnalysisSummarySchema | None
    created_at: datetime


class FileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    relative_path: str
    file_type: str
    language: str | None
    size_bytes: int
    line_count: int
    has_errors: bool
    parser_backend: str | None = None
    parser_tier: str | None = None
    error_count: int = 0


class EntityResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    file_id: UUID
    entity_type: str
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    docstring: str | None
    is_async: bool
    decorators: list[Any] | None
    parameters: list[Any] | None
    return_annotation: str | None
    parent_name: str | None


class ImportRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    file_id: UUID
    module_name: str
    imported_name: str | None
    alias: str | None
    import_type: str
    line_number: int
    is_from_import: bool


class PaginatedFilesResponse(BaseModel):
    items: list[FileResponse]
    total: int
    offset: int
    limit: int


class PaginatedEntitiesResponse(BaseModel):
    items: list[EntityResponse]
    total: int
    offset: int
    limit: int


class PaginatedImportsResponse(BaseModel):
    items: list[ImportRecordResponse]
    total: int
    offset: int
    limit: int
