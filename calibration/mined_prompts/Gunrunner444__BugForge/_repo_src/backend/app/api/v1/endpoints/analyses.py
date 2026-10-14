from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.analysis import Analysis
from app.repositories.analysis_repo import AnalysisRepository
from app.repositories.finding_repo import FindingRepository
from app.schemas.analysis import (
    AnalysisResponse,
    AnalysisSummarySchema,
    PaginatedEntitiesResponse,
    PaginatedFilesResponse,
    PaginatedImportsResponse,
)
from app.schemas.finding import PaginatedFindingsResponse

router = APIRouter(prefix="/analyses", tags=["Analyses"])
logger = logging.getLogger(__name__)


def _analysis_to_response(analysis: Analysis) -> AnalysisResponse:
    summary = None
    if analysis.summary:
        summary = AnalysisSummarySchema(**analysis.summary)
    return AnalysisResponse(
        id=analysis.id,
        project_id=analysis.project_id,
        status=analysis.status,
        repository_path=analysis.repository_path,
        started_at=analysis.started_at,
        completed_at=analysis.completed_at,
        error_message=analysis.error_message,
        summary=summary,
        created_at=analysis.created_at,
    )


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    repo = AnalysisRepository(db)
    analysis = await repo.get_by_id(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return _analysis_to_response(analysis)


@router.get("/{analysis_id}/files", response_model=PaginatedFilesResponse)
async def list_analysis_files(
    analysis_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> PaginatedFilesResponse:
    repo = AnalysisRepository(db)
    analysis = await repo.get_by_id(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    files, total = await repo.list_files(analysis_id, offset=offset, limit=limit)
    return PaginatedFilesResponse(
        items=files,  # type: ignore[arg-type]
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/{analysis_id}/entities", response_model=PaginatedEntitiesResponse)
async def list_analysis_entities(
    analysis_id: UUID,
    file_id: UUID | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> PaginatedEntitiesResponse:
    repo = AnalysisRepository(db)
    analysis = await repo.get_by_id(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    entities, total = await repo.list_entities(
        analysis_id, file_id=file_id, offset=offset, limit=limit
    )
    return PaginatedEntitiesResponse(
        items=entities,  # type: ignore[arg-type]
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/{analysis_id}/imports", response_model=PaginatedImportsResponse)
async def list_analysis_imports(
    analysis_id: UUID,
    file_id: UUID | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> PaginatedImportsResponse:
    repo = AnalysisRepository(db)
    analysis = await repo.get_by_id(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    imports, total = await repo.list_imports(
        analysis_id, file_id=file_id, offset=offset, limit=limit
    )
    return PaginatedImportsResponse(
        items=imports,  # type: ignore[arg-type]
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/{analysis_id}/findings", response_model=PaginatedFindingsResponse)
async def list_analysis_findings(
    analysis_id: UUID,
    severity: str | None = Query(
        None, description="Filter by severity (info/low/medium/high/critical)"
    ),
    category: str | None = Query(None, description="Filter by rule category"),
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> PaginatedFindingsResponse:
    analysis_repo = AnalysisRepository(db)
    if await analysis_repo.get_by_id(analysis_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    finding_repo = FindingRepository(db)
    findings, total = await finding_repo.list_for_analysis(
        analysis_id, severity=severity, category=category, offset=offset, limit=limit
    )
    return PaginatedFindingsResponse(
        items=findings,  # type: ignore[arg-type]
        total=total,
        offset=offset,
        limit=limit,
    )
