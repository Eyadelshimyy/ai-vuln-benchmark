from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.repositories.test_run_repo import TestRunRepository
from app.schemas.test_run import (
    PaginatedTestResultsResponse,
    TestRunResponse,
)

router = APIRouter(prefix="/test-runs", tags=["Test Runs"])
logger = logging.getLogger(__name__)


def _run_to_response(run: object, include_output: bool = False) -> TestRunResponse:
    from app.models.test_run import TestRun

    r: TestRun = run  # type: ignore[assignment]
    return TestRunResponse(
        id=r.id,
        project_id=r.project_id,
        status=r.status,
        framework=r.framework,
        repository_path=r.repository_path,
        command=r.command,
        exit_code=r.exit_code,
        duration_seconds=r.duration_seconds,
        error_message=r.error_message,
        total_tests=r.total_tests,
        passed=r.passed,
        failed=r.failed,
        skipped=r.skipped,
        errors=r.errors,
        started_at=r.started_at,
        completed_at=r.completed_at,
        created_at=r.created_at,
        stdout=r.stdout if include_output else None,
        stderr=r.stderr if include_output else None,
    )


@router.get("/{run_id}", response_model=TestRunResponse)
async def get_test_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> TestRunResponse:
    repo = TestRunRepository(db)
    run = await repo.get_by_id(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Test run not found")
    return _run_to_response(run, include_output=True)


@router.get("/{run_id}/results", response_model=PaginatedTestResultsResponse)
async def get_test_run_results(
    run_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=2000),
    db: AsyncSession = Depends(get_db),
) -> PaginatedTestResultsResponse:
    repo = TestRunRepository(db)
    run = await repo.get_by_id(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Test run not found")
    results, total = await repo.list_results(run_id, offset=offset, limit=limit)
    return PaginatedTestResultsResponse(
        items=results,  # type: ignore[arg-type]
        total=total,
        offset=offset,
        limit=limit,
    )
