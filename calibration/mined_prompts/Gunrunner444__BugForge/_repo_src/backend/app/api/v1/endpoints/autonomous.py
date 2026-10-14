"""Autonomous analysis run API endpoints (v1.1.0).

GET  /api/v1/autonomous/runs        — list runs
GET  /api/v1/autonomous/runs/{id}   — get one run
POST /api/v1/autonomous/runs/{id}/cancel — cancel a running analysis
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.repositories.discovery_repo import AutonomousAnalysisRunRepository
from app.schemas.discovery import AutonomousRunResponse, AutonomousRunsListResponse

router = APIRouter(prefix="/autonomous", tags=["Autonomous Analysis"])
logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "inconclusive"})


@router.get("/runs", response_model=AutonomousRunsListResponse)
async def list_autonomous_runs(
    run_status: str | None = Query(None, alias="status"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> AutonomousRunsListResponse:
    repo = AutonomousAnalysisRunRepository(db)
    runs, total = await repo.list_all(status=run_status, limit=limit, offset=offset)
    return AutonomousRunsListResponse(
        items=[AutonomousRunResponse.model_validate(r) for r in runs],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/runs/{run_id}", response_model=AutonomousRunResponse)
async def get_autonomous_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> AutonomousRunResponse:
    repo = AutonomousAnalysisRunRepository(db)
    run = await repo.get_by_id(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Autonomous analysis run not found")
    return AutonomousRunResponse.model_validate(run)


@router.post(
    "/runs/{run_id}/cancel",
    response_model=AutonomousRunResponse,
    status_code=status.HTTP_200_OK,
)
async def cancel_autonomous_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> AutonomousRunResponse:
    """Request cancellation of an in-progress autonomous analysis run.

    If the run is already in a terminal state, this is a no-op.
    """
    repo = AutonomousAnalysisRunRepository(db)
    run = await repo.get_by_id(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Autonomous analysis run not found")

    if run.status in _TERMINAL_STATUSES:
        return AutonomousRunResponse.model_validate(run)

    await repo.cancel(run_id, reason="user_requested")
    await db.commit()
    await db.refresh(run)
    logger.info("Autonomous run %s cancelled by user request.", run_id)
    return AutonomousRunResponse.model_validate(run)
