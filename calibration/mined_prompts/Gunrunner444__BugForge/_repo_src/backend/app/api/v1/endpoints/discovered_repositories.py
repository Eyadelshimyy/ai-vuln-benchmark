"""Discovered repository API endpoints (v1.1.0).

GET  /api/v1/repositories/discovered            — list candidates
GET  /api/v1/repositories/discovered/counts     — status counts
GET  /api/v1/repositories/discovered/{id}       — get one candidate
POST /api/v1/repositories/discovered/{id}/analyze — queue autonomous analysis
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database import get_db
from app.repositories.discovery_repo import (
    AutonomousAnalysisRunRepository,
    RepositoryCandidateRepository,
)
from app.schemas.discovery import (
    AutonomousRunResponse,
    CandidateStatusCounts,
    RepositoryCandidateResponse,
    RepositoryCandidatesListResponse,
    TriggerAnalysisRequest,
)

router = APIRouter(prefix="/repositories/discovered", tags=["Discovered Repositories"])
logger = logging.getLogger(__name__)


async def _run_autonomous_analysis(
    candidate_id: UUID,
    run_id: UUID,
    force_rescan: bool,
) -> None:
    """Background task: run the autonomous analysis pipeline."""
    from app.database import async_session_factory
    from app.repositories.discovery_repo import RepositoryCandidateRepository
    from app.services.autonomous_service import AutonomousAnalysisService

    async with async_session_factory() as session:
        cand_repo = RepositoryCandidateRepository(session)
        candidate = await cand_repo.get_by_id(candidate_id)
        if candidate is None:
            logger.error("Autonomous analysis %s: candidate %s not found", run_id, candidate_id)
            return

    svc = AutonomousAnalysisService(settings)
    await svc.run(candidate, run_id, force_rescan=force_rescan)


@router.get("", response_model=RepositoryCandidatesListResponse)
async def list_candidates(
    eligibility_status: str | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> RepositoryCandidatesListResponse:
    repo = RepositoryCandidateRepository(db)
    candidates, total = await repo.list_all(status=eligibility_status, limit=limit, offset=offset)
    return RepositoryCandidatesListResponse(
        items=[RepositoryCandidateResponse.model_validate(c) for c in candidates],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/counts", response_model=CandidateStatusCounts)
async def get_candidate_counts(
    db: AsyncSession = Depends(get_db),
) -> CandidateStatusCounts:
    repo = RepositoryCandidateRepository(db)
    counts = await repo.count_by_status()
    return CandidateStatusCounts(counts=counts)


@router.get("/{candidate_id}", response_model=RepositoryCandidateResponse)
async def get_candidate(
    candidate_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> RepositoryCandidateResponse:
    repo = RepositoryCandidateRepository(db)
    candidate = await repo.get_by_id(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Repository candidate not found")
    return RepositoryCandidateResponse.model_validate(candidate)


@router.post(
    "/{candidate_id}/analyze",
    response_model=AutonomousRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_analysis(
    candidate_id: UUID,
    request: TriggerAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> AutonomousRunResponse:
    """Queue an autonomous analysis run for the specified candidate.

    Returns immediately with the new run record; analysis continues in
    the background.
    """
    cand_repo = RepositoryCandidateRepository(db)
    candidate = await cand_repo.get_by_id(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Repository candidate not found")

    if candidate.eligibility_status not in {"eligible", "queued", "completed", "failed"}:
        if not request.force_rescan:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Candidate eligibility_status is '{candidate.eligibility_status}'; "
                    "only eligible, completed, or failed candidates can be (re-)analyzed. "
                    "Set force_rescan=true to override."
                ),
            )

    run_repo = AutonomousAnalysisRunRepository(db)

    # Prevent duplicate concurrent runs
    if not request.force_rescan and await run_repo.has_active_run_for_candidate(candidate_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An active analysis run already exists for this candidate.",
        )

    run = await run_repo.create(
        candidate_id=candidate_id,
        ai_provider=settings.ai_provider,
        ai_model=settings.ai_model,
        is_local_ai=settings.is_local_ai(),
    )
    await db.commit()
    await db.refresh(run)

    background_tasks.add_task(_run_autonomous_analysis, candidate_id, run.id, request.force_rescan)
    logger.info("Autonomous run %s queued for candidate %s.", run.id, candidate_id)
    return AutonomousRunResponse.model_validate(run)
