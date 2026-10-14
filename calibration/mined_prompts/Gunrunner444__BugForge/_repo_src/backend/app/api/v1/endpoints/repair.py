from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.repositories.repair_repo import RepairRepository
from app.repositories.verification_repo import VerificationRepository
from app.schemas.repair import (
    PatchCandidateResponse,
    RepairSessionDetailResponse,
    RepairSessionResponse,
)
from app.schemas.verification import VerificationResponse

router = APIRouter(prefix="/repair", tags=["Automated Repair"])
logger = logging.getLogger(__name__)


@router.get("/{session_id}", response_model=RepairSessionDetailResponse)
async def get_repair_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> RepairSessionDetailResponse:
    repo = RepairRepository(db)
    s = await repo.get_session_by_id(session_id)
    if s is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Repair session not found"
        )
    candidates = [PatchCandidateResponse.from_orm_with_files(c) for c in s.candidates]
    base = RepairSessionResponse.model_validate(s)
    return RepairSessionDetailResponse(**base.model_dump(), candidates=candidates)


@router.get("/{session_id}/candidates", response_model=list[PatchCandidateResponse])
async def list_candidates(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[PatchCandidateResponse]:
    repo = RepairRepository(db)
    s = await repo.get_session_by_id(session_id)
    if s is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Repair session not found"
        )
    return [PatchCandidateResponse.from_orm_with_files(c) for c in s.candidates]


@router.get("/{session_id}/candidates/{candidate_id}", response_model=PatchCandidateResponse)
async def get_candidate(
    session_id: UUID,
    candidate_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> PatchCandidateResponse:
    repo = RepairRepository(db)
    c = await repo.get_candidate(candidate_id)
    if c is None or c.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patch candidate not found"
        )
    return PatchCandidateResponse.from_orm_with_files(c)


@router.post(
    "/{session_id}/candidates/{candidate_id}/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start patch verification for a candidate",
)
async def start_verification(
    session_id: UUID,
    candidate_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> VerificationResponse:
    """Create and start a verification run for the specified patch candidate.

    Returns the verification record immediately; verification runs asynchronously.
    Poll GET /repair/{session_id}/candidates/{candidate_id}/verify to track progress.
    """
    repair_repo = RepairRepository(db)
    candidate = await repair_repo.get_candidate(candidate_id)
    if candidate is None or candidate.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patch candidate not found"
        )

    ver_repo = VerificationRepository(db)
    existing = await ver_repo.get_by_candidate(candidate_id)
    if existing is not None:
        return VerificationResponse.from_orm(existing)

    from app.services.verification_service import VerificationService
    from app.workers.job_runner import FastAPIBackgroundRunner

    repair_session = await repair_repo.get_session_by_id(session_id)
    if repair_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Repair session not found"
        )

    v = await ver_repo.create(
        candidate_id=candidate_id,
        session_id=session_id,
        project_id=repair_session.project_id,
    )
    await db.commit()
    await db.refresh(v)

    svc = VerificationService()
    FastAPIBackgroundRunner(background_tasks).submit(
        svc._run_pipeline,
        verification_id=v.id,
        candidate_id=candidate_id,
    )
    return VerificationResponse.from_orm(v)


@router.get(
    "/{session_id}/candidates/{candidate_id}/verify",
    response_model=VerificationResponse,
    summary="Get verification status for a candidate",
)
async def get_candidate_verification(
    session_id: UUID,
    candidate_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> VerificationResponse:
    repair_repo = RepairRepository(db)
    candidate = await repair_repo.get_candidate(candidate_id)
    if candidate is None or candidate.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patch candidate not found"
        )

    ver_repo = VerificationRepository(db)
    v = await ver_repo.get_by_candidate(candidate_id)
    if v is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No verification found for this candidate"
        )
    return VerificationResponse.from_orm(v)


@router.get(
    "/{session_id}/verifications",
    response_model=list[VerificationResponse],
    summary="List all verifications for a repair session",
)
async def list_session_verifications(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[VerificationResponse]:
    repair_repo = RepairRepository(db)
    s = await repair_repo.get_session_by_id(session_id)
    if s is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Repair session not found"
        )

    ver_repo = VerificationRepository(db)
    verifications = await ver_repo.list_for_session(session_id)
    return [VerificationResponse.from_orm(v) for v in verifications]
