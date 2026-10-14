"""Patch Verification API endpoints — Phase 8."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.repositories.verification_repo import VerificationRepository
from app.schemas.verification import VerificationResponse

router = APIRouter(prefix="/verification", tags=["Patch Verification"])
logger = logging.getLogger(__name__)


@router.get(
    "/{verification_id}",
    response_model=VerificationResponse,
    summary="Retrieve a verification record by ID",
)
async def get_verification_by_id(
    verification_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> VerificationResponse:
    repo = VerificationRepository(db)
    v = await repo.get_by_id(verification_id)
    if v is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Verification not found")
    return VerificationResponse.from_orm(v)


@router.get(
    "/by-candidate/{candidate_id}",
    response_model=VerificationResponse,
    summary="Retrieve a verification record by candidate ID",
)
async def get_verification_by_candidate(
    candidate_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> VerificationResponse:
    repo = VerificationRepository(db)
    v = await repo.get_by_candidate(candidate_id)
    if v is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Verification not found")
    return VerificationResponse.from_orm(v)
