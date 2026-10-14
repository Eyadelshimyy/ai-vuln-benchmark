from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.repositories.reproduction_repo import ReproductionRepository
from app.schemas.reproduction import (
    BugReproductionDetailResponse,
    BugReproductionSessionResponse,
    ReproductionAttemptResponse,
)

router = APIRouter(prefix="/reproduction", tags=["Bug Reproduction"])
logger = logging.getLogger(__name__)


@router.get("/{session_id}", response_model=BugReproductionDetailResponse)
async def get_reproduction_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> BugReproductionDetailResponse:
    repo = ReproductionRepository(db)
    s = await repo.get_by_id(session_id)
    if s is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Reproduction session not found"
        )
    return BugReproductionDetailResponse(
        **BugReproductionSessionResponse.model_validate(s).model_dump(),
        attempts=[ReproductionAttemptResponse.model_validate(a) for a in s.attempts],
    )


@router.get("/{session_id}/attempts", response_model=list[ReproductionAttemptResponse])
async def list_attempts(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[ReproductionAttemptResponse]:
    repo = ReproductionRepository(db)
    s = await repo.get_by_id(session_id)
    if s is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Reproduction session not found"
        )
    return [ReproductionAttemptResponse.model_validate(a) for a in s.attempts]
