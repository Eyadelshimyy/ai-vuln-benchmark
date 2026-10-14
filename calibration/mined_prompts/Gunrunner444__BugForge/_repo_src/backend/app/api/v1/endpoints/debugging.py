from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.repositories.debugging_repo import DebuggingRepository
from app.schemas.debugging import (
    DebuggingHypothesisResponse,
    DebuggingSessionDetailResponse,
    DebuggingSessionResponse,
)

router = APIRouter(prefix="/debugging", tags=["Debugging"])
logger = logging.getLogger(__name__)


def _session_to_response(ds: object) -> DebuggingSessionResponse:
    from app.models.debugging import DebuggingSession

    s: DebuggingSession = ds  # type: ignore[assignment]
    return DebuggingSessionResponse(
        id=s.id,
        project_id=s.project_id,
        analysis_id=s.analysis_id,
        test_run_id=s.test_run_id,
        status=s.status,
        error_message=s.error_message,
        created_at=s.created_at,
        started_at=s.started_at,
        completed_at=s.completed_at,
        hypothesis_count=len(s.hypotheses) if hasattr(s, "hypotheses") else 0,
    )


@router.get("/{session_id}", response_model=DebuggingSessionDetailResponse)
async def get_debugging_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DebuggingSessionDetailResponse:
    repo = DebuggingRepository(db)
    ds = await repo.get_by_id(session_id)
    if ds is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Debugging session not found"
        )

    hypotheses = [DebuggingHypothesisResponse.from_orm_row(h) for h in ds.hypotheses]
    from app.schemas.debugging import AIModelCallResponse

    calls = [AIModelCallResponse.model_validate(c) for c in ds.ai_calls]

    return DebuggingSessionDetailResponse(
        id=ds.id,
        project_id=ds.project_id,
        analysis_id=ds.analysis_id,
        test_run_id=ds.test_run_id,
        status=ds.status,
        error_message=ds.error_message,
        created_at=ds.created_at,
        started_at=ds.started_at,
        completed_at=ds.completed_at,
        hypothesis_count=len(hypotheses),
        hypotheses=hypotheses,
        ai_calls=calls,
    )


@router.get("/{session_id}/hypotheses", response_model=list[DebuggingHypothesisResponse])
async def list_hypotheses(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[DebuggingHypothesisResponse]:
    repo = DebuggingRepository(db)
    ds = await repo.get_by_id(session_id)
    if ds is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Debugging session not found"
        )
    return [DebuggingHypothesisResponse.from_orm_row(h) for h in ds.hypotheses]
