from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.repositories.test_generation_repo import TestGenerationRepository
from app.schemas.test_generation import (
    PaginatedGeneratedTestsResponse,
    TestGenerationSessionResponse,
)

router = APIRouter(prefix="/test-generation", tags=["Test Generation"])
logger = logging.getLogger(__name__)


@router.get("/{session_id}", response_model=TestGenerationSessionResponse)
async def get_test_generation_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> TestGenerationSessionResponse:
    repo = TestGenerationRepository(db)
    s = await repo.get_session(session_id)
    if s is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Test generation session not found"
        )
    return TestGenerationSessionResponse.model_validate(s)


@router.get("/{session_id}/tests", response_model=PaginatedGeneratedTestsResponse)
async def list_generated_tests(
    session_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> PaginatedGeneratedTestsResponse:
    repo = TestGenerationRepository(db)
    if await repo.get_session(session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Test generation session not found"
        )
    tests, total = await repo.list_tests(session_id, offset=offset, limit=limit)
    return PaginatedGeneratedTestsResponse(
        items=tests,  # type: ignore[arg-type]
        total=total,
        offset=offset,
        limit=limit,
    )
