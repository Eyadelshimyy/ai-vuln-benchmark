from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.test_generation import GeneratedTest, TestGenerationSession


class TestGenerationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_session(
        self,
        project_id: UUID,
        analysis_id: UUID | None,
        test_run_id: UUID | None,
        debugging_session_id: UUID | None,
    ) -> TestGenerationSession:
        s = TestGenerationSession(
            project_id=project_id,
            analysis_id=analysis_id,
            test_run_id=test_run_id,
            debugging_session_id=debugging_session_id,
            status="pending",
        )
        self.session.add(s)
        await self.session.flush()
        await self.session.refresh(s)
        return s

    async def get_session(self, session_id: UUID) -> TestGenerationSession | None:
        return await self.session.get(TestGenerationSession, session_id)

    async def list_sessions(
        self, project_id: UUID, offset: int = 0, limit: int = 20
    ) -> tuple[list[TestGenerationSession], int]:
        cnt = await self.session.execute(
            select(func.count())
            .select_from(TestGenerationSession)
            .where(TestGenerationSession.project_id == project_id)
        )
        total: int = cnt.scalar_one()
        result = await self.session.execute(
            select(TestGenerationSession)
            .where(TestGenerationSession.project_id == project_id)
            .order_by(TestGenerationSession.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def update_status(
        self, session_id: UUID, status: str, error_message: str | None = None
    ) -> None:
        s = await self.session.get(TestGenerationSession, session_id)
        if s is None:
            raise ValueError(f"TestGenerationSession {session_id} not found")
        s.status = status
        if status == "running":
            s.started_at = datetime.now(UTC)
        if status in ("completed", "failed"):
            s.completed_at = datetime.now(UTC)
        if error_message is not None:
            s.error_message = error_message
        await self.session.flush()

    async def add_generated_test(
        self,
        session_id: UUID,
        project_id: UUID,
        target_file: str,
        target_symbol: str,
        category: str,
        rationale: str,
        generated_code: str,
        confidence: float,
        validation_status: str,
        validation_error: str | None,
        execution_status: str,
        execution_output: str | None,
        quality_score: float | None,
        quality_notes: str | None,
        hypothesis_id: UUID | None,
    ) -> GeneratedTest:
        gt = GeneratedTest(
            session_id=session_id,
            project_id=project_id,
            target_file=target_file,
            target_symbol=target_symbol,
            category=category,
            rationale=rationale,
            generated_code=generated_code,
            confidence=confidence,
            validation_status=validation_status,
            validation_error=validation_error,
            execution_status=execution_status,
            execution_output=execution_output,
            quality_score=quality_score,
            quality_notes=quality_notes,
            hypothesis_id=hypothesis_id,
        )
        self.session.add(gt)
        await self.session.flush()

        # Update count on the session
        s = await self.session.get(TestGenerationSession, session_id)
        if s is not None:
            s.candidate_count = (
                await self.session.execute(
                    select(func.count())
                    .select_from(GeneratedTest)
                    .where(GeneratedTest.session_id == session_id)
                )
            ).scalar_one()

        return gt

    async def list_tests(
        self, session_id: UUID, offset: int = 0, limit: int = 50
    ) -> tuple[list[GeneratedTest], int]:
        cnt = await self.session.execute(
            select(func.count())
            .select_from(GeneratedTest)
            .where(GeneratedTest.session_id == session_id)
        )
        total: int = cnt.scalar_one()
        result = await self.session.execute(
            select(GeneratedTest)
            .where(GeneratedTest.session_id == session_id)
            .order_by(GeneratedTest.created_at)
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def list_for_project(
        self, project_id: UUID, offset: int = 0, limit: int = 50
    ) -> tuple[list[GeneratedTest], int]:
        cnt = await self.session.execute(
            select(func.count())
            .select_from(GeneratedTest)
            .where(GeneratedTest.project_id == project_id)
        )
        total: int = cnt.scalar_one()
        result = await self.session.execute(
            select(GeneratedTest)
            .where(GeneratedTest.project_id == project_id)
            .order_by(GeneratedTest.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total
