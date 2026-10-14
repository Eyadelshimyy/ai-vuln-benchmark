from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.reproduction import BugReproductionAttempt, BugReproductionSession


class ReproductionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        project_id: UUID,
        hypothesis_id: UUID | None,
        generated_test_id: UUID | None,
        debugging_session_id: UUID | None,
        total_attempts: int = 3,
    ) -> BugReproductionSession:
        s = BugReproductionSession(
            project_id=project_id,
            hypothesis_id=hypothesis_id,
            generated_test_id=generated_test_id,
            debugging_session_id=debugging_session_id,
            total_attempts=total_attempts,
            status="pending",
        )
        self.session.add(s)
        await self.session.flush()
        await self.session.refresh(s)
        return s

    async def get_by_id(self, session_id: UUID) -> BugReproductionSession | None:
        result = await self.session.execute(
            select(BugReproductionSession)
            .where(BugReproductionSession.id == session_id)
            .options(selectinload(BugReproductionSession.attempts))
        )
        return result.scalar_one_or_none()

    async def list_for_project(
        self, project_id: UUID, offset: int = 0, limit: int = 20
    ) -> tuple[list[BugReproductionSession], int]:
        cnt = await self.session.execute(
            select(func.count())
            .select_from(BugReproductionSession)
            .where(BugReproductionSession.project_id == project_id)
        )
        total: int = cnt.scalar_one()
        result = await self.session.execute(
            select(BugReproductionSession)
            .where(BugReproductionSession.project_id == project_id)
            .order_by(BugReproductionSession.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def update_status(
        self,
        session_id: UUID,
        status: str,
        error_message: str | None = None,
    ) -> None:
        s = await self.session.get(BugReproductionSession, session_id)
        if s is None:
            raise ValueError(f"BugReproductionSession {session_id} not found")
        s.status = status
        if status == "running":
            s.started_at = datetime.now(UTC)
        if error_message is not None:
            s.error_message = error_message
        await self.session.flush()

    async def add_attempt(self, session_id: UUID, data: dict[str, Any]) -> BugReproductionAttempt:
        attempt = BugReproductionAttempt(
            session_id=session_id,
            attempt_number=data["attempt_number"],
            command=data.get("command"),
            input_description=data.get("input_description"),
            reproducer_code=data.get("reproducer_code"),
            exit_code=data.get("exit_code"),
            stdout=data.get("stdout"),
            stderr=data.get("stderr"),
            traceback=data.get("traceback"),
            duration_seconds=data.get("duration_seconds"),
            timed_out=data.get("timed_out", False),
            reproduced=data.get("reproduced", False),
            classification=data.get("classification", "failed"),
            evidence_matched=data.get("evidence_matched"),
        )
        self.session.add(attempt)

        # Increment session counter
        s = await self.session.get(BugReproductionSession, session_id)
        if s is not None:
            s.attempt_count = (s.attempt_count or 0) + 1

        await self.session.flush()
        return attempt

    async def complete(
        self,
        session_id: UUID,
        successful: int,
        total: int,
        rate: float,
        classification: str,
    ) -> None:
        s = await self.session.get(BugReproductionSession, session_id)
        if s is None:
            raise ValueError(f"BugReproductionSession {session_id} not found")
        s.status = "completed"
        s.completed_at = datetime.now(UTC)
        s.successful_attempts = successful
        s.attempt_count = total
        s.reproducibility_rate = round(rate, 3)
        s.final_classification = classification
        await self.session.flush()
