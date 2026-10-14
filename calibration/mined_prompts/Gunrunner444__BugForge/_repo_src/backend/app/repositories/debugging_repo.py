from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.debugging import AIModelCall, DebuggingHypothesis, DebuggingSession

logger = logging.getLogger(__name__)


class DebuggingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_session(
        self,
        project_id: UUID,
        analysis_id: UUID | None,
        test_run_id: UUID | None,
    ) -> DebuggingSession:
        ds = DebuggingSession(
            project_id=project_id,
            analysis_id=analysis_id,
            test_run_id=test_run_id,
            status="pending",
        )
        self.session.add(ds)
        await self.session.flush()
        await self.session.refresh(ds)
        return ds

    async def get_by_id(self, session_id: UUID) -> DebuggingSession | None:
        result = await self.session.execute(
            select(DebuggingSession)
            .where(DebuggingSession.id == session_id)
            .options(
                selectinload(DebuggingSession.hypotheses),
                selectinload(DebuggingSession.ai_calls),
            )
        )
        return result.scalar_one_or_none()

    async def list_for_project(
        self, project_id: UUID, offset: int = 0, limit: int = 20
    ) -> tuple[list[DebuggingSession], int]:
        count_q = await self.session.execute(
            select(func.count())
            .select_from(DebuggingSession)
            .where(DebuggingSession.project_id == project_id)
        )
        total: int = count_q.scalar_one()

        result = await self.session.execute(
            select(DebuggingSession)
            .where(DebuggingSession.project_id == project_id)
            .options(selectinload(DebuggingSession.hypotheses))
            .order_by(DebuggingSession.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def update_status(
        self,
        session_id: UUID,
        status: str,
        error_message: str | None = None,
        context_summary: str | None = None,
    ) -> None:
        ds = await self.session.get(DebuggingSession, session_id)
        if ds is None:
            raise ValueError(f"DebuggingSession {session_id} not found")
        ds.status = status
        if status == "running":
            ds.started_at = datetime.now(UTC)
        if status in ("completed", "failed"):
            ds.completed_at = datetime.now(UTC)
        if error_message is not None:
            ds.error_message = error_message
        if context_summary is not None:
            ds.context_summary = context_summary
        await self.session.flush()

    async def add_hypothesis(
        self,
        session_id: UUID,
        hypothesis: object,  # HypothesisResult from ai.provider
        provider: str,
        model: str,
    ) -> DebuggingHypothesis:
        from app.ai.provider import HypothesisResult

        h: HypothesisResult = hypothesis  # type: ignore[assignment]
        dh = DebuggingHypothesis(
            session_id=session_id,
            root_cause=h.root_cause,
            confidence=h.confidence,
            confidence_label=h.confidence_label,
            affected_files=json.dumps(h.affected_files),
            affected_symbols=json.dumps(h.affected_symbols),
            evidence_summary=json.dumps(h.evidence_summary),
            contradictory_evidence=json.dumps(h.contradictory_evidence),
            reproduction_strategy=h.reproduction_strategy,
            recommended_tests=json.dumps(h.recommended_tests),
            explanation=h.explanation,
            ai_provider=provider,
            ai_model=model,
        )
        self.session.add(dh)
        await self.session.flush()
        return dh

    async def record_ai_call(
        self,
        session_id: UUID,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_seconds: float,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        self.session.add(
            AIModelCall(
                session_id=session_id,
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_seconds=duration_seconds,
                success=success,
                error_message=error_message,
            )
        )
        await self.session.flush()
