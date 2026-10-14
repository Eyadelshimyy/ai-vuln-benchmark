from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.repair import PatchCandidate, RepairSession


class RepairRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_session(
        self,
        project_id: UUID,
        hypothesis_id: UUID | None,
        reproduction_session_id: UUID | None,
        debugging_session_id: UUID | None,
    ) -> RepairSession:
        s = RepairSession(
            project_id=project_id,
            hypothesis_id=hypothesis_id,
            reproduction_session_id=reproduction_session_id,
            debugging_session_id=debugging_session_id,
            status="pending",
        )
        self.session.add(s)
        await self.session.flush()
        await self.session.refresh(s)
        return s

    async def get_session_by_id(self, session_id: UUID) -> RepairSession | None:
        result = await self.session.execute(
            select(RepairSession)
            .where(RepairSession.id == session_id)
            .options(
                selectinload(RepairSession.candidates),
            )
        )
        return result.scalar_one_or_none()

    async def list_for_project(
        self, project_id: UUID, offset: int = 0, limit: int = 20
    ) -> tuple[list[RepairSession], int]:
        cnt = await self.session.execute(
            select(func.count())
            .select_from(RepairSession)
            .where(RepairSession.project_id == project_id)
        )
        total: int = cnt.scalar_one()
        result = await self.session.execute(
            select(RepairSession)
            .where(RepairSession.project_id == project_id)
            .order_by(RepairSession.created_at.desc())
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
        s = await self.session.get(RepairSession, session_id)
        if s is None:
            raise ValueError(f"RepairSession {session_id} not found")
        s.status = status
        if status == "running":
            s.started_at = datetime.now(UTC)
        if error_message is not None:
            s.error_message = error_message
        await self.session.flush()

    async def complete_session(
        self,
        session_id: UUID,
        total_candidates: int,
        best_candidate_id: UUID | None,
    ) -> None:
        s = await self.session.get(RepairSession, session_id)
        if s is None:
            raise ValueError(f"RepairSession {session_id} not found")
        s.status = "completed"
        s.completed_at = datetime.now(UTC)
        s.total_candidates = total_candidates
        s.best_candidate_id = best_candidate_id
        await self.session.flush()

    async def create_candidate(
        self,
        session_id: UUID,
        rank: int,
        patch_plan: str,
        patch_diff: str,
        changed_files: list[str],
        provider: str,
        model: str,
    ) -> PatchCandidate:
        c = PatchCandidate(
            session_id=session_id,
            rank=rank,
            patch_plan=patch_plan,
            patch_diff=patch_diff,
            changed_files_json=json.dumps(changed_files),
            patch_provider=provider,
            patch_model=model,
            status="pending",
            disposition="pending",
        )
        self.session.add(c)
        await self.session.flush()
        await self.session.refresh(c)
        return c

    async def update_candidate(
        self,
        candidate_id: UUID,
        **fields: object,
    ) -> None:
        c = await self.session.get(PatchCandidate, candidate_id)
        if c is None:
            raise ValueError(f"PatchCandidate {candidate_id} not found")
        for key, value in fields.items():
            setattr(c, key, value)
        await self.session.flush()

    async def get_candidate(self, candidate_id: UUID) -> PatchCandidate | None:
        return await self.session.get(PatchCandidate, candidate_id)
