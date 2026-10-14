from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import Analysis, CodeEntity, ImportRecord, RepositoryFile

logger = logging.getLogger(__name__)


class AnalysisRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, project_id: UUID, repository_path: str) -> Analysis:
        analysis = Analysis(
            project_id=project_id, repository_path=repository_path, status="pending"
        )
        self.session.add(analysis)
        await self.session.flush()
        await self.session.refresh(analysis)
        return analysis

    async def get_by_id(self, analysis_id: UUID) -> Analysis | None:
        return await self.session.get(Analysis, analysis_id)

    async def list_for_project(
        self, project_id: UUID, offset: int = 0, limit: int = 20
    ) -> tuple[list[Analysis], int]:
        count_result = await self.session.execute(
            select(func.count()).select_from(Analysis).where(Analysis.project_id == project_id)
        )
        total: int = count_result.scalar_one()

        result = await self.session.execute(
            select(Analysis)
            .where(Analysis.project_id == project_id)
            .order_by(Analysis.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def update_status(
        self,
        analysis_id: UUID,
        status: str,
        error_message: str | None = None,
    ) -> None:
        analysis = await self.session.get(Analysis, analysis_id)
        if analysis is None:
            raise ValueError(f"Analysis {analysis_id} not found")
        analysis.status = status
        if status == "running":
            analysis.started_at = datetime.now(UTC)
        if error_message is not None:
            analysis.error_message = error_message
        await self.session.flush()

    async def complete_analysis(self, analysis_id: UUID, summary: dict[str, Any]) -> None:
        analysis = await self.session.get(Analysis, analysis_id)
        if analysis is None:
            raise ValueError(f"Analysis {analysis_id} not found")
        analysis.status = "completed"
        analysis.completed_at = datetime.now(UTC)
        analysis.summary = summary
        await self.session.flush()

    async def list_files(
        self, analysis_id: UUID, offset: int = 0, limit: int = 100
    ) -> tuple[list[RepositoryFile], int]:
        count_result = await self.session.execute(
            select(func.count())
            .select_from(RepositoryFile)
            .where(RepositoryFile.analysis_id == analysis_id)
        )
        total: int = count_result.scalar_one()

        result = await self.session.execute(
            select(RepositoryFile)
            .where(RepositoryFile.analysis_id == analysis_id)
            .order_by(RepositoryFile.relative_path)
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def list_entities(
        self,
        analysis_id: UUID,
        file_id: UUID | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[CodeEntity], int]:
        base_filter: Any = CodeEntity.file_id.in_(
            select(RepositoryFile.id).where(RepositoryFile.analysis_id == analysis_id)
        )
        if file_id is not None:
            base_filter = CodeEntity.file_id == file_id

        count_result = await self.session.execute(
            select(func.count()).select_from(CodeEntity).where(base_filter)
        )
        total: int = count_result.scalar_one()

        result = await self.session.execute(
            select(CodeEntity)
            .where(base_filter)
            .order_by(CodeEntity.qualified_name)
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def list_imports(
        self,
        analysis_id: UUID,
        file_id: UUID | None = None,
        offset: int = 0,
        limit: int = 200,
    ) -> tuple[list[ImportRecord], int]:
        base_filter: Any = ImportRecord.file_id.in_(
            select(RepositoryFile.id).where(RepositoryFile.analysis_id == analysis_id)
        )
        if file_id is not None:
            base_filter = ImportRecord.file_id == file_id

        count_result = await self.session.execute(
            select(func.count()).select_from(ImportRecord).where(base_filter)
        )
        total: int = count_result.scalar_one()

        result = await self.session.execute(
            select(ImportRecord)
            .where(base_filter)
            .order_by(ImportRecord.module_name)
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total
