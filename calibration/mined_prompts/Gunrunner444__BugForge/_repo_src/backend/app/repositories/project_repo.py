from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.project import Project

logger = logging.getLogger(__name__)


class ProjectRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self, name: str, repository_path: str, description: str | None = None
    ) -> Project:
        project = Project(name=name, description=description, repository_path=repository_path)
        self.session.add(project)
        await self.session.flush()
        await self.session.refresh(project, ["analyses"])
        return project

    async def get_by_id(self, project_id: UUID) -> Project | None:
        result = await self.session.execute(
            select(Project).where(Project.id == project_id).options(selectinload(Project.analyses))
        )
        return result.scalar_one_or_none()

    async def list_all(self, offset: int = 0, limit: int = 50) -> tuple[list[Project], int]:
        count_result = await self.session.execute(select(func.count()).select_from(Project))
        total: int = count_result.scalar_one()

        result = await self.session.execute(
            select(Project)
            .options(selectinload(Project.analyses))
            .order_by(Project.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        projects = list(result.scalars().all())
        return projects, total

    async def delete(self, project_id: UUID) -> bool:
        project = await self.session.get(Project, project_id)
        if project is None:
            return False
        await self.session.delete(project)
        await self.session.flush()
        return True
