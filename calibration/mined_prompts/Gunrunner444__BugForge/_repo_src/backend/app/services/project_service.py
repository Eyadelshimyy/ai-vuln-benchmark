from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project
from app.repositories.project_repo import ProjectRepository
from app.schemas.project import ProjectCreate, ProjectListResponse, ProjectResponse

logger = logging.getLogger(__name__)


class ProjectNotFoundError(Exception):
    pass


class ProjectService:
    def __init__(self, session: AsyncSession) -> None:
        self._repo = ProjectRepository(session)
        self._session = session

    async def create_project(self, data: ProjectCreate) -> Project:
        path = Path(data.repository_path)
        if not path.is_absolute():
            raise ValueError("repository_path must be an absolute path")
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {data.repository_path}")
        if not path.is_dir():
            raise ValueError(f"Repository path is not a directory: {data.repository_path}")

        project = await self._repo.create(
            name=data.name,
            description=data.description,
            repository_path=str(path.resolve()),
        )
        await self._session.commit()
        logger.info("Created project %s (%s)", project.id, project.name)
        return project

    async def get_project(self, project_id: UUID) -> Project:
        project = await self._repo.get_by_id(project_id)
        if project is None:
            raise ProjectNotFoundError(f"Project {project_id} not found")
        return project

    async def list_projects(self, offset: int = 0, limit: int = 50) -> ProjectListResponse:
        projects, total = await self._repo.list_all(offset=offset, limit=limit)

        items: list[ProjectResponse] = []
        for p in projects:
            latest = p.analyses[0] if p.analyses else None
            items.append(
                ProjectResponse(
                    id=p.id,
                    name=p.name,
                    description=p.description,
                    repository_path=p.repository_path,
                    created_at=p.created_at,
                    updated_at=p.updated_at,
                    latest_analysis_id=latest.id if latest else None,
                    latest_analysis_status=latest.status if latest else None,
                )
            )
        return ProjectListResponse(items=items, total=total)

    async def delete_project(self, project_id: UUID) -> None:
        deleted = await self._repo.delete(project_id)
        if not deleted:
            raise ProjectNotFoundError(f"Project {project_id} not found")
        await self._session.commit()
        logger.info("Deleted project %s", project_id)
