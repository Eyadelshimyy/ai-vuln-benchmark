"""Repository layer for GitHub domain objects — Phase 9."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.github import GitHubDelivery, GitHubRepository


class GitHubRepositoryRepo:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        project_id: UUID,
        owner: str,
        repo: str,
        default_branch: str,
        html_url: str,
        github_id: int | None = None,
    ) -> GitHubRepository:
        gr = GitHubRepository(
            project_id=project_id,
            owner=owner,
            repo=repo,
            default_branch=default_branch,
            html_url=html_url,
            github_id=github_id,
            connected=False,
        )
        self.session.add(gr)
        await self.session.flush()
        await self.session.refresh(gr)
        return gr

    async def get_by_project(self, project_id: UUID) -> GitHubRepository | None:
        result = await self.session.execute(
            select(GitHubRepository).where(GitHubRepository.project_id == project_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, github_repo_id: UUID) -> GitHubRepository | None:
        result = await self.session.execute(
            select(GitHubRepository).where(GitHubRepository.id == github_repo_id)
        )
        return result.scalar_one_or_none()

    async def update(self, github_repo_id: UUID, **kwargs: object) -> None:
        gr = await self.get_by_id(github_repo_id)
        if gr is None:
            return
        for key, value in kwargs.items():
            setattr(gr, key, value)
        await self.session.flush()

    async def delete(self, github_repo_id: UUID) -> None:
        gr = await self.get_by_id(github_repo_id)
        if gr is not None:
            await self.session.delete(gr)
            await self.session.flush()


class GitHubDeliveryRepo:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        project_id: UUID,
        github_repository_id: UUID,
        candidate_id: UUID,
        verification_id: UUID,
        owner: str,
        repo: str,
        base_branch: str,
        delivery_branch: str,
        verified_patch_hash: str,
    ) -> GitHubDelivery:
        d = GitHubDelivery(
            project_id=project_id,
            github_repository_id=github_repository_id,
            candidate_id=candidate_id,
            verification_id=verification_id,
            owner=owner,
            repo=repo,
            base_branch=base_branch,
            delivery_branch=delivery_branch,
            verified_patch_hash=verified_patch_hash,
            status="pending",
        )
        self.session.add(d)
        await self.session.flush()
        await self.session.refresh(d)
        return d

    async def get_by_id(self, delivery_id: UUID) -> GitHubDelivery | None:
        result = await self.session.execute(
            select(GitHubDelivery).where(GitHubDelivery.id == delivery_id)
        )
        return result.scalar_one_or_none()

    async def get_by_candidate(self, candidate_id: UUID) -> GitHubDelivery | None:
        result = await self.session.execute(
            select(GitHubDelivery)
            .where(GitHubDelivery.candidate_id == candidate_id)
            .order_by(GitHubDelivery.created_at.desc())
        )
        return result.scalars().first()

    async def list_for_project(self, project_id: UUID) -> list[GitHubDelivery]:
        result = await self.session.execute(
            select(GitHubDelivery)
            .where(GitHubDelivery.project_id == project_id)
            .order_by(GitHubDelivery.created_at.desc())
        )
        return list(result.scalars().all())

    async def update(self, delivery_id: UUID, **kwargs: object) -> None:
        d = await self.get_by_id(delivery_id)
        if d is None:
            return
        for key, value in kwargs.items():
            setattr(d, key, value)
        await self.session.flush()
