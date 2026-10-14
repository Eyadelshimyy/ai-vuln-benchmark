"""Data-access layer for discovery domain objects (v1.1.0)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.discovery import AutonomousAnalysisRun, DiscoveryRun, RepositoryCandidate

# ---------------------------------------------------------------------------
# DiscoveryRunRepository
# ---------------------------------------------------------------------------


class DiscoveryRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, search_criteria: dict[str, object]) -> DiscoveryRun:
        run = DiscoveryRun(
            status="running",
            search_criteria=json.dumps(search_criteria),
        )
        self.session.add(run)
        await self.session.flush()
        await self.session.refresh(run)
        return run

    async def get_by_id(self, run_id: UUID) -> DiscoveryRun | None:
        result = await self.session.execute(select(DiscoveryRun).where(DiscoveryRun.id == run_id))
        return result.scalar_one_or_none()

    async def list_recent(self, limit: int = 20, offset: int = 0) -> tuple[list[DiscoveryRun], int]:
        total_result = await self.session.execute(select(func.count()).select_from(DiscoveryRun))
        total = total_result.scalar_one()
        result = await self.session.execute(
            select(DiscoveryRun)
            .order_by(DiscoveryRun.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def complete(
        self,
        run_id: UUID,
        discovered: int,
        eligible: int,
        rejected: int,
        api_requests: int,
        error_message: str | None = None,
    ) -> None:
        now = datetime.now(UTC)
        status = "failed" if error_message else "completed"
        await self.session.execute(
            update(DiscoveryRun)
            .where(DiscoveryRun.id == run_id)
            .values(
                status=status,
                completed_at=now,
                discovered_count=discovered,
                eligible_count=eligible,
                rejected_count=rejected,
                github_api_requests=api_requests,
                error_message=error_message,
            )
        )
        await self.session.flush()

    async def cancel(self, run_id: UUID) -> None:
        await self.session.execute(
            update(DiscoveryRun)
            .where(DiscoveryRun.id == run_id)
            .values(status="cancelled", completed_at=datetime.now(UTC))
        )
        await self.session.flush()


# ---------------------------------------------------------------------------
# RepositoryCandidateRepository
# ---------------------------------------------------------------------------


class RepositoryCandidateRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_from_github(
        self,
        github_repo_id: int,
        owner: str,
        name: str,
        full_name: str,
        html_url: str,
        clone_url: str,
        stars: int,
        is_fork: bool,
        is_archived: bool,
        default_branch: str,
        primary_language: str | None,
        license_key: str | None,
        size_kb: int,
        open_issues: int,
        topics: list[str],
        description: str | None,
        last_updated_at: datetime | None,
        last_pushed_at: datetime | None,
    ) -> tuple[RepositoryCandidate, bool]:
        """Create or refresh a candidate; return (candidate, created)."""
        existing = await self.get_by_github_id(github_repo_id)
        if existing is not None:
            # Refresh mutable metadata but do NOT reset eligibility_status
            existing.stars = stars
            existing.size_kb = size_kb
            existing.open_issues = open_issues
            existing.topics = json.dumps(topics)
            existing.description = description
            existing.last_updated_at = last_updated_at
            existing.last_pushed_at = last_pushed_at
            existing.updated_at = datetime.now(UTC)
            await self.session.flush()
            await self.session.refresh(existing)
            return existing, False

        candidate = RepositoryCandidate(
            github_repo_id=github_repo_id,
            owner=owner,
            name=name,
            full_name=full_name,
            html_url=html_url,
            clone_url=clone_url,
            stars=stars,
            is_fork=is_fork,
            is_archived=is_archived,
            default_branch=default_branch,
            primary_language=primary_language,
            license_key=license_key,
            size_kb=size_kb,
            open_issues=open_issues,
            topics=json.dumps(topics),
            description=description,
            last_updated_at=last_updated_at,
            last_pushed_at=last_pushed_at,
            eligibility_status="discovered",
        )
        self.session.add(candidate)
        await self.session.flush()
        await self.session.refresh(candidate)
        return candidate, True

    async def get_by_id(self, candidate_id: UUID) -> RepositoryCandidate | None:
        result = await self.session.execute(
            select(RepositoryCandidate).where(RepositoryCandidate.id == candidate_id)
        )
        return result.scalar_one_or_none()

    async def get_by_github_id(self, github_repo_id: int) -> RepositoryCandidate | None:
        result = await self.session.execute(
            select(RepositoryCandidate).where(RepositoryCandidate.github_repo_id == github_repo_id)
        )
        return result.scalar_one_or_none()

    async def get_by_full_name(self, full_name: str) -> RepositoryCandidate | None:
        result = await self.session.execute(
            select(RepositoryCandidate).where(RepositoryCandidate.full_name == full_name)
        )
        return result.scalar_one_or_none()

    async def list_all(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RepositoryCandidate], int]:
        q = select(RepositoryCandidate)
        count_q = select(func.count()).select_from(RepositoryCandidate)
        if status:
            q = q.where(RepositoryCandidate.eligibility_status == status)
            count_q = count_q.where(RepositoryCandidate.eligibility_status == status)
        total = (await self.session.execute(count_q)).scalar_one()
        items = (
            (
                await self.session.execute(
                    q.order_by(RepositoryCandidate.stars.desc()).offset(offset).limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return list(items), total

    async def update_eligibility(
        self,
        candidate_id: UUID,
        eligibility_status: str,
        eligibility_score: float | None = None,
        rejection_reason: str | None = None,
        safety_classification: str | None = None,
        safety_detail: str | None = None,
    ) -> None:
        values: dict[str, object] = {
            "eligibility_status": eligibility_status,
            "updated_at": datetime.now(UTC),
        }
        if eligibility_score is not None:
            values["eligibility_score"] = eligibility_score
        if rejection_reason is not None:
            values["rejection_reason"] = rejection_reason
        if safety_classification is not None:
            values["safety_classification"] = safety_classification
        if safety_detail is not None:
            values["safety_detail"] = safety_detail

        await self.session.execute(
            update(RepositoryCandidate)
            .where(RepositoryCandidate.id == candidate_id)
            .values(**values)
        )
        await self.session.flush()

    async def mark_analyzed(
        self,
        candidate_id: UUID,
        commit_sha: str,
        analysis_status: str,
    ) -> None:
        await self.session.execute(
            update(RepositoryCandidate)
            .where(RepositoryCandidate.id == candidate_id)
            .values(
                last_analyzed_commit=commit_sha,
                last_analyzed_at=datetime.now(UTC),
                analysis_status=analysis_status,
                updated_at=datetime.now(UTC),
            )
        )
        await self.session.flush()

    async def count_by_status(self) -> dict[str, int]:
        result = await self.session.execute(
            select(RepositoryCandidate.eligibility_status, func.count()).group_by(
                RepositoryCandidate.eligibility_status
            )
        )
        return {row[0]: row[1] for row in result.all()}


# ---------------------------------------------------------------------------
# AutonomousAnalysisRunRepository
# ---------------------------------------------------------------------------


class AutonomousAnalysisRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        candidate_id: UUID,
        ai_provider: str,
        ai_model: str,
        is_local_ai: bool,
    ) -> AutonomousAnalysisRun:
        run = AutonomousAnalysisRun(
            candidate_id=candidate_id,
            status="queued",
            ai_provider=ai_provider,
            ai_model=ai_model,
            is_local_ai=is_local_ai,
        )
        self.session.add(run)
        await self.session.flush()
        await self.session.refresh(run)
        return run

    async def get_by_id(self, run_id: UUID) -> AutonomousAnalysisRun | None:
        result = await self.session.execute(
            select(AutonomousAnalysisRun).where(AutonomousAnalysisRun.id == run_id)
        )
        return result.scalar_one_or_none()

    async def list_for_candidate(self, candidate_id: UUID) -> list[AutonomousAnalysisRun]:
        result = await self.session.execute(
            select(AutonomousAnalysisRun)
            .where(AutonomousAnalysisRun.candidate_id == candidate_id)
            .order_by(AutonomousAnalysisRun.created_at.desc())
        )
        return list(result.scalars().all())

    async def list_all(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AutonomousAnalysisRun], int]:
        q = select(AutonomousAnalysisRun)
        count_q = select(func.count()).select_from(AutonomousAnalysisRun)
        if status:
            q = q.where(AutonomousAnalysisRun.status == status)
            count_q = count_q.where(AutonomousAnalysisRun.status == status)
        total = (await self.session.execute(count_q)).scalar_one()
        items = (
            (
                await self.session.execute(
                    q.order_by(AutonomousAnalysisRun.created_at.desc()).offset(offset).limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return list(items), total

    async def update_status(
        self,
        run_id: UUID,
        status: str,
        current_stage: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        **counters: int,
    ) -> None:
        values: dict[str, object] = {
            "status": status,
            "updated_at": datetime.now(UTC),
        }
        if current_stage is not None:
            values["current_stage"] = current_stage
        if error_code is not None:
            values["error_code"] = error_code
        if error_message is not None:
            values["error_message"] = error_message
        if status in {"completed", "failed", "cancelled", "inconclusive"}:
            values["completed_at"] = datetime.now(UTC)
        if status not in {"queued"}:
            values.setdefault("started_at", datetime.now(UTC))
        for k, v in counters.items():
            values[k] = v
        await self.session.execute(
            update(AutonomousAnalysisRun).where(AutonomousAnalysisRun.id == run_id).values(**values)
        )
        await self.session.flush()

    async def set_project_id(self, run_id: UUID, project_id: UUID) -> None:
        await self.session.execute(
            update(AutonomousAnalysisRun)
            .where(AutonomousAnalysisRun.id == run_id)
            .values(project_id=project_id, updated_at=datetime.now(UTC))
        )
        await self.session.flush()

    async def cancel(self, run_id: UUID, reason: str = "user_requested") -> None:
        await self.session.execute(
            update(AutonomousAnalysisRun)
            .where(AutonomousAnalysisRun.id == run_id)
            .values(
                status="cancelled",
                error_code=reason,
                completed_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        await self.session.flush()

    async def has_active_run_for_candidate(self, candidate_id: UUID) -> bool:
        """Return True if there is already a non-terminal run for this candidate."""
        active_statuses = (
            "queued",
            "screening",
            "acquiring",
            "static_analyzing",
            "ai_analyzing",
            "finding_validation",
        )
        result = await self.session.execute(
            select(func.count())
            .select_from(AutonomousAnalysisRun)
            .where(
                AutonomousAnalysisRun.candidate_id == candidate_id,
                AutonomousAnalysisRun.status.in_(active_statuses),
            )
        )
        return result.scalar_one() > 0
