from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.test_run import TestResult, TestRun
from app.testing.pytest_parser import ParsedTestRun

logger = logging.getLogger(__name__)


class TestRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, project_id: UUID, repository_path: str) -> TestRun:
        run = TestRun(project_id=project_id, repository_path=repository_path, status="pending")
        self.session.add(run)
        await self.session.flush()
        await self.session.refresh(run)
        return run

    async def get_by_id(self, run_id: UUID) -> TestRun | None:
        return await self.session.get(TestRun, run_id)

    async def list_for_project(
        self, project_id: UUID, offset: int = 0, limit: int = 20
    ) -> tuple[list[TestRun], int]:
        count_result = await self.session.execute(
            select(func.count()).select_from(TestRun).where(TestRun.project_id == project_id)
        )
        total: int = count_result.scalar_one()

        result = await self.session.execute(
            select(TestRun)
            .where(TestRun.project_id == project_id)
            .order_by(TestRun.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def list_results(
        self, run_id: UUID, offset: int = 0, limit: int = 500
    ) -> tuple[list[TestResult], int]:
        count_result = await self.session.execute(
            select(func.count()).select_from(TestResult).where(TestResult.test_run_id == run_id)
        )
        total: int = count_result.scalar_one()

        result = await self.session.execute(
            select(TestResult)
            .where(TestResult.test_run_id == run_id)
            .order_by(TestResult.node_id)
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def update_status(
        self, run_id: UUID, status: str, error_message: str | None = None
    ) -> None:
        run = await self.session.get(TestRun, run_id)
        if run is None:
            raise ValueError(f"TestRun {run_id} not found")
        run.status = status
        if status == "running":
            run.started_at = datetime.now(UTC)
        if error_message is not None:
            run.error_message = error_message
        await self.session.flush()

    async def complete(
        self,
        run_id: UUID,
        parsed: ParsedTestRun,
        repository_path: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
        duration_seconds: float = 0.0,
        command: str | None = None,
    ) -> None:
        run = await self.session.get(TestRun, run_id)
        if run is None:
            raise ValueError(f"TestRun {run_id} not found")
        # Determine semantic status: passed (all green) vs failed (some failed/errored)
        run.status = "passed" if (parsed.failed == 0 and parsed.errors == 0) else "failed"
        run.completed_at = datetime.now(UTC)
        run.total_tests = parsed.total
        run.passed = parsed.passed
        run.failed = parsed.failed
        run.skipped = parsed.skipped
        run.errors = parsed.errors
        run.stdout = stdout
        run.stderr = stderr
        run.exit_code = exit_code
        run.duration_seconds = duration_seconds
        run.command = command

        for pr in parsed.results:
            self.session.add(
                TestResult(
                    test_run_id=run_id,
                    node_id=pr.node_id,
                    test_file=pr.test_file,
                    test_name=pr.test_name,
                    status=pr.status,
                    duration_seconds=pr.duration_seconds,
                    traceback=pr.traceback,
                    stdout=pr.stdout,
                    stderr=pr.stderr,
                    skip_reason=pr.skip_reason,
                )
            )
        await self.session.flush()

    async def fail(self, run_id: UUID, error_message: str) -> None:
        run = await self.session.get(TestRun, run_id)
        if run is None:
            raise ValueError(f"TestRun {run_id} not found")
        # "error" = BugForge failed to execute the run, as opposed to "failed" = tests ran but failed
        run.status = "error"
        run.completed_at = datetime.now(UTC)
        run.error_message = error_message
        await self.session.flush()
