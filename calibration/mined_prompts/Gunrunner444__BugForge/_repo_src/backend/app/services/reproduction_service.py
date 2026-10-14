"""Bug reproduction service — Phase 6 orchestration."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from app.testing.reproduction_planner import ReproductionPlan, ReproductionPlanner
from app.testing.test_validator import validate_test_code

logger = logging.getLogger(__name__)

_OUTPUT_LIMIT = 4_096
_DEFAULT_ATTEMPTS = 3

# Container path where the analyzed repository is mounted read-only
_REPO_CONTAINER_PATH = "/bugforge-repo"


def _classify_attempt(
    exec_result: Any,
    expected_failure_pattern: str | None,
) -> tuple[str, bool, str | None]:
    """Return (classification, reproduced, evidence_matched) for one attempt.

    Classification semantics:
      reproduced        — test ran, failed, AND matched the expected failure pattern
      no_evidence       — test failed (exit_code 1) but did NOT match the pattern
      failed            — test ran and passed (exit_code 0); bug not triggered
      timeout           — execution timed out
      environment_error — executor reported an infrastructure error
      error             — unexpected executor exit code (2, 3, 4, 5, ...)
    """
    if exec_result.timed_out:
        return "timeout", False, None

    if exec_result.exit_code == 0:
        return "failed", False, None

    if exec_result.error_message and exec_result.exit_code not in (1,):
        return "environment_error", False, None

    if exec_result.exit_code == 1:
        combined = (exec_result.stdout or "") + (exec_result.stderr or "")
        if expected_failure_pattern:
            evidence = _find_evidence(combined, expected_failure_pattern)
            if evidence:
                return "reproduced", True, evidence
            return "no_evidence", False, None
        # No pattern — cannot confirm specificity
        return "no_evidence", False, None

    # Exit codes 2+: collection error, usage error, etc.
    return "error", False, None


def _find_evidence(output: str, pattern: str) -> str | None:
    """Return a matching snippet if the pattern appears in output, else None."""
    lines = output.splitlines()
    pattern_lower = pattern.lower()
    for line in lines:
        if pattern_lower in line.lower():
            return line[:500]

    try:
        m = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
        if m:
            start = max(0, m.start() - 50)
            return output[start : m.end() + 200][:500]
    except re.error:
        pass

    return None


class BugReproductionService:
    async def run_session(
        self,
        session_id: UUID,
        project_id: UUID,
        repository_path: str,
        hypothesis_id: UUID | None,
        generated_test_id: UUID | None,
        total_attempts: int = _DEFAULT_ATTEMPTS,
    ) -> None:
        from app.database import async_session_factory
        from app.repositories.reproduction_repo import ReproductionRepository

        logger.info("Starting reproduction session %s", session_id)

        async with async_session_factory() as db:
            repo = ReproductionRepository(db)
            await repo.update_status(session_id, "running")
            await db.commit()

        try:
            plan = await self._get_plan(
                repository_path, hypothesis_id, generated_test_id, project_id
            )
            if plan is None:
                await self._fail(session_id, "No reproducer code available")
                return

            validation = validate_test_code(plan.reproducer_code)
            if not validation.valid:
                await self._fail(session_id, f"Reproducer rejected: {validation.error}")
                return

            await self._store_plan_metadata(session_id, plan)

            results = await self._run_attempts(
                plan.reproducer_code,
                repository_path,
                total_attempts,
                expected_failure_pattern=plan.expected_failure or None,
            )
            await self._persist_results(session_id, plan.reproducer_code, results)

        except Exception as exc:
            logger.exception("Reproduction session %s failed: %s", session_id, exc)
            await self._fail(session_id, str(exc))

    async def _get_plan(
        self,
        repository_path: str,
        hypothesis_id: UUID | None,
        generated_test_id: UUID | None,
        project_id: UUID,
    ) -> ReproductionPlan | None:
        from app.database import async_session_factory

        if generated_test_id is not None:
            async with async_session_factory() as db:
                from app.models.test_generation import GeneratedTest

                row = await db.get(GeneratedTest, generated_test_id)
                if row and row.generated_code:
                    return ReproductionPlan(
                        target_behavior="Reproduce bug via generated test",
                        preconditions=[],
                        input_description="",
                        expected_failure="",
                        observable_evidence="",
                        reproducer_code=row.generated_code,
                        cleanup_required=False,
                        provider="generated_test",
                        model="",
                    )

        if hypothesis_id is not None:
            async with async_session_factory() as db:
                from app.models.debugging import DebuggingHypothesis

                hyp = await db.get(DebuggingHypothesis, hypothesis_id)
                if hyp:
                    failing_tests = await self._load_failing_tests(db, hyp.session_id)
                    findings = await self._load_findings_for_project(db, project_id)
                    planner = ReproductionPlanner()
                    return await planner.plan(
                        hypothesis_text=hyp.root_cause,
                        repository_path=repository_path,
                        failing_tests=failing_tests,
                        static_findings=findings,
                    )

        return None

    async def _store_plan_metadata(self, session_id: UUID, plan: ReproductionPlan) -> None:
        from app.database import async_session_factory

        async with async_session_factory() as db:
            from app.models.reproduction import BugReproductionSession

            s = await db.get(BugReproductionSession, session_id)
            if s is not None:
                s.target_behavior = (plan.target_behavior or "")[:2000] or None
                s.expected_failure_pattern = (plan.expected_failure or "")[:500] or None
                s.observable_evidence = (plan.observable_evidence or "")[:1000] or None
                s.strategy_summary = (
                    "; ".join(plan.preconditions[:5]) if plan.preconditions else None
                )
            await db.commit()

    async def _load_failing_tests(
        self, db: Any, debugging_session_id: UUID
    ) -> list[dict[str, Any]]:
        from sqlalchemy import select

        from app.models.debugging import DebuggingSession
        from app.models.test_run import TestResult

        session = await db.get(DebuggingSession, debugging_session_id)
        if session is None or session.test_run_id is None:
            return []
        result = await db.execute(
            select(TestResult)
            .where(TestResult.test_run_id == session.test_run_id)
            .where(TestResult.status.in_(["failed", "error"]))
            .limit(3)
        )
        rows = result.scalars().all()
        return [{"node_id": r.node_id, "traceback": r.traceback or ""} for r in rows]

    async def _load_findings_for_project(self, db: Any, project_id: UUID) -> list[dict[str, Any]]:
        from sqlalchemy import select

        from app.models.analysis import Analysis
        from app.models.finding import DBFinding

        latest_analysis_id = await db.execute(
            select(Analysis.id)
            .where(Analysis.project_id == project_id)
            .where(Analysis.status == "completed")
            .order_by(Analysis.created_at.desc())
            .limit(1)
        )
        analysis_id = latest_analysis_id.scalar_one_or_none()
        if not analysis_id:
            return []
        result = await db.execute(
            select(DBFinding)
            .where(DBFinding.analysis_id == analysis_id)
            .order_by(DBFinding.severity.desc())
            .limit(5)
        )
        rows = result.scalars().all()
        return [
            {
                "category": r.category,
                "severity": r.severity,
                "file": r.file_path,
                "line": r.line,
                "message": r.message,
            }
            for r in rows
        ]

    async def _run_attempts(
        self,
        code: str,
        repository_path: str,
        total: int,
        expected_failure_pattern: str | None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for n in range(1, total + 1):
            result = await self._single_attempt(code, repository_path, n, expected_failure_pattern)
            results.append(result)
        return results

    async def _single_attempt(
        self,
        code: str,
        repository_path: str,
        attempt_number: int,
        expected_failure_pattern: str | None,
    ) -> dict[str, Any]:
        from app.execution import ExecutorFactory
        from app.execution.base import ExecutionConfig

        try:
            with tempfile.TemporaryDirectory(prefix="bugforge_repro_") as tmpdir:
                test_file = Path(tmpdir) / "test_reproducer.py"
                test_file.write_text(code)
                executor = ExecutorFactory.create()

                config = ExecutionConfig(
                    command=[
                        "python3",
                        "-m",
                        "pytest",
                        "test_reproducer.py",
                        "-v",
                        "--tb=short",
                        "-q",
                    ],
                    working_directory=tmpdir,
                    timeout_seconds=60,
                    output_dir=tmpdir,
                    environment={"PYTHONPATH": _REPO_CONTAINER_PATH},
                    read_only_volumes={repository_path: _REPO_CONTAINER_PATH},
                )
                exec_result = await executor.execute(config)

                cls, reproduced, evidence_matched = _classify_attempt(
                    exec_result, expected_failure_pattern
                )

                return {
                    "attempt_number": attempt_number,
                    "exit_code": exec_result.exit_code,
                    "stdout": exec_result.stdout[:_OUTPUT_LIMIT],
                    "stderr": exec_result.stderr[:_OUTPUT_LIMIT],
                    "duration_seconds": exec_result.duration_seconds,
                    "timed_out": exec_result.timed_out,
                    "reproduced": reproduced,
                    "classification": cls,
                    "evidence_matched": evidence_matched,
                    "reproducer_code": code,
                    "command": " ".join(config.command),
                }
        except Exception as exc:
            return {
                "attempt_number": attempt_number,
                "exit_code": None,
                "stdout": "",
                "stderr": str(exc),
                "duration_seconds": 0.0,
                "timed_out": False,
                "reproduced": False,
                "classification": "error",
                "evidence_matched": None,
                "reproducer_code": code,
                "command": "",
            }

    async def _persist_results(
        self, session_id: UUID, reproducer_code: str, results: list[dict[str, Any]]
    ) -> None:
        from app.database import async_session_factory
        from app.repositories.reproduction_repo import ReproductionRepository

        successful = sum(1 for r in results if r["reproduced"])
        total = len(results)
        rate = successful / total if total > 0 else 0.0

        if successful == 0:
            cls = "not_reproduced"
        elif successful == total:
            cls = "consistently_reproduced"
        elif successful >= total * 0.5:
            cls = "intermittent"
        else:
            cls = "inconclusive"

        async with async_session_factory() as db:
            repo = ReproductionRepository(db)
            for r in results:
                await repo.add_attempt(session_id, r)
            await repo.complete(session_id, successful, total, rate, cls)
            await db.commit()

        logger.info(
            "Reproduction session %s: %d/%d reproduced → %s",
            session_id,
            successful,
            total,
            cls,
        )

    async def _fail(self, session_id: UUID, message: str) -> None:
        from app.database import async_session_factory
        from app.repositories.reproduction_repo import ReproductionRepository

        async with async_session_factory() as db:
            repo = ReproductionRepository(db)
            await repo.update_status(session_id, "failed", error_message=message)
            await db.commit()
