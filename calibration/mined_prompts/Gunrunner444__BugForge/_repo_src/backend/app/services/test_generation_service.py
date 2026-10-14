"""Test generation service — Phase 5 orchestration."""

from __future__ import annotations

import ast
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from app.testing.test_generator import TestCandidate, TestGenerationRequest, TestGenerator
from app.testing.test_validator import compute_quality_score, validate_test_code

logger = logging.getLogger(__name__)

_MAX_CANDIDATES = 5
_EXECUTION_TIMEOUT = 60


class TestGenerationService:
    async def run_session(
        self,
        session_id: UUID,
        project_id: UUID,
        repository_path: str,
        project_name: str,
        analysis_id: UUID | None,
        test_run_id: UUID | None,
        debugging_session_id: UUID | None,
    ) -> None:
        from app.ai import get_provider
        from app.database import async_session_factory
        from app.repositories.test_generation_repo import TestGenerationRepository

        logger.info("Starting test generation session %s", session_id)

        async with async_session_factory() as db:
            repo = TestGenerationRepository(db)
            await repo.update_status(session_id, "running")
            await db.commit()

        try:
            req = await self._build_request(
                project_name,
                repository_path,
                test_run_id,
                analysis_id,
                debugging_session_id,
                project_id,
            )
            provider = get_provider()
            generator = TestGenerator(provider)
            candidates = await generator.generate(req)

            if not candidates:
                logger.warning("No test candidates generated for session %s", session_id)

            for candidate in candidates:
                await self._process_candidate(
                    session_id=session_id,
                    project_id=project_id,
                    candidate=candidate,
                    repository_path=repository_path,
                )

            async with async_session_factory() as db:
                repo = TestGenerationRepository(db)
                await repo.update_status(session_id, "completed")
                await db.commit()

            logger.info(
                "Test generation session %s completed — %d candidates", session_id, len(candidates)
            )

        except Exception as exc:
            logger.exception("Test generation session %s failed: %s", session_id, exc)
            async with async_session_factory() as db:
                repo = TestGenerationRepository(db)
                await repo.update_status(session_id, "failed", error_message=str(exc))
                await db.commit()

    async def _build_request(
        self,
        project_name: str,
        repository_path: str,
        test_run_id: UUID | None,
        analysis_id: UUID | None,
        debugging_session_id: UUID | None,
        project_id: UUID,
    ) -> TestGenerationRequest:
        from app.database import async_session_factory

        async with async_session_factory() as db:
            failing_tests = await self._load_failing_tests(db, test_run_id)
            static_findings = await self._load_findings(db, analysis_id)
            hypotheses = await self._load_hypotheses(db, debugging_session_id)

        source_summaries = self._build_source_summaries(repository_path)

        return TestGenerationRequest(
            project_name=project_name,
            repository_path=repository_path,
            source_summaries=source_summaries,
            failing_tests=failing_tests,
            static_findings=static_findings,
            hypotheses=hypotheses,
            max_candidates=_MAX_CANDIDATES,
        )

    async def _load_failing_tests(self, db: Any, test_run_id: UUID | None) -> list[dict[str, Any]]:
        if test_run_id is None:
            return []
        from sqlalchemy import select

        from app.models.test_run import TestResult

        result = await db.execute(
            select(TestResult)
            .where(TestResult.test_run_id == test_run_id)
            .where(TestResult.status.in_(["failed", "error"]))
            .limit(5)
        )
        rows = result.scalars().all()
        return [{"node_id": r.node_id, "traceback": r.traceback or ""} for r in rows]

    async def _load_findings(self, db: Any, analysis_id: UUID | None) -> list[dict[str, Any]]:
        if analysis_id is None:
            return []
        from sqlalchemy import select

        from app.models.finding import DBFinding

        result = await db.execute(
            select(DBFinding)
            .where(DBFinding.analysis_id == analysis_id)
            .order_by(DBFinding.severity.desc())
            .limit(10)
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

    async def _load_hypotheses(
        self, db: Any, debugging_session_id: UUID | None
    ) -> list[dict[str, Any]]:
        if debugging_session_id is None:
            return []
        import json

        from sqlalchemy import select

        from app.models.debugging import DebuggingHypothesis

        result = await db.execute(
            select(DebuggingHypothesis)
            .where(DebuggingHypothesis.session_id == debugging_session_id)
            .limit(3)
        )
        rows = result.scalars().all()
        hyps = []
        for r in rows:
            try:
                rec_tests = json.loads(r.recommended_tests)
            except Exception:
                rec_tests = []
            hyps.append(
                {
                    "id": str(r.id),
                    "root_cause": r.root_cause,
                    "confidence_label": r.confidence_label,
                    "recommended_tests": rec_tests,
                }
            )
        return hyps

    def _build_source_summaries(self, repository_path: str) -> list[dict[str, Any]]:
        """Extract function/class signatures from Python source files for context."""
        import ast as _ast

        summaries: list[dict[str, Any]] = []
        repo_root = Path(repository_path)
        py_files = list(repo_root.rglob("*.py"))[:20]

        for py_file in py_files:
            if any(
                part.startswith(".") or part in ("__pycache__", ".venv", "venv")
                for part in py_file.parts
            ):
                continue
            try:
                src = py_file.read_text(encoding="utf-8", errors="replace")
                tree = _ast.parse(src)
            except Exception:
                continue

            try:
                rel = py_file.relative_to(repo_root).as_posix()
            except ValueError:
                continue

            for node in _ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    summaries.append(
                        {
                            "file": rel,
                            "symbol": node.name,
                            "signature": _ast.unparse(node).split("\n")[0][:120],
                            "docstring": _ast.get_docstring(node) or "",
                        }
                    )
                    if len(summaries) >= 30:
                        return summaries
        return summaries

    async def _process_candidate(
        self,
        session_id: UUID,
        project_id: UUID,
        candidate: TestCandidate,
        repository_path: str,
    ) -> None:
        from app.database import async_session_factory
        from app.repositories.test_generation_repo import TestGenerationRepository

        # 1. Validate code
        validation = validate_test_code(candidate.test_code)
        quality_score, quality_notes = None, None
        exec_status = "not_run"
        exec_output = None

        # 2. Execute if valid
        if validation.valid:
            exec_status, exec_output = await self._execute_candidate(candidate, repository_path)
            quality_score, quality_notes = compute_quality_score(
                candidate.test_code, candidate.target_symbol
            )

        async with async_session_factory() as db:
            repo = TestGenerationRepository(db)
            await repo.add_generated_test(
                session_id=session_id,
                project_id=project_id,
                target_file=candidate.target_file,
                target_symbol=candidate.target_symbol,
                category=candidate.category,
                rationale=candidate.rationale,
                generated_code=candidate.test_code,
                confidence=candidate.confidence,
                validation_status="valid" if validation.valid else "invalid",
                validation_error=validation.error,
                execution_status=exec_status,
                execution_output=exec_output,
                quality_score=quality_score,
                quality_notes=quality_notes,
                hypothesis_id=None,
            )
            await db.commit()

    async def _execute_candidate(
        self, candidate: TestCandidate, repository_path: str
    ) -> tuple[str, str]:
        """Run a validated generated test in an isolated temp directory."""
        from app.execution import ExecutorFactory
        from app.execution.base import ExecutionConfig

        try:
            with tempfile.TemporaryDirectory(prefix="bugforge_gen_") as tmpdir:
                test_file = Path(tmpdir) / "test_generated.py"
                test_file.write_text(candidate.test_code)

                executor = ExecutorFactory.create()
                config = ExecutionConfig(
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "test_generated.py",
                        "-v",
                        "--tb=short",
                        "-q",
                    ],
                    working_directory=tmpdir,
                    timeout_seconds=_EXECUTION_TIMEOUT,
                    output_dir=tmpdir,
                    environment={"PYTHONPATH": repository_path},
                )
                result = await executor.execute(config)

                if result.timed_out:
                    return "timeout", "Execution timed out"
                output = (result.stdout + result.stderr)[:4_000]
                status = "passed" if result.exit_code == 0 else "failed"
                return status, output
        except Exception as exc:
            logger.warning("Generated test execution failed: %s", exc)
            return "error", str(exc)
