"""
Test runner service — Phase 2 entry point.

Discovers pytest, executes it via the configured TestExecutor, retrieves
the structured JSON report, and persists results.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from app.execution.base import ExecutionConfig, ExecutionResult, TestExecutor
from app.testing.pytest_parser import ParsedTestRun, parse_pytest_json

logger = logging.getLogger(__name__)

_MAX_OUTPUT_BYTES = 2 * 1024 * 1024  # 2 MB cap on stored stdout/stderr
_REPORT_FILENAME = "report.json"


@dataclass
class _PytestOutput:
    parsed: ParsedTestRun
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    command: str
    timed_out: bool = False


class TestRunNotFoundError(Exception):
    pass


class TestRunnerService:
    """Orchestrates test discovery, execution, and persistence."""

    def __init__(self, executor: TestExecutor, timeout_seconds: int = 120) -> None:
        self._executor = executor
        self._timeout = timeout_seconds

    async def execute_test_run(self, test_run_id: UUID, repository_path: str) -> None:
        """Full test pipeline. Designed to run as a background task."""
        from app.database import async_session_factory
        from app.repositories.test_run_repo import TestRunRepository

        logger.info("Starting test run %s for %s", test_run_id, repository_path)

        async with async_session_factory() as session:
            repo = TestRunRepository(session)
            await repo.update_status(test_run_id, "running")
            await session.commit()

        try:
            output = await self._run_pytest(repository_path)
        except Exception as exc:
            logger.exception("Test run %s failed during execution: %s", test_run_id, exc)
            async with async_session_factory() as session:
                repo = TestRunRepository(session)
                # "error" = BugForge infrastructure failure (not a test failure)
                await repo.fail(test_run_id, str(exc))
                await session.commit()
            return

        async with async_session_factory() as session:
            repo = TestRunRepository(session)
            try:
                await repo.complete(
                    run_id=test_run_id,
                    parsed=output.parsed,
                    repository_path=repository_path,
                    stdout=output.stdout,
                    stderr=output.stderr,
                    exit_code=output.exit_code,
                    duration_seconds=output.duration_seconds,
                    command=output.command,
                )
                # If the executor timed out, override the status to "timeout"
                if output.timed_out:
                    await repo.update_status(test_run_id, "timeout")
                await session.commit()
                logger.info(
                    "Test run %s completed — %d tests (%d passed, %d failed)",
                    test_run_id,
                    output.parsed.total,
                    output.parsed.passed,
                    output.parsed.failed,
                )
            except Exception as exc:
                logger.exception("Failed to persist test run %s: %s", test_run_id, exc)
                await session.rollback()
                async with async_session_factory() as s2:
                    await TestRunRepository(s2).fail(test_run_id, f"Persistence error: {exc}")
                    await s2.commit()

    async def _run_pytest(self, repository_path: str) -> _PytestOutput:
        """Execute pytest with JSON report output and return structured results."""
        repo_path = Path(repository_path)
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repository_path}")

        # Detect which Python executable to use (prefer the one running BugForge)
        python_exe = sys.executable or "python3"

        with tempfile.TemporaryDirectory(prefix="bugforge_run_") as output_dir:
            report_write_path = self._executor.get_artifact_write_path(
                ExecutionConfig(
                    command=[], working_directory=repository_path, output_dir=output_dir
                ),
                _REPORT_FILENAME,
            )

            command = [
                python_exe,
                "-m",
                "pytest",
                "--tb=short",
                f"--json-report-file={report_write_path}",
                "-q",
                "--no-header",
            ]
            command_str = " ".join(command)

            config = ExecutionConfig(
                command=command,
                working_directory=repository_path,
                timeout_seconds=self._timeout,
                output_dir=output_dir,
                environment={"PYTHONPATH": repository_path},
            )

            exec_result: ExecutionResult = await self._executor.execute(config)

            stdout = exec_result.stdout[:_MAX_OUTPUT_BYTES]
            stderr = exec_result.stderr[:_MAX_OUTPUT_BYTES]

            # Primary: read the structured JSON report
            report_json = exec_result.artifact_contents.get(_REPORT_FILENAME, "")
            parsed: ParsedTestRun

            if report_json:
                parsed = parse_pytest_json(report_json)
                if parsed.total == 0 and not parsed.results:
                    logger.debug("JSON report parsed but contains no tests")
            else:
                logger.warning("No JSON report produced for test run; falling back to text parsing")
                parsed = self._parse_text_output(stdout, stderr)

            if exec_result.timed_out:
                # Persist whatever partial results we got, then override status to timeout
                pass  # status is set in repo.complete() based on parsed counts

        return _PytestOutput(
            parsed=parsed,
            stdout=stdout,
            stderr=stderr,
            exit_code=exec_result.exit_code,
            duration_seconds=exec_result.duration_seconds,
            command=command_str,
            timed_out=exec_result.timed_out,
        )

    @staticmethod
    def _parse_text_output(stdout: str, stderr: str) -> ParsedTestRun:
        """Fallback text parser — used only when JSON report is unavailable."""
        run = ParsedTestRun()
        for line in (stdout + "\n" + stderr).splitlines():
            stripped = line.strip()
            if not (" passed" in stripped or " failed" in stripped or " error" in stripped):
                continue
            for part in stripped.split(","):
                part = part.strip()
                try:
                    count = int(part.split()[0])
                except (ValueError, IndexError):
                    continue
                if "passed" in part:
                    run.passed = count
                elif "failed" in part:
                    run.failed = count
                elif "skipped" in part:
                    run.skipped = count
                elif "error" in part:
                    run.errors = count
        run.total = run.passed + run.failed + run.skipped + run.errors
        return run
