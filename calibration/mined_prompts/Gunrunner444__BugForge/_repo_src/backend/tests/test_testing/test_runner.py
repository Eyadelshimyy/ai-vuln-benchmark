"""Tests for the TestRunnerService and pytest parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.execution.base import ExecutionConfig
from app.testing.pytest_parser import parse_pytest_json


class TestPytestParser:
    def _make_report(self, tests: list[dict], summary: dict) -> str:
        return json.dumps({"tests": tests, "summary": summary})

    def test_all_passed(self) -> None:
        report = self._make_report(
            tests=[
                {"nodeid": "tests/test_a.py::test_one", "outcome": "passed", "duration": 0.01},
                {"nodeid": "tests/test_a.py::test_two", "outcome": "passed", "duration": 0.02},
            ],
            summary={"total": 2, "passed": 2, "failed": 0, "skipped": 0},
        )
        result = parse_pytest_json(report)
        assert result.passed == 2
        assert result.failed == 0
        assert result.total == 2

    def test_with_failure(self) -> None:
        report = self._make_report(
            tests=[
                {
                    "nodeid": "tests/test_a.py::test_bad",
                    "outcome": "failed",
                    "duration": 0.05,
                    "call": {
                        "longrepr": "AssertionError: assert 1 == 2",
                        "stdout": "",
                        "stderr": "",
                    },
                },
            ],
            summary={"total": 1, "passed": 0, "failed": 1, "skipped": 0},
        )
        result = parse_pytest_json(report)
        assert result.failed == 1
        assert result.results[0].traceback == "AssertionError: assert 1 == 2"

    def test_with_skipped(self) -> None:
        report = self._make_report(
            tests=[
                {
                    "nodeid": "tests/test_a.py::test_skip",
                    "outcome": "skipped",
                    "duration": 0.001,
                    "setup": {"longrepr": "Skipped: not now"},
                },
            ],
            summary={"total": 1, "passed": 0, "failed": 0, "skipped": 1},
        )
        result = parse_pytest_json(report)
        assert result.skipped == 1
        assert result.results[0].skip_reason == "Skipped: not now"

    def test_node_id_splitting(self) -> None:
        report = self._make_report(
            tests=[
                {
                    "nodeid": "tests/test_calc.py::TestClass::test_method",
                    "outcome": "passed",
                    "duration": 0.01,
                },
            ],
            summary={"total": 1, "passed": 1},
        )
        result = parse_pytest_json(report)
        r = result.results[0]
        assert r.test_file == "tests/test_calc.py"
        assert r.test_name == "TestClass::test_method"

    def test_invalid_json_returns_empty(self) -> None:
        result = parse_pytest_json("not json")
        assert result.total == 0
        assert result.results == []

    def test_empty_json_object(self) -> None:
        result = parse_pytest_json("{}")
        assert result.total == 0

    def test_duration_captured(self) -> None:
        report = self._make_report(
            tests=[
                {"nodeid": "tests/test_a.py::test_one", "outcome": "passed", "duration": 1.234},
            ],
            summary={"total": 1, "passed": 1},
        )
        result = parse_pytest_json(report)
        assert result.results[0].duration_seconds == pytest.approx(1.234)


class TestLocalExecutor:
    async def test_successful_execution(self, tmp_path: Path) -> None:
        from app.execution.local_executor import LocalTestExecutor

        executor = LocalTestExecutor()
        result = await executor.execute(
            ExecutionConfig(
                command=["python3", "-c", "print('hello')"],
                working_directory=str(tmp_path),
                timeout_seconds=10,
            )
        )
        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_failing_command(self, tmp_path: Path) -> None:
        from app.execution.local_executor import LocalTestExecutor

        executor = LocalTestExecutor()
        result = await executor.execute(
            ExecutionConfig(
                command=["python3", "-c", "raise SystemExit(1)"],
                working_directory=str(tmp_path),
                timeout_seconds=10,
            )
        )
        assert result.exit_code != 0

    async def test_timeout(self, tmp_path: Path) -> None:
        from app.execution.local_executor import LocalTestExecutor

        executor = LocalTestExecutor()
        result = await executor.execute(
            ExecutionConfig(
                command=["python3", "-c", "import time; time.sleep(10)"],
                working_directory=str(tmp_path),
                timeout_seconds=1,
            )
        )
        assert result.timed_out is True

    async def test_missing_command(self, tmp_path: Path) -> None:
        from app.execution.local_executor import LocalTestExecutor

        executor = LocalTestExecutor()
        result = await executor.execute(
            ExecutionConfig(
                command=["__command_that_does_not_exist__"],
                working_directory=str(tmp_path),
                timeout_seconds=5,
            )
        )
        assert result.exit_code == -1
        assert result.error_message is not None

    async def test_is_available(self) -> None:
        from app.execution.local_executor import LocalTestExecutor

        executor = LocalTestExecutor()
        assert await executor.is_available() is True

    async def test_artifact_capture(self, tmp_path: Path) -> None:
        """Executor should read files placed in output_dir after execution."""
        from app.execution.local_executor import LocalTestExecutor

        executor = LocalTestExecutor()
        result = await executor.execute(
            ExecutionConfig(
                command=["python3", "-c", f"open('{tmp_path}/artifact.txt','w').write('hello')"],
                working_directory=str(tmp_path),
                timeout_seconds=10,
                output_dir=str(tmp_path),
            )
        )
        assert result.exit_code == 0
        assert "artifact.txt" in result.artifact_contents
        assert result.artifact_contents["artifact.txt"] == "hello"

    async def test_secrets_not_inherited(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SECRET_KEY must be stripped from the subprocess environment."""
        monkeypatch.setenv("SECRET_KEY", "should_not_leak")

        from app.execution.local_executor import LocalTestExecutor

        executor = LocalTestExecutor()
        result = await executor.execute(
            ExecutionConfig(
                command=[
                    "python3",
                    "-c",
                    "import os; print(os.environ.get('SECRET_KEY','MISSING'))",
                ],
                working_directory=str(tmp_path),
                timeout_seconds=10,
            )
        )
        assert "MISSING" in result.stdout
        assert "should_not_leak" not in result.stdout


class TestExecutorFactory:
    def test_factory_returns_executor(self) -> None:
        from app.execution import ExecutorFactory

        executor = ExecutorFactory.create()
        # Should return some concrete executor without raising
        from app.execution.base import TestExecutor

        assert isinstance(executor, TestExecutor)
