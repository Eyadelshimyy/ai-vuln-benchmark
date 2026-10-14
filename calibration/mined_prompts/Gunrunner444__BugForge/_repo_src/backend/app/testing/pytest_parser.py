"""Parse pytest JSON report output into structured TestResult records."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedTestResult:
    node_id: str
    test_file: str | None
    test_name: str
    status: str  # passed | failed | skipped | error
    duration_seconds: float | None
    traceback: str | None
    stdout: str | None
    stderr: str | None
    skip_reason: str | None


@dataclass
class ParsedTestRun:
    results: list[ParsedTestResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0


def parse_pytest_json(json_output: str) -> ParsedTestRun:
    """Parse pytest's --json-report output (from pytest-json-report plugin)."""
    try:
        data = json.loads(json_output)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse pytest JSON output: %s", exc)
        return ParsedTestRun()

    run = ParsedTestRun()
    summary = data.get("summary", {})
    run.total = summary.get("total", 0)
    run.passed = summary.get("passed", 0)
    run.failed = summary.get("failed", 0)
    run.skipped = summary.get("skipped", 0)
    run.errors = summary.get("error", 0)

    for test in data.get("tests", []):
        node_id: str = test.get("nodeid", "")
        # Split node_id into file and test name
        test_file: str | None = None
        test_name: str = node_id
        if "::" in node_id:
            test_file, test_name = node_id.split("::", 1)

        outcome = test.get("outcome", "error")
        status_map = {
            "passed": "passed",
            "failed": "failed",
            "skipped": "skipped",
            "error": "error",
            "xfailed": "skipped",
            "xpassed": "passed",
        }
        status = status_map.get(outcome, "error")

        duration = test.get("duration")

        # Capture traceback from longrepr
        traceback: str | None = None
        call = test.get("call", {})
        if call and call.get("longrepr"):
            traceback = str(call["longrepr"])
        elif test.get("longrepr"):
            traceback = str(test["longrepr"])

        stdout_str: str | None = call.get("stdout") if call else None
        stderr_str: str | None = call.get("stderr") if call else None

        # Skip reason
        skip_reason: str | None = None
        if status == "skipped":
            setup = test.get("setup", {})
            if setup and setup.get("longrepr"):
                skip_reason = str(setup["longrepr"])

        run.results.append(
            ParsedTestResult(
                node_id=node_id,
                test_file=test_file,
                test_name=test_name,
                status=status,
                duration_seconds=float(duration) if duration is not None else None,
                traceback=traceback,
                stdout=stdout_str,
                stderr=stderr_str,
                skip_reason=skip_reason,
            )
        )

    return run
