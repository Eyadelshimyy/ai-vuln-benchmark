"""The pre-commit hook is the only automatic gate; keep its contract pinned."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

# Imported as a module: pytest would collect a bare `tests_needed` as a test.
from jobbot.ops import precommit

HOOK = Path(__file__).resolve().parents[2] / ".githooks" / "pre-commit"


def test_hook_is_executable() -> None:
    assert HOOK.is_file()
    assert os.access(HOOK, os.X_OK), "git silently ignores a non-executable hook"


def test_hook_runs_the_pii_guard_and_the_tests() -> None:
    body = HOOK.read_text(encoding="utf-8")
    assert "jobbot.ops.pii_guard" in body
    assert "pytest" in body
    # The guard runs first: a commit carrying PII must fail before anything slower.
    assert body.index("pii_guard") < body.index("pytest")


def test_hook_is_valid_shell() -> None:
    subprocess.run(["sh", "-n", str(HOOK)], check=True)


@pytest.mark.parametrize(
    "staged, runs_tests",
    [
        ("src/jobbot/cli.py", True),
        ("tests/unit/test_matching.py", True),
        ("pyproject.toml", True),
        ("templates/cv_moderncv.tex.j2", True),
        ("AGENTS.md", True),
        (".githooks/pre-commit", True),
        ("README.md", False),
        ("docs/library-audit.md", False),
    ],
)
def test_hook_only_runs_tests_when_code_is_staged(staged: str, runs_tests: bool) -> None:
    """A docs-only commit should not pay for the suite."""
    assert precommit.tests_needed([staged]) is runs_tests


def test_hook_delegates_the_decision_to_the_tested_module() -> None:
    """A shell regex in the hook would drift from what this file checks."""
    assert "jobbot.ops.precommit" in HOOK.read_text(encoding="utf-8")
