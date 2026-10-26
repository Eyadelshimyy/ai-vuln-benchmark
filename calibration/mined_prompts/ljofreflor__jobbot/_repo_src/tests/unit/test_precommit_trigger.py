"""Which staged paths must run the suite before the commit lands."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.ops import precommit


@pytest.mark.parametrize(
    "path",
    [
        "src/jobbot/cv/advisor.py",
        "tests/unit/test_cv_advisor.py",
        "pyproject.toml",
        "templates/cv.tex.j2",
        "AGENTS.md",
        "asesoria/index.html",
        "asesoria/styles.css",
        ".github/workflows/pages.yml",
    ],
)
def test_these_paths_run_the_suite(path: str) -> None:
    assert precommit.tests_needed([path]), f"{path} must trigger the tests"


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "docs/tutorial.md",
        "data/profile.example.yaml",
        ".gitignore",
    ],
)
def test_prose_and_examples_do_not(path: str) -> None:
    assert not precommit.tests_needed([path]), f"{path} should not trigger the tests"


def test_policy_changes_are_verified_like_code() -> None:
    """AGENTS.md is the whole policy for a remote agent: a change there is code."""
    assert precommit.tests_needed(["AGENTS.md"])
    assert precommit.tests_needed(["README.md", "AGENTS.md"])


def test_nothing_staged_means_nothing_to_run() -> None:
    assert not precommit.tests_needed([])
    assert not precommit.tests_needed(["", "   "])


def test_the_hook_asks_this_module_instead_of_holding_its_own_pattern(
    project_root: Path,
) -> None:
    """A regex duplicated in shell drifts from the tested one."""
    hook = (project_root / ".githooks" / "pre-commit").read_text(encoding="utf-8")

    assert "jobbot.ops.precommit" in hook
    assert "^templates/" not in hook, "the trigger list must live in one place only"


def test_help_names_the_paths_it_watches() -> None:
    assert "AGENTS.md" in precommit.TRIGGER_HELP
