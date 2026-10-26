"""Decide whether a commit has to run the unit suite. Local only, no network.

The pre-commit hook used to hold this list as a shell regex, which drifts from
what the tests check. Policy counts as code here: a remote agent working from a
clone has `AGENTS.md` and nothing else, so a change there can break behaviour as
surely as a change in a module.
"""

from __future__ import annotations

import re
import subprocess
import sys

_TRIGGERS: tuple[tuple[str, str, str], ...] = (
    (r"\.py$", "*.py", "python module"),
    (r"^pyproject\.toml$", "pyproject.toml", "dependencies and tool config"),
    (r"^templates/", "templates/", "rendered CV output"),
    (r"^tests/", "tests/", "the suite itself"),
    (r"^AGENTS\.md$", "AGENTS.md", "project policy for humans and agents"),
    (r"^\.githooks/", ".githooks/", "the guard that runs before every commit"),
    (r"^asesoria/", "asesoria/", "the public page, checked for privacy by its own test"),
    (r"^\.github/workflows/", ".github/workflows/", "the Pages deploy, checked by a test"),
)

TRIGGER_HELP = "runs the suite for: " + ", ".join(
    f"{label} ({why})" for _, label, why in _TRIGGERS
)

_COMPILED = tuple(re.compile(pattern) for pattern, _, _ in _TRIGGERS)


def tests_needed(paths: list[str]) -> bool:
    """True when any staged path can change behaviour."""
    return any(
        rule.search(clean) for path in paths if (clean := path.strip()) for rule in _COMPILED
    )


def staged_paths() -> list[str]:
    result = subprocess.run(  # noqa: S603, S607 — fixed git command, no shell
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    """Exit 0 when the suite must run, 1 when this commit cannot break it."""
    return 0 if tests_needed(staged_paths()) else 1


if __name__ == "__main__":
    sys.exit(main())
