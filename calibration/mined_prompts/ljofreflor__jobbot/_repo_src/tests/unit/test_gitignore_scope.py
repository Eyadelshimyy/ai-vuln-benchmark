"""Regression: the private-CV ignore patterns also swallowed source packages.

`cv/` (meant for a private CV folder at the repo root) matched `src/jobbot/cv/` too, so
seven modules — including the CV builder and `cv propagate` — were never tracked by git.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PRIVATE_PATHS = (
    "cv/cv.tex",
    "vitae/old.tex",
    "legacy/notes.md",
    "private/secrets.yaml",
    "data/.vault.yaml",
    "sandboxes/rocio/data/.vault.yaml",
)


def _ignored(project_root: Path, paths: list[str]) -> list[str]:
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "check-ignore", "--stdin"],  # noqa: S607 - git resolved from PATH
        input="\n".join(paths),
        capture_output=True,
        text=True,
        check=False,
        cwd=project_root,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


@pytest.fixture
def repo(project_root: Path) -> Path:
    if not (project_root / ".git").exists():
        pytest.skip("not a git checkout")
    return project_root


def test_no_source_module_is_hidden_from_git(repo: Path) -> None:
    modules = [
        str(path.relative_to(repo))
        for path in (repo / "src").rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    assert modules, "no source modules found"
    assert _ignored(repo, modules) == []


def test_private_cv_folders_stay_ignored_at_the_root(repo: Path) -> None:
    assert sorted(_ignored(repo, list(PRIVATE_PATHS))) == sorted(PRIVATE_PATHS)


DATA_BACKUPS = (
    "data/companies.yaml.bak-2026-10-01",
    "data/profile.yaml.bak",
    "data/profile.yaml.2026-10-01",
    "data/portals.yaml.orig",
    "data/recruiters.yaml.old",
    "data/form_knowledge.yaml.bak-1",
    "data/notes.bak-2026-10-01",
    "data/notes.orig",
)

DATA_TEMPLATES = (
    "data/profile.example.yaml",
    "data/vault.example.yaml",
    "data/portals.example.yaml",
    "data/companies.example.yaml",
    "data/companies-cl.example.yaml",
    "data/README.md",
)


def test_backups_of_local_data_files_are_ignored(repo: Path) -> None:
    assert sorted(_ignored(repo, list(DATA_BACKUPS))) == sorted(DATA_BACKUPS)


def test_example_templates_are_not_ignored(repo: Path) -> None:
    assert _ignored(repo, list(DATA_TEMPLATES)) == []
