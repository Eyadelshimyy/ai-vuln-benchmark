"""`profile suggest-from-market` writes files without stdin by default."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jobbot.exit_codes import SUCCESS
from jobbot.models.job import JobPosting


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / ".jobbot.toml").write_text(
        f'[paths]\ntemplates = "{project_root / "templates"}"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def _seed_jobs_with_gap(tmp_path: Path) -> None:
    """Two JDs sharing a term absent from profile.example so a gap appears."""
    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository

    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    repo = JobRepository(session)
    for i in range(2):
        repo.upsert_external(
            JobPosting(
                id="PENDING",
                title="Analyst",
                company=f"Co{i}",
                description=(
                    "Looking for Tableau experience and clear written reports. "
                    "Manejo de Tableau is required."
                ),
                skills=["Tableau"],
            )
        )


def test_suggest_from_market_default_skips_confirm(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    _seed_jobs_with_gap(tmp_path)

    def boom(*_a: object, **_k: object) -> bool:
        raise AssertionError("typer.confirm must not run without --ask/--promote")

    monkeypatch.setattr("typer.confirm", boom)

    from jobbot.cli import run_cli

    assert run_cli(["profile", "suggest-from-market"], standalone_mode=False) == SUCCESS
    assert (tmp_path / "output" / "profile_market_suggestion.md").is_file()
    suggested = tmp_path / "data" / "profile.suggested.yaml"
    assert suggested.is_file()
    raw = yaml.safe_load(suggested.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
