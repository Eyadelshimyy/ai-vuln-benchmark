"""CLI: application apply --all (HITL queue, never auto-submit)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.exit_codes import SUCCESS, VALIDATION_FAILURE
from jobbot.models.application import ApplicationStatus
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


def _seed_two_jobs(tmp_path: Path) -> tuple[str, str]:
    from jobbot.applications.manager import ApplicationRepository
    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository

    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    repo = JobRepository(session)
    a = repo.upsert_external(
        JobPosting(
            id="PENDING",
            title="Analyst",
            company="Northwind",
            url="https://careers.example-corp.test/jobs/a",
            ats_url="https://careers.example-corp.test/jobs/a",
            ats_kind="unknown",
            match_score=80.0,
            description="Role A",
        )
    )
    b = repo.upsert_external(
        JobPosting(
            id="PENDING",
            title="Editor",
            company="Contoso",
            url="https://careers.example-corp.test/jobs/b",
            ats_url="https://careers.example-corp.test/jobs/b",
            ats_kind="unknown",
            match_score=40.0,
            description="Role B",
        )
    )
    apps = ApplicationRepository(session)
    apps.upsert_for_job(a.id, status=ApplicationStatus.PREPARED, package_dir="x")
    apps.upsert_for_job(b.id, status=ApplicationStatus.PREPARED, package_dir="y")
    return a.id, b.id


def test_apply_all_dry_run_lists_and_opens_nothing(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    first, second = _seed_two_jobs(tmp_path)
    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.adapters.ats.apply.open_ats_in_browser",
        lambda url: opened.append(url),
    )
    from jobbot.cli import run_cli

    assert run_cli(["application", "apply", "--all"], standalone_mode=False) == SUCCESS
    out = capsys.readouterr().out
    assert "Batch apply" in out
    assert first in out
    assert second in out
    assert "Dry-run" in out
    assert opened == []


def test_apply_all_apply_opens_one_then_stops_when_refusing_next(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    first, second = _seed_two_jobs(tmp_path)
    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.adapters.ats.apply.open_ats_in_browser",
        lambda url: opened.append(url),
    )
    answers = iter([True, False])  # open first; refuse next

    def confirm(msg: str, default: bool = False) -> bool:
        _ = default
        try:
            return next(answers)
        except StopIteration:
            return False

    monkeypatch.setattr("typer.confirm", confirm)
    monkeypatch.setattr("jobbot.cli._learn_form_from_apply", lambda *_a, **_k: None)
    monkeypatch.setattr("jobbot.cli._filled_vacancy", lambda _job: None)
    from jobbot.cli import run_cli

    assert (
        run_cli(["application", "apply", "--all", "--apply"], standalone_mode=False)
        == SUCCESS
    )
    assert len(opened) == 1
    assert first in opened[0] or "jobs/a" in opened[0]
    assert second not in "".join(opened)


def test_apply_rejects_job_id_with_all(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    code = run_cli(
        ["application", "apply", "J0001", "--all"],
        standalone_mode=False,
    )
    assert code == VALIDATION_FAILURE
