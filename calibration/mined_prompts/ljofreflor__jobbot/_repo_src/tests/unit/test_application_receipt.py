"""Opening an ATS is not evidence the application was submitted."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jobbot.exit_codes import SUCCESS, VALIDATION_FAILURE
from jobbot.models.application import ApplicationStatus


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


def test_yes_to_apply_stays_prepared_and_repeats_the_url(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    import jobbot.cli as cli
    from jobbot.applications.manager import ApplicationRepository
    from jobbot.cli import run_cli
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository
    from jobbot.models.job import JobPosting

    monkeypatch.setattr("jobbot.browser.background.open_url", lambda url, **_k: url)
    monkeypatch.setattr("typer.confirm", lambda *_a, **_k: True)
    monkeypatch.setattr("jobbot.cli._fetch_public_html", lambda _url: None)
    monkeypatch.setattr(
        "jobbot.jobs.closure.fetch_posting_text",
        lambda _url, **_k: "<html>still open</html>",
    )
    config = cli.load_config()
    session = make_session_factory(make_engine(config.database_path))()
    url = "https://boards.greenhouse.io/acme/jobs/1"
    stored = JobRepository(session).upsert_external(
        JobPosting.model_validate(
            {
                "id": "PENDING",
                "source": "manual",
                "title": "Un cargo",
                "company": "Acme",
                "url": url,
                "ats_url": url,
                "ats_kind": "greenhouse",
                "description": "Una descripción.",
            }
        )
    )
    code = run_cli(
        ["application", "apply", stored.id, "--apply", "--yes"],
        standalone_mode=False,
    )
    assert code == SUCCESS
    out = capsys.readouterr().out
    assert "No portal receipt" in out
    assert url in out
    assert "Status → applied" not in out
    apps = ApplicationRepository(session).list_all()
    assert len(apps) == 1
    assert apps[0].status is ApplicationStatus.PREPARED
    events = ApplicationRepository(session).events_for(apps[0].id)
    assert any(event.event_type == "no_receipt" and event.detail == url for event in events)
    sheet = yaml.safe_load(
        (config.output_dir / "jobs" / stored.id / "application" / "ats_prefill.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert "email" not in sheet["fields"]
    assert "phone" not in sheet["fields"]
    assert "example.com" not in yaml.safe_dump(sheet["fields"])


def test_filled_posting_is_not_opened(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    import jobbot.cli as cli
    from jobbot.cli import run_cli
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository
    from jobbot.models.job import JobPosting

    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.browser.background.open_url", lambda url, **_k: opened.append(url) or url
    )
    config = cli.load_config()
    session = make_session_factory(make_engine(config.database_path))()
    stored = JobRepository(session).upsert_external(
        JobPosting.model_validate(
            {
                "id": "PENDING",
                "source": "manual",
                "title": "Un cargo",
                "company": "Empresa",
                "url": "https://careers.example.com/global/en/job/abc/un-cargo",
                "description": "Este cargo ya fue cubierto.",
            }
        )
    )
    code = run_cli(
        ["application", "apply", stored.id, "--apply", "--yes"],
        standalone_mode=False,
    )
    assert code == VALIDATION_FAILURE
    assert opened == []
    err = capsys.readouterr().err
    assert "Not opening" in err
    assert "not applied" in err.casefold()
    assert "Recorded failure" not in err
