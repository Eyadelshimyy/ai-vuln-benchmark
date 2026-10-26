"""HITL: --yes must not claim the application was sent or submitted."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.exit_codes import SUCCESS
from jobbot.models.application import ApplicationStatus


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / ".jobbot.toml").write_text(
        f'[paths]\ntemplates = "{project_root / "templates"}"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def test_email_apply_yes_stays_prepared_until_user_sends(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)

    import jobbot.cli as cli
    from jobbot.applications.manager import ApplicationRepository
    from jobbot.cli import run_cli
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository
    from jobbot.models.job import JobPosting

    config = cli.load_config()
    session = make_session_factory(make_engine(config.database_path))()
    stored = JobRepository(session).upsert_external(
        JobPosting.model_validate(
            {
                "id": "PENDING",
                "source": "linkedin_post",
                "source_job_id": "email-1",
                "title": "Un cargo",
                "company": "Acme",
                "url": "https://www.linkedin.com/feed/update/urn:li:activity:1/",
                "ats_url": "mailto:seleccion@empresa.cl",
                "ats_kind": "email",
                "description": "Enviar CV a seleccion@empresa.cl",
            }
        )
    )
    session.commit()

    monkeypatch.setattr("jobbot.browser.background.open_url", lambda url, **_k: url)
    # Avoid the Playwright Gmail attach path; --no-attach keeps the HITL compose URL.
    code = run_cli(
        ["application", "apply", stored.id, "--apply", "--yes", "--no-attach"],
        standalone_mode=False,
    )
    assert code == SUCCESS
    apps = ApplicationRepository(session).list_all()
    assert len(apps) == 1
    assert apps[0].job_id == stored.id
    assert apps[0].status is ApplicationStatus.PREPARED


def test_linkedin_message_apply_yes_stays_prepared_and_never_sends(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)

    import jobbot.cli as cli
    from jobbot.applications.manager import ApplicationRepository
    from jobbot.cli import run_cli
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository
    from jobbot.models.job import JobPosting

    config = cli.load_config()
    session = make_session_factory(make_engine(config.database_path))()
    stored = JobRepository(session).upsert_external(
        JobPosting.model_validate(
            {
                "id": "PENDING",
                "source": "linkedin_post",
                "source_job_id": "dm-1",
                "title": "Enfermera Clínica",
                "company": "Unknown company",
                "url": (
                    "https://www.linkedin.com/posts/recruiter_activity-7511877806738411520-MJ5G"
                ),
                "description": (
                    "Si te interesa, mandame un mensaje indicando que te interesa este puesto."
                ),
            }
        )
    )
    session.commit()

    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.browser.background.open_url",
        lambda url, **_k: opened.append(url) or url,
    )
    code = run_cli(
        ["application", "apply", stored.id, "--apply", "--yes"],
        standalone_mode=False,
    )
    assert code == SUCCESS
    assert opened == [stored.url]
    apps = ApplicationRepository(session).list_all()
    assert len(apps) == 1
    assert apps[0].status is ApplicationStatus.PREPARED
