"""Learning a form from the CLI: local knowledge, nothing submitted."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.exit_codes import SUCCESS
from jobbot.portals.form_learn import default_form_knowledge_path, load_form_knowledge

FIXTURE = Path("tests/fixtures/forms/greenhouse_apply.html")


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def test_form_learn_from_a_fixture_stores_the_questions(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    code = run_cli(
        [
            "portals",
            "form-learn",
            "https://boards.greenhouse.io/acme/jobs/4001",
            "--fixture",
            str(project_root / FIXTURE),
            "--company",
            "Acme",
        ],
        standalone_mode=False,
    )

    assert code == SUCCESS
    forms = load_form_knowledge(default_form_knowledge_path(tmp_path))
    assert len(forms) == 1
    assert forms[0].company == "Acme"
    assert "When could you start?" in forms[0].screening_questions()


def test_learning_the_same_form_twice_keeps_one_entry(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    args = [
        "portals",
        "form-learn",
        "https://boards.greenhouse.io/acme/jobs/4001",
        "--fixture",
        str(project_root / FIXTURE),
    ]
    assert run_cli(args, standalone_mode=False) == SUCCESS
    assert run_cli(args, standalone_mode=False) == SUCCESS

    assert len(load_form_knowledge(default_form_knowledge_path(tmp_path))) == 1


def test_no_typed_value_reaches_the_stored_file(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    run_cli(
        [
            "portals",
            "form-learn",
            "https://boards.greenhouse.io/acme/jobs/4001",
            "--fixture",
            str(project_root / FIXTURE),
        ],
        standalone_mode=False,
    )

    stored = default_form_knowledge_path(tmp_path).read_text(encoding="utf-8")
    assert "ada@example.com" not in stored
    assert "do-not-store-me" not in stored


def test_a_url_without_a_fixture_needs_the_network_flag(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The suite must never reach out; fetching is opt-in and explicit."""
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    code = run_cli(
        ["portals", "form-learn", "https://boards.greenhouse.io/acme/jobs/4001"],
        standalone_mode=False,
    )

    assert code != SUCCESS
    assert not default_form_knowledge_path(tmp_path).exists()


def test_apply_learns_the_form_while_you_fill_it(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The return path of the loop: postulating teaches us what the company asks."""
    _workspace(tmp_path, project_root, monkeypatch)

    import jobbot.cli as cli
    from jobbot.cli import run_cli
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository
    from jobbot.models.job import JobPosting

    config = cli.load_config()
    session = make_session_factory(make_engine(config.database_path))()
    JobRepository(session).upsert_external(
        JobPosting.model_validate(
            {
                "id": "J0001",
                "source": "indeed",
                "source_job_id": "abc123",
                "title": "Un cargo",
                "company": "Acme",
                "url": "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97",
                "ats_url": "https://boards.greenhouse.io/acme/jobs/4001",
                "description": "Una vacante.",
            }
        )
    )
    session.commit()

    html = (project_root / FIXTURE).read_text(encoding="utf-8")
    monkeypatch.setattr(cli, "_fetch_public_html", lambda url: html)
    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.browser.background.open_url", lambda url, **_k: opened.append(url) or url
    )

    code = run_cli(
        ["application", "apply", "J0001", "--apply", "--yes"],
        standalone_mode=False,
    )

    assert code == SUCCESS
    assert opened == ["https://boards.greenhouse.io/acme/jobs/4001"], "it opens, it does not submit"
    forms = load_form_knowledge(default_form_knowledge_path(tmp_path))
    assert len(forms) == 1
    assert "When could you start?" in forms[0].screening_questions()

    from jobbot.applications.manager import ApplicationRepository
    from jobbot.models.application import ApplicationStatus

    apps = ApplicationRepository(session).list_all()
    assert len(apps) == 1
    assert apps[0].status is ApplicationStatus.PREPARED, (
        "--yes must not mark applied after only opening the ATS"
    )
