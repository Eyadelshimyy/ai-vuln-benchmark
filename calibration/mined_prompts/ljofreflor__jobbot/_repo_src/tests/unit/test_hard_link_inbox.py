"""Park / drain hard-link inbox (phone share → desktop ingest)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.config import JobbotConfig, PathsConfig
from jobbot.exit_codes import SUCCESS, VALIDATION_FAILURE
from jobbot.jobs.inbox import (
    inbox_path,
    list_parked,
    normalize_park_url,
    park_url,
    remove_parked,
)


def _config(tmp_path: Path) -> JobbotConfig:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    return JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(templates=Path("templates"), output=Path("output")),
    )


def test_normalize_strips_ios_share_tracking() -> None:
    url = (
        "https://cl.indeed.com/viewjob?jk=9e61738ac9095c40&from=appshareios"
    )
    assert normalize_park_url(url) == (
        "https://cl.indeed.com/viewjob?jk=9e61738ac9095c40"
    )


def test_normalize_splits_double_pasted_share_link() -> None:
    glued = (
        "https://cl.indeed.com/viewjob?jk=9e61738ac9095c40&from=appshareios"
        "https://cl.indeed.com/viewjob?jk=9e61738ac9095c40&from=appshareios"
    )
    assert normalize_park_url(glued) == (
        "https://cl.indeed.com/viewjob?jk=9e61738ac9095c40"
    )


def test_park_dedupes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    url = "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97&from=appshareios"
    first, existed = park_url(config, url)
    second, again = park_url(config, url)
    assert first == second == "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97"
    assert existed is False
    assert again is True
    assert list_parked(config) == [first]


def test_remove_parked(tmp_path: Path) -> None:
    config = _config(tmp_path)
    park_url(config, "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97")
    park_url(config, "https://cl.indeed.com/viewjob?jk=25db4649b81bfb9d")
    assert remove_parked(config, "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97")
    assert list_parked(config) == ["https://cl.indeed.com/viewjob?jk=25db4649b81bfb9d"]


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


def test_cli_park_does_not_fetch(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli
    from jobbot.config import load_config

    fetched: list[str] = []

    def _boom(*_a: object, **_k: object) -> object:
        fetched.append("called")
        raise AssertionError("park must not fetch")

    monkeypatch.setattr("jobbot.adapters.indeed.jobs.IndeedJobSource.get_job", _boom)

    glued = (
        "https://cl.indeed.com/viewjob?jk=9e61738ac9095c40&from=appshareios"
        "https://cl.indeed.com/viewjob?jk=9e61738ac9095c40&from=appshareios"
    )
    assert run_cli(["get", glued, "--park"], standalone_mode=False) == SUCCESS
    out = capsys.readouterr().out
    assert "Parked" in out
    assert "9e61738ac9095c40" in out
    assert fetched == []
    assert list_parked(load_config()) == [
        "https://cl.indeed.com/viewjob?jk=9e61738ac9095c40"
    ]
    assert inbox_path(load_config()).is_file()


def test_cli_parked_drains_with_fixture_ingest_path(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drain calls the normal ingest path; stub live Indeed with a fixture parse."""
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.adapters.indeed.jobs import card_to_job_posting
    from jobbot.cli import run_cli
    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository

    html = """<!DOCTYPE html><html><body>
    <h1 data-testid="jobsearch-JobInfoHeader-title">Data Scientist</h1>
    <div data-testid="inlineHeader-companyName">Acme</div>
    <div data-testid="jobsearch-JobInfoHeader-locationText">Santiago</div>
    <div id="jobDescriptionText"><p>Python.</p></div>
    </body></html>"""

    def fake_get_job(self: object, job_id: str) -> object:  # noqa: ANN001
        card = {
            "source_job_id": job_id,
            "url": f"https://cl.indeed.com/viewjob?jk={job_id}",
            "title": "",
            "company": "",
            "location": None,
            "snippet": "",
        }
        return card_to_job_posting(card, detail_html=html, placeholder_id="TMP")

    monkeypatch.setattr(
        "jobbot.adapters.indeed.jobs.IndeedJobSource.get_job",
        fake_get_job,
    )

    assert (
        run_cli(
            [
                "get",
                "https://cl.indeed.com/viewjob?jk=9e61738ac9095c40&from=appshareios",
                "--park",
            ],
            standalone_mode=False,
        )
        == SUCCESS
    )
    assert run_cli(["get", "--parked"], standalone_mode=False) == SUCCESS

    config = load_config()
    assert list_parked(config) == []
    session = make_session_factory(make_engine(config.database_path))()
    jobs = JobRepository(session).list_all()
    assert len(jobs) == 1
    assert jobs[0].title == "Data Scientist"
    assert jobs[0].source_job_id == "9e61738ac9095c40"


def test_cli_park_rejects_bad_url(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    assert (
        run_cli(["get", "https://cl.indeed.com/viewjob?foo=bar", "--park"], standalone_mode=False)
        == VALIDATION_FAILURE
    )
