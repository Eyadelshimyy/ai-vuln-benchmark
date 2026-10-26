"""Indeed hard links via develop's ingest_hard_link (store → match → package)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.config import JobbotConfig, PathsConfig
from jobbot.db.engine import make_engine, make_session_factory
from jobbot.jobs.career_page import ClosedPostingError
from jobbot.jobs.from_url import (
    UnknownPortalError,
    UnsupportedPortalFetchError,
    ingest_hard_link,
)
from jobbot.models.candidate import Candidate
from jobbot.portals.detect import AtsKind
from tests.fixtures.profile import nurse_profile_dict


def _config(tmp_path: Path, project_root: Path) -> JobbotConfig:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    return JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(templates=project_root / "templates", output=Path("output")),
    )


def _candidate() -> Candidate:
    return Candidate.model_validate(nurse_profile_dict())


def test_get_indeed_job_with_fixture(tmp_path: Path, project_root: Path) -> None:
    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    html = (project_root / "tests" / "fixtures" / "indeed_job_detail.html").read_text(
        encoding="utf-8"
    )
    url = "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97"

    result = ingest_hard_link(config, session, url, candidate=_candidate(), html=html)

    assert result.job.source == "indeed"
    assert result.job.source_job_id == "8a5fab1a7c476a97"
    assert result.job.url == url
    assert result.job.ats_kind == AtsKind.INDEED.value
    assert result.job.title
    assert result.package_dir is not None


def test_get_strips_tracking_params_from_stored_url(
    tmp_path: Path, project_root: Path
) -> None:
    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    html = (project_root / "tests" / "fixtures" / "indeed_job_detail.html").read_text(
        encoding="utf-8"
    )
    url_with_tracking = "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97&from=email&tk=xyz"

    result = ingest_hard_link(
        config, session, url_with_tracking, candidate=_candidate(), html=html
    )

    assert result.job.url == "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97"
    assert "from=" not in (result.job.url or "")
    assert "tk=" not in (result.job.url or "")


def test_get_rejects_url_without_jk(tmp_path: Path, project_root: Path) -> None:
    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    with pytest.raises(UnsupportedPortalFetchError, match="Missing 'jk'"):
        ingest_hard_link(
            config,
            session,
            "https://cl.indeed.com/viewjob?foo=bar",
            candidate=_candidate(),
        )


def test_get_rejects_unknown_non_indeed_url(tmp_path: Path, project_root: Path) -> None:
    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    with pytest.raises((UnknownPortalError, UnsupportedPortalFetchError)):
        ingest_hard_link(
            config,
            session,
            "https://careers.totally-unknown-corp.example/jobs/view/123",
            candidate=_candidate(),
        )


def test_closed_indeed_fixture_is_refused(tmp_path: Path, project_root: Path) -> None:
    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    html = """<!DOCTYPE html><html><body>
    <h1 data-testid="jobsearch-JobInfoHeader-title">Editor</h1>
    <div data-testid="inlineHeader-companyName">Diario</div>
    <div id="jobDescriptionText"><p>SEO.</p></div>
    <p>This job has expired on Indeed.</p>
    </body></html>"""
    with pytest.raises(ClosedPostingError, match="(?i)expir"):
        ingest_hard_link(
            config,
            session,
            "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97",
            candidate=_candidate(),
            html=html,
        )
