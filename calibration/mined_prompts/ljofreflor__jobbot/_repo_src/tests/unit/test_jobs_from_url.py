"""Unit tests for ingest_hard_link orchestration (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.config import JobbotConfig, PathsConfig
from jobbot.db.engine import make_engine, make_session_factory
from jobbot.jobs.from_url import (
    UnknownPortalError,
    UnsupportedPortalFetchError,
    ingest_hard_link,
)
from jobbot.models.candidate import Candidate
from jobbot.portals.detect import AtsKind
from tests.fixtures.profile import nurse_profile_dict

FIXTURE = Path("tests/fixtures/jobs/getonboard_applied_scientist.html")
URL = (
    "https://www.getonbrd.com/empleos/data-science-analytics/"
    "applied-scientist-neuralworks-santiago-e3c8"
)


def _config(tmp_path: Path, project_root: Path) -> JobbotConfig:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    return JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(templates=project_root / "templates", output=Path("output")),
    )


def test_ingest_unknown_portal_raises(tmp_path: Path, project_root: Path) -> None:
    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    candidate = Candidate.model_validate(nurse_profile_dict())
    with pytest.raises(UnknownPortalError):
        ingest_hard_link(
            config,
            session,
            "https://careers.totally-unknown-corp.example/jobs/1",
            candidate=candidate,
        )


def test_ingest_gob_fixture_builds_package(tmp_path: Path, project_root: Path) -> None:
    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    candidate = Candidate.model_validate(
        __import__("yaml").safe_load(
            (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8")
        )
    )
    html = (project_root / FIXTURE).read_text(encoding="utf-8")
    result = ingest_hard_link(
        config, session, URL, candidate=candidate, html=html
    )
    assert result.job.id.startswith("J")
    assert result.job.ats_kind == AtsKind.GETONBOARD.value
    assert result.package_dir is not None
    assert result.package_dir.is_dir()
    assert result.match is not None


def test_ingest_greenhouse_known_but_no_fetcher(
    tmp_path: Path, project_root: Path
) -> None:
    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    candidate = Candidate.model_validate(nurse_profile_dict())
    with pytest.raises(UnsupportedPortalFetchError):
        ingest_hard_link(
            config,
            session,
            "https://boards.greenhouse.io/acme/jobs/1",
            candidate=candidate,
        )


def test_ingest_career_fixture_on_unknown_host(
    tmp_path: Path, project_root: Path
) -> None:
    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    candidate = Candidate.model_validate(nurse_profile_dict())
    html = (
        project_root / "tests/fixtures/jobs/career_phenom_open.html"
    ).read_text(encoding="utf-8")
    url = "https://careers.example-corp.test/global/en/job/abc123/role"
    result = ingest_hard_link(
        config, session, url, candidate=candidate, html=html
    )
    assert result.job.company == "Northwind Labs"
    assert result.job.title == "Applied Research Analyst"
    assert result.job.source == "career_page"
    assert result.job.ats_kind == AtsKind.UNKNOWN.value
    assert result.package_dir is not None
    assert not result.portal.known


def test_ingest_filled_career_fixture_raises(
    tmp_path: Path, project_root: Path
) -> None:
    from jobbot.jobs.from_url import ClosedPostingError

    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    candidate = Candidate.model_validate(nurse_profile_dict())
    html = (
        project_root / "tests/fixtures/jobs/career_phenom_filled.html"
    ).read_text(encoding="utf-8")
    with pytest.raises(ClosedPostingError):
        ingest_hard_link(
            config,
            session,
            "https://careers.example-corp.test/job/1",
            candidate=candidate,
            html=html,
        )
