"""Empleos Públicos (Chile) — board detect, ficha parse, fixture search (#118.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.adapters.empleospublicos.jobs import (
    EmpleosPublicosJobSource,
    EmpleosPublicosParseError,
    canonical_ficha_url,
    job_from_ficha_html,
    load_search_fixture,
)
from jobbot.companies.discovery import classify_url
from jobbot.companies.models import CareerSiteType
from jobbot.config import JobbotConfig, PathsConfig, load_config
from jobbot.db.engine import make_engine, make_session_factory
from jobbot.jobs.from_url import UnsupportedPortalFetchError, ingest_hard_link
from jobbot.jobs.sources import JobSearchQuery, get_job_source
from jobbot.models.candidate import Candidate
from jobbot.portals.detect import JOB_BOARD_KINDS, AtsKind, detect_ats
from jobbot.portals.knowledge import lookup_portal
from tests.fixtures.profile import public_health_profile_dict

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "empleospublicos"
FICHA_URL = (
    "https://www.empleospublicos.cl/pub/convocatorias/avisotrabajoficha.aspx?i=990001"
)


def _config(tmp_path: Path, project_root: Path) -> JobbotConfig:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    return JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(templates=project_root / "templates", output=Path("output")),
    )


def test_empleos_publicos_is_a_job_board() -> None:
    assert detect_ats(FICHA_URL) == AtsKind.EMPLEOS_PUBLICOS
    assert AtsKind.EMPLEOS_PUBLICOS in JOB_BOARD_KINDS
    found = classify_url(FICHA_URL)
    assert found.site_type == CareerSiteType.JOB_BOARD
    assert not found.is_company_specific


def test_canonical_ficha_keeps_id_and_strips_tracking() -> None:
    dirty = FICHA_URL + "&utm_source=linkedin&foo=1"
    assert canonical_ficha_url(dirty) == FICHA_URL


def test_ficha_html_becomes_a_job_posting() -> None:
    html = (FIXTURES / "aviso_ficha.html").read_text(encoding="utf-8")
    job = job_from_ficha_html(html, url=FICHA_URL)

    assert job.title.startswith("Profesional Epidemiólogo")
    assert "Neutro" in job.company
    assert job.ats_kind == AtsKind.EMPLEOS_PUBLICOS.value
    assert job.source_job_id == "990001"
    assert "vigilancia epidemiológica" in (job.description or "").casefold()
    assert any("vigilancia" in r.casefold() for r in job.requirements)


def test_search_fixture_lists_leads() -> None:
    jobs = load_search_fixture(FIXTURES / "search_results.json")
    assert len(jobs) == 3
    assert all(j.ats_kind == AtsKind.EMPLEOS_PUBLICOS.value for j in jobs)
    assert jobs[0].url and "i=990001" in jobs[0].url


def test_search_filters_by_query() -> None:
    from jobbot.jobs.normalization import fold_text

    source = EmpleosPublicosJobSource(fixture=FIXTURES / "search_results.json")
    hits = source.search_jobs(JobSearchQuery(query="epidemiol", limit=10))
    assert len(hits) == 2
    assert all("epidemiol" in fold_text(job.title) for job in hits)


def test_search_without_fixture_explains_fixture_first() -> None:
    source = EmpleosPublicosJobSource(fixture=None)
    with pytest.raises(EmpleosPublicosParseError, match="fixture-first"):
        source.search_jobs(JobSearchQuery(query="salud", limit=5))


def test_get_job_source_empleos_publicos() -> None:
    source = get_job_source(
        "empleos_publicos",
        load_config(),
        fixture=FIXTURES / "search_results.json",
    )
    assert isinstance(source, EmpleosPublicosJobSource)


def test_ingest_hard_link_ficha_fixture(tmp_path: Path, project_root: Path) -> None:
    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    html = (FIXTURES / "aviso_ficha.html").read_text(encoding="utf-8")
    portal = lookup_portal(config, FICHA_URL)
    assert portal.known and portal.ats_kind == AtsKind.EMPLEOS_PUBLICOS

    result = ingest_hard_link(
        config,
        session,
        FICHA_URL,
        candidate=Candidate.model_validate(public_health_profile_dict()),
        html=html,
        build=False,
        prepare=False,
    )
    assert result.job.source == "empleos_publicos"
    assert result.job.source_job_id == "990001"


def test_ingest_without_fixture_is_refused(tmp_path: Path, project_root: Path) -> None:
    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    with pytest.raises(UnsupportedPortalFetchError, match="fixture"):
        ingest_hard_link(
            config,
            session,
            FICHA_URL,
            candidate=Candidate.model_validate(public_health_profile_dict()),
            html=None,
            build=False,
            prepare=False,
        )


def test_workday_international_url_is_known_and_fixture_ingests(
    tmp_path: Path, project_root: Path
) -> None:
    """#118.3 acceptance slice: a Workday tenant URL is consultable via --fixture."""
    workday_url = (
        "https://who.wd103.myworkdayjobs.com/en-US/ExternalStaffJobs/"
        "job/Somewhere/Advisor_R-0001"
    )
    assert detect_ats(workday_url) == AtsKind.WORKDAY
    html = (project_root / "tests/fixtures/jobs/career_phenom_open.html").read_text(
        encoding="utf-8"
    )
    config = _config(tmp_path, project_root)
    session = make_session_factory(make_engine(config.database_path))()
    result = ingest_hard_link(
        config,
        session,
        workday_url,
        candidate=Candidate.model_validate(public_health_profile_dict()),
        html=html,
        build=False,
        prepare=False,
    )
    assert result.portal.known
    assert result.job.title
