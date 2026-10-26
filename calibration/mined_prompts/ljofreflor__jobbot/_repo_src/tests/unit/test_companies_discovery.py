"""URL classification for company career platforms (evidence, never appearance)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.companies.discovery import career_root_url, classify_url
from jobbot.companies.models import CareerSiteType
from jobbot.companies.urls import (
    PrivateRouteRejected,
    canonical_key,
    company_hint_from_url,
    public_url,
)
from jobbot.portals.detect import AtsKind, detect_ats, detect_ats_in_html


def test_public_url_drops_query_fragment_and_www() -> None:
    assert public_url("HTTPS://WWW.Empresa.cl/careers/?utm_source=x#top") == (
        "https://empresa.cl/careers"
    )
    assert canonical_key("https://empresa.cl/careers/") == canonical_key(
        "http://www.empresa.cl/careers?token=abc"
    )


def test_public_url_rejects_email_routes() -> None:
    with pytest.raises(PrivateRouteRejected):
        public_url("mailto:recruiter@example.com")


def test_corporate_career_path_stays_ats_unknown() -> None:
    """No ATS marker means unknown — JobBot never guesses by looks."""
    found = classify_url("https://empresa.cl/trabaja-con-nosotros")
    assert found.site_type == CareerSiteType.COMPANY_CAREER_PORTAL
    assert found.ats == AtsKind.UNKNOWN
    assert "career naming pattern" in found.evidence
    assert found.is_company_specific


def test_ats_instance_from_host_rule() -> None:
    found = classify_url("https://empresa.wd3.myworkdayjobs.com/en-US/External")
    assert found.site_type == CareerSiteType.ATS_INSTANCE
    assert found.ats == AtsKind.WORKDAY
    assert "host rule" in found.evidence


def test_new_ats_vendors_are_detected() -> None:
    assert detect_ats("https://empresa.teamtailor.com/jobs") == AtsKind.TEAMTAILOR
    assert detect_ats("https://apply.workable.com/empresa/") == AtsKind.WORKABLE
    assert detect_ats("https://empresa.recruitee.com/o/data-scientist") == AtsKind.RECRUITEE
    assert (
        detect_ats("https://career5.successfactors.eu/careers?company=empresa")
        == AtsKind.SUCCESSFACTORS
    )
    assert detect_ats("https://empresa.taleo.net/careersection/x") == AtsKind.ORACLE


def test_posting_url_collapses_to_portal_root() -> None:
    found = classify_url("https://boards.greenhouse.io/acme/jobs/4123456")
    assert found.site_type == CareerSiteType.JOB_POSTING
    assert found.ats == AtsKind.GREENHOUSE
    assert career_root_url(found.url, found.ats) == "https://boards.greenhouse.io/acme"


def test_workday_root_drops_locale_segment() -> None:
    url = "https://empresa.wd3.myworkdayjobs.com/en-US/External/job/Santiago/Data-Scientist_R-1"
    assert career_root_url(url, AtsKind.WORKDAY) == (
        "https://empresa.wd3.myworkdayjobs.com/External"
    )


def test_corporate_posting_collapses_to_parent_path() -> None:
    url = "https://empresa.cl/trabaja-con-nosotros/vacantes/data-scientist-senior"
    assert career_root_url(url) == "https://empresa.cl/trabaja-con-nosotros/vacantes"


def test_job_id_plus_title_slug_is_not_stored_as_the_portal() -> None:
    """Regression: dropping only the last segment left /job/<id>, still the vacancy."""
    url = "https://careers.example.com/global/en/job/fe3426f378a7100/Data-Scientist-Senior"
    root = career_root_url(url)
    assert root == "https://careers.example.com/global/en"
    assert "fe3426f378a7100" not in root
    assert not root.rstrip("/").endswith("/job")


def test_redirect_to_external_ats_is_recorded() -> None:
    """LinkedIn post → corporate page → Workday: keep both ends of the chain."""

    def resolver(url: str) -> str:
        return "https://empresa.wd3.myworkdayjobs.com/en-US/External"

    found = classify_url("https://empresa.cl/careers", resolve=True, resolver=resolver)
    assert found.ats == AtsKind.WORKDAY
    assert found.requested_url == "https://empresa.cl/careers"
    assert found.redirects_to == "https://empresa.wd3.myworkdayjobs.com/en-US/External"
    assert "redirect" in found.evidence
    assert career_root_url(found.url, found.ats) == (
        "https://empresa.wd3.myworkdayjobs.com/External"
    )


def test_html_marker_is_technical_evidence(project_root: Path) -> None:
    html = (project_root / "tests/fixtures/companies/career_page_greenhouse.html").read_text(
        encoding="utf-8"
    )
    kind, evidence = detect_ats_in_html(html)
    assert kind == AtsKind.GREENHOUSE
    assert "embed" in evidence
    found = classify_url("https://empresa.cl/trabaja-con-nosotros", html=html)
    assert found.ats == AtsKind.GREENHOUSE
    assert "html marker" in found.evidence


def test_job_boards_never_identify_a_company() -> None:
    for url in (
        "https://www.linkedin.com/jobs/view/123",
        "https://cl.indeed.com/viewjob?jk=abc",
        "https://www.getonbrd.com/jobs/data-scientist-empresa",
    ):
        found = classify_url(url)
        assert found.site_type == CareerSiteType.JOB_BOARD
        assert not found.is_company_specific


def test_company_hint_strips_career_prefixes() -> None:
    assert company_hint_from_url("https://trabajaenbci.cl") == "bci"
    assert company_hint_from_url("https://careers.empresa.cl/jobs") == "empresa"


def test_company_hint_on_ats_uses_tenant_not_vendor() -> None:
    """The employer is the board/tenant, never the ATS vendor."""
    assert company_hint_from_url("https://boards.greenhouse.io/acme/jobs/1") == "acme"
    assert company_hint_from_url("https://acme.wd3.myworkdayjobs.com/en-US/External") == "acme"
    assert company_hint_from_url("https://jobs.lever.co/acme/abc") == "acme"
    assert company_hint_from_url("https://acme.teamtailor.com/jobs") == "acme"


def test_workable_root_keeps_the_company_slug() -> None:
    """Regression: apply.workable.com/<company>/j/<id> collapsed to the bare vendor host."""
    url = "https://apply.workable.com/empresa-demo/j/AB12CD34EF/"
    found = classify_url(url)
    assert found.ats == AtsKind.WORKABLE
    assert career_root_url(found.url, found.ats) == "https://apply.workable.com/empresa-demo"
    other = career_root_url("https://apply.workable.com/otra-demo/j/ZX98/", AtsKind.WORKABLE)
    assert canonical_key(other) != canonical_key(career_root_url(url, AtsKind.WORKABLE))
    assert (
        career_root_url("https://empresa-demo.workable.com/jobs/123456", AtsKind.WORKABLE)
        == "https://empresa-demo.workable.com"
    )


def test_vacancy_slug_with_id_collapses_to_the_listing() -> None:
    """Regression: /vacancies/<title>-<id> was stored as the company's career portal."""
    url = "https://career.empresa-demo.cl/en-us/vacancies/lead-data-analyst-89203"
    found = classify_url(url)
    assert found.site_type == CareerSiteType.JOB_POSTING
    root = career_root_url(found.url, found.ats)
    assert root == "https://career.empresa-demo.cl/en-us/vacancies"
    assert "89203" not in root
    assert career_root_url("https://empresa-demo.cl/vacancy/4521") == "https://empresa-demo.cl"


def test_vacancies_listing_is_not_a_posting() -> None:
    found = classify_url("https://career.empresa-demo.cl/en-us/vacancies")
    assert found.site_type == CareerSiteType.COMPANY_CAREER_PORTAL
    assert career_root_url(found.url) == "https://career.empresa-demo.cl/en-us/vacancies"


@pytest.mark.parametrize(
    "url",
    [
        "https://remoteyeah.com/jobs",
        "https://remoteyeah.com/jobs/senior-analyst-at-empresa-123",
        "https://jobgether.com/offer/abc123-senior-analyst",
        "https://jobs.lever.co/jobgether/0f1e2d3c-4b5a-6978-8899-aabbccddeeff",
        "https://vacantes.com/es/vacantes/back-end-engineer-senior-llm-y-agentic-ai-witi-5d091ba5",
        "https://vacantes.com/es/",
    ],
)
def test_job_boards_and_aggregators_are_not_companies(url: str) -> None:
    """Regression: a remote job board and an aggregator's ATS tenant became companies."""
    found = classify_url(url)
    assert found.site_type == CareerSiteType.JOB_BOARD
    assert not found.is_company_specific


def test_manatal_careers_page_is_an_ats_host() -> None:
    """Regression: careers-page.com is Manatal's shared host, not a company domain."""
    for url in (
        "https://careers-page.com/empresa-demo",
        "https://empresa-demo.careers-page.com/jobs",
    ):
        assert detect_ats(url) == AtsKind.MANATAL
        found = classify_url(url)
        assert found.site_type == CareerSiteType.ATS_INSTANCE
    assert (
        career_root_url("https://careers-page.com/empresa-demo/job/R123", AtsKind.MANATAL)
        == "https://careers-page.com/empresa-demo"
    )
    assert (
        career_root_url("https://empresa-demo.careers-page.com/jobs/R123", AtsKind.MANATAL)
        == "https://empresa-demo.careers-page.com"
    )
    kind, evidence = detect_ats_in_html(
        '<a href="https://careers-page.com/empresa-demo">Ver vacantes</a>'
    )
    assert kind == AtsKind.MANATAL and "careers-page.com/empresa-demo" in evidence


def test_ashby_marker_keeps_a_dotted_slug() -> None:
    """Regression: jobs.ashbyhq.com/empresa.io was read as the slug 'empresa'."""
    kind, evidence = detect_ats_in_html(
        '<a href="https://jobs.ashbyhq.com/empresa.io/jobs">Open roles</a>.'
    )
    assert kind == AtsKind.ASHBY
    assert "jobs.ashbyhq.com/empresa.io" in evidence
    kind, evidence = detect_ats_in_html("Apply at jobs.ashbyhq.com/empresa.")
    assert "jobs.ashbyhq.com/empresa)" in evidence


@pytest.mark.parametrize(
    ("host", "reserved"),
    [
        ("example.com", True),
        ("careers.example.com", True),
        ("example.org", True),
        ("jobs.example.net", True),
        ("careers.acme.example", True),
        ("empresa.test", True),
        ("empresa.invalid", True),
        ("localhost", True),
        ("app.localhost", True),
        ("empresa.cl", False),
        ("myexample.com", False),
        ("example.com.ar", False),
    ],
)
def test_reserved_example_hosts(host: str, reserved: bool) -> None:
    from jobbot.companies.urls import is_reserved_host

    assert is_reserved_host(host) is reserved


def test_display_url_drops_only_the_scheme() -> None:
    from jobbot.companies.urls import display_url

    assert display_url("https://careers-page.com/empresa-demo") == "careers-page.com/empresa-demo"
    assert display_url("http://empresa.cl") == "empresa.cl"
