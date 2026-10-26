"""Learning company topology from URLs seen during normal JobBot use."""

from __future__ import annotations

from pathlib import Path

from jobbot.companies.learn import learn_from_job, learn_from_url
from jobbot.companies.models import CareerSiteType, DiscoverySource, KnowledgeStatus
from jobbot.companies.registry import CompanyRegistry, default_companies_path, load_companies
from jobbot.config import JobbotConfig
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind


def _job(**kwargs: object) -> JobPosting:
    base: dict[str, object] = {
        "id": "J0001",
        "source": "linkedin_post",
        "title": "Senior Data Scientist",
        "company": "Empresa Multi Portal",
    }
    base.update(kwargs)
    return JobPosting.model_validate(base)


def test_posting_url_is_stored_as_the_portal_that_lists_openings(tmp_path: Path) -> None:
    config = JobbotConfig(root=tmp_path)
    result = learn_from_job(
        config,
        _job(
            ats_url="https://boards.greenhouse.io/multiportallabs/jobs/4123456?gh_src=abc",
            ats_kind="greenhouse",
        ),
        source=DiscoverySource.LINKEDIN_POST,
    )
    assert result is not None
    assert result.url == "https://boards.greenhouse.io/multiportallabs"
    assert result.ats == "greenhouse"
    assert result.created

    registry = load_companies(default_companies_path(tmp_path))
    site = registry.companies[0].career_sites[0]
    assert site.status == KnowledgeStatus.CANDIDATE
    assert site.site_type == CareerSiteType.ATS_INSTANCE
    assert site.observations[0].source == DiscoverySource.LINKEDIN_POST


def test_recruiter_names_never_become_companies(tmp_path: Path) -> None:
    """Regression: on post sweeps job.company is the post author, not the employer."""
    config = JobbotConfig(root=tmp_path)
    result = learn_from_job(
        config,
        _job(
            company="Carol HR",
            source="linkedin_post",
            ats_url="https://jobs.ashbyhq.com/corp/role-99",
            ats_kind="ashby",
        ),
        source=DiscoverySource.LINKEDIN_POST,
    )
    assert result is not None
    assert result.company_id == "corp"
    stored = default_companies_path(tmp_path).read_text(encoding="utf-8")
    assert "Carol" not in stored


def test_employer_field_is_kept_for_real_job_sources(tmp_path: Path) -> None:
    config = JobbotConfig(root=tmp_path)
    result = learn_from_job(
        config,
        _job(
            company="Empresa Retail",
            source="indeed",
            ats_url="https://empresa-retail.cl/trabaja-con-nosotros",
        ),
        source=DiscoverySource.JOB_SOURCE,
    )
    assert result is not None
    assert result.company_id == "empresa-retail"


def test_email_and_board_routes_are_not_company_knowledge(tmp_path: Path) -> None:
    config = JobbotConfig(root=tmp_path)
    assert learn_from_job(config, _job(ats_url="mailto:reclutamiento@example.com")) is None
    assert learn_from_job(config, _job(ats_url="https://www.getonbrd.com/jobs/ds-empresa")) is None
    assert learn_from_job(config, _job(url="https://www.linkedin.com/jobs/view/1")) is None
    assert not default_companies_path(tmp_path).exists()


def test_second_sighting_merges_instead_of_duplicating(tmp_path: Path) -> None:
    config = JobbotConfig(root=tmp_path)
    job = _job(ats_url="https://boards.greenhouse.io/multiportallabs/jobs/1", ats_kind="greenhouse")
    learn_from_job(config, job, source=DiscoverySource.LINKEDIN_POST)
    second = learn_from_job(config, job, source=DiscoverySource.JOB_SOURCE)
    assert second is not None and second.merged and not second.created

    registry = load_companies(default_companies_path(tmp_path))
    assert len(registry.companies[0].career_sites) == 1
    assert registry.companies[0].career_sites[0].confidence == 2


def test_learn_from_url_keeps_corporate_portal_type() -> None:
    registry = CompanyRegistry()
    outcome = learn_from_url(
        registry,
        company="BCI",
        url="https://trabajaenbci.cl/ofertas/",
        source=DiscoverySource.USER_OBSERVATION,
        country="CL",
    )
    assert outcome is not None and outcome.site is not None
    assert outcome.site.site_type == CareerSiteType.COMPANY_CAREER_PORTAL
    assert outcome.site.ats == AtsKind.UNKNOWN
    assert outcome.company.country == "CL"
    assert "trabajaenbci.cl" in outcome.company.domains


def test_opaque_job_url_is_stored_as_the_career_section(tmp_path: Path) -> None:
    config = JobbotConfig(root=tmp_path)
    result = learn_from_job(
        config,
        _job(
            company="Empresa Ejemplo",
            source="manual",
            url="https://careers.empresa-ejemplo.cl/global/en/job/fe3426f378a7100/Data-Scientist-Senior",
            ats_url="https://careers.empresa-ejemplo.cl/global/en/job/fe3426f378a7100/Data-Scientist-Senior",
        ),
    )
    assert result is not None
    assert result.url == "https://careers.empresa-ejemplo.cl/global/en"
    assert result.ats == "unknown"
    assert "fe3426f378a7100" not in result.url


def test_job_source_posting_is_stored_as_the_listing(tmp_path: Path) -> None:
    """Regression: the job_source path stored /vacancies/<slug-id> as the company site."""
    config = JobbotConfig(root=tmp_path)
    result = learn_from_job(
        config,
        _job(
            company="Empresa Demo",
            source="manual",
            url="https://career.empresa-demo.cl/en-us/vacancies/lead-data-analyst-89203",
        ),
    )
    assert result is not None
    assert result.url == "https://career.empresa-demo.cl/en-us/vacancies"


def test_aggregator_tenant_is_not_learned_as_a_company(tmp_path: Path) -> None:
    config = JobbotConfig(root=tmp_path)
    for url in (
        "https://jobs.lever.co/jobgether/0f1e2d3c-4b5a-6978-8899-aabbccddeeff",
        "https://remoteyeah.com/jobs/senior-analyst-at-empresa-123",
    ):
        assert learn_from_job(config, _job(company="Empresa Demo", url=url)) is None
    assert load_companies(default_companies_path(tmp_path)).companies == []


def test_manatal_host_is_never_a_company_domain() -> None:
    registry = CompanyRegistry()
    outcome = learn_from_url(
        registry,
        company="Empresa Demo",
        url="https://careers-page.com/empresa-demo/job/R123",
        source=DiscoverySource.JOB_SOURCE,
    )
    assert outcome is not None and outcome.site is not None
    assert outcome.site.ats == AtsKind.MANATAL
    assert outcome.site.url == "https://careers-page.com/empresa-demo"
    assert outcome.company.domains == []


def test_reserved_example_domain_is_never_learned(tmp_path: Path) -> None:
    """Regression: fixture URLs on example.com landed in the real company base."""
    config = JobbotConfig(root=tmp_path)
    result = learn_from_job(
        config,
        _job(
            company="Empresa Demo",
            source="manual",
            url="https://careers.example.com/global/en/job/fe3426f378a7100",
        ),
    )
    assert result is None
    assert load_companies(default_companies_path(tmp_path)).companies == []
