"""Company ↔ career platform registry: dedup, provenance, candidate → promote."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from jobbot.companies.models import CareerSiteType, DiscoverySource, KnowledgeStatus
from jobbot.companies.registry import (
    CompanyRegistry,
    active_career_sites,
    load_companies,
    save_companies,
    shareable_payload,
)
from jobbot.companies.urls import PrivateRouteRejected
from jobbot.portals.detect import AtsKind


def _registry_with_portal() -> CompanyRegistry:
    registry = CompanyRegistry()
    registry.observe(
        company="Empresa Multi Portal",
        url="https://trabajaenmultiportal.cl",
        source=DiscoverySource.USER_OBSERVATION,
        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
        country="CL",
        evidence="career naming pattern: host label 'trabajaenmultiportal'",
    )
    return registry


def test_company_holds_many_portals() -> None:
    registry = _registry_with_portal()
    registry.observe(
        company="Empresa Multi Portal",
        url="https://multiportal.wd1.myworkdayjobs.com/Careers",
        source=DiscoverySource.LINKEDIN_POST,
        site_type=CareerSiteType.ATS_INSTANCE,
        ats=AtsKind.WORKDAY,
        evidence="host rule: workday",
    )
    registry.observe(
        company="Empresa Multi Portal",
        url="https://boards.greenhouse.io/multiportallabs",
        source=DiscoverySource.WEB_DISCOVERY,
        site_type=CareerSiteType.ATS_INSTANCE,
        ats=AtsKind.GREENHOUSE,
        evidence="host rule: greenhouse",
    )
    assert len(registry.companies) == 1
    record = registry.companies[0]
    assert len(record.career_sites) == 3
    assert {site.ats for site in record.career_sites} == {
        AtsKind.UNKNOWN,
        AtsKind.WORKDAY,
        AtsKind.GREENHOUSE,
    }
    # The ATS host is shared infrastructure, not a corporate domain.
    assert "boards.greenhouse.io" not in record.domains
    assert "trabajaenmultiportal.cl" in record.domains


def test_equivalent_urls_merge_and_raise_confidence() -> None:
    registry = _registry_with_portal()
    outcome = registry.observe(
        company="Empresa Multi Portal",
        url="http://www.trabajaenmultiportal.cl/?utm_source=linkedin",
        source=DiscoverySource.LINKEDIN_POST,
        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
        evidence="seen again in a recruiter post",
    )
    site = registry.companies[0].career_sites[0]
    assert outcome.merged and not outcome.created_site
    assert len(registry.companies[0].career_sites) == 1
    assert len(site.observations) == 2
    assert site.confidence == 2


def test_same_source_twice_does_not_inflate_confidence() -> None:
    registry = _registry_with_portal()
    registry.observe(
        company="Empresa Multi Portal",
        url="https://trabajaenmultiportal.cl",
        source=DiscoverySource.USER_OBSERVATION,
        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
    )
    site = registry.companies[0].career_sites[0]
    assert len(site.observations) == 2
    assert site.confidence == 1


def test_email_routes_are_never_company_knowledge() -> None:
    registry = CompanyRegistry()
    with pytest.raises(PrivateRouteRejected):
        registry.observe(
            company="Empresa",
            url="mailto:reclutamiento@example.com",
            source=DiscoverySource.LINKEDIN_POST,
        )


def test_promote_is_the_only_path_to_active() -> None:
    registry = _registry_with_portal()
    site = registry.companies[0].career_sites[0]
    assert site.status == KnowledgeStatus.CANDIDATE
    assert site.last_verified is None
    promoted = registry.promote("Empresa Multi Portal", url="https://trabajaenmultiportal.cl")
    assert [s.url for s in promoted] == ["https://trabajaenmultiportal.cl"]
    assert site.status == KnowledgeStatus.ACTIVE
    assert site.last_verified is not None


def test_contradiction_keeps_both_observations_and_flags_stale() -> None:
    """A portal can change ATS; the stored fact is not overwritten silently."""
    registry = CompanyRegistry()
    registry.observe(
        company="Empresa",
        url="https://empresa.cl/jobs",
        source=DiscoverySource.OFFICIAL_SITE,
        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
        ats=AtsKind.GREENHOUSE,
        evidence="html marker: greenhouse embed script",
        status=KnowledgeStatus.ACTIVE,
    )
    outcome = registry.observe(
        company="Empresa",
        url="https://empresa.cl/jobs",
        source=DiscoverySource.WEB_DISCOVERY,
        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
        ats=AtsKind.WORKDAY,
        evidence="html marker: workday tenant host",
    )
    site = registry.companies[0].career_sites[0]
    assert outcome.conflict is not None
    assert site.ats == AtsKind.GREENHOUSE
    assert site.status == KnowledgeStatus.STALE
    assert len(site.observations) == 2
    assert any("ats changed" in conflict for conflict in site.conflicts)


def test_same_url_under_another_company_is_not_duplicated() -> None:
    registry = _registry_with_portal()
    outcome = registry.observe(
        company="Otra Empresa",
        url="https://trabajaenmultiportal.cl",
        source=DiscoverySource.WEB_DISCOVERY,
        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
    )
    assert outcome.site is None
    assert outcome.conflict is not None and "already registered" in outcome.conflict
    assert not registry.companies[1].career_sites


def test_export_shares_only_promoted_public_facts() -> None:
    registry = _registry_with_portal()
    registry.observe(
        company="Empresa Multi Portal",
        url="https://boards.greenhouse.io/multiportallabs",
        source=DiscoverySource.WEB_DISCOVERY,
        site_type=CareerSiteType.ATS_INSTANCE,
        ats=AtsKind.GREENHOUSE,
        notes="local note about my own application",
    )
    registry.promote("Empresa Multi Portal", url="https://trabajaenmultiportal.cl")
    payload = shareable_payload(registry)
    companies = payload["companies"]
    assert isinstance(companies, list) and len(companies) == 1
    sites = companies[0]["career_sites"]
    assert [site["url"] for site in sites] == ["https://trabajaenmultiportal.cl"]
    assert "notes" not in sites[0]
    assert sites[0]["evidence"][0]["source"] == "user_observation"


def test_reusable_sites_exclude_boards_and_unverified() -> None:
    registry = _registry_with_portal()
    registry.observe(
        company="Empresa Multi Portal",
        url="https://www.getonbrd.com/companies/multiportal",
        source=DiscoverySource.JOB_SOURCE,
        site_type=CareerSiteType.JOB_BOARD,
        ats=AtsKind.GETONBOARD,
    )
    assert active_career_sites(registry) == []
    registry.promote("Empresa Multi Portal")
    reusable = active_career_sites(registry)
    assert [site.url for _record, site in reusable] == ["https://trabajaenmultiportal.cl"]


def test_registry_roundtrip(tmp_path: Path) -> None:
    registry = _registry_with_portal()
    registry.promote("Empresa Multi Portal", now=datetime(2026, 9, 16, tzinfo=UTC))
    path = tmp_path / "data" / "companies.yaml"
    save_companies(registry, path)
    loaded = load_companies(path)
    site = loaded.companies[0].career_sites[0]
    assert site.status == KnowledgeStatus.ACTIVE
    assert site.last_verified == datetime(2026, 9, 16, tzinfo=UTC)
    assert site.observations[0].source == DiscoverySource.USER_OBSERVATION


def test_shipped_example_covers_three_architectures(project_root: Path) -> None:
    registry = load_companies(project_root / "data" / "companies.example.yaml")
    by_id = {record.id: record for record in registry.companies}
    own = by_id["portal-propio"].career_sites[0]
    assert own.site_type == CareerSiteType.COMPANY_CAREER_PORTAL
    assert own.ats == AtsKind.UNKNOWN

    redirected = by_id["redirige-a-ats"].career_sites[0]
    assert redirected.ats == AtsKind.WORKDAY
    assert redirected.reached_from == "https://empresa-redirige.cl/careers"

    multi = by_id["multi-portal"].career_sites
    assert len(multi) == 3
    assert {site.status for site in multi} == {KnowledgeStatus.ACTIVE, KnowledgeStatus.CANDIDATE}


@pytest.mark.parametrize(
    "url",
    [
        "https://careers.example.com/global/en",
        "https://example.org/jobs",
        "https://careers.acme.example/join",
        "https://empresa.test/careers",
        "https://empresa.invalid/careers",
        "http://localhost:8000/careers",
    ],
)
def test_reserved_example_domains_are_refused(url: str) -> None:
    """Regression: nothing stopped fixture domains from reaching the real base."""
    from jobbot.companies.urls import ReservedDomainRejected

    registry = CompanyRegistry()
    with pytest.raises(ReservedDomainRejected, match="reserved"):
        registry.observe(
            company="Empresa Demo",
            url=url,
            source=DiscoverySource.USER_OBSERVATION,
            site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
        )
    assert registry.companies == []
