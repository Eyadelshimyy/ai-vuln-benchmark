"""One-shot company portal discovery: candidates only, never canonical truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobbot.companies.models import CareerSiteType, DiscoverySource, KnowledgeStatus
from jobbot.companies.oneshot import (
    CompanySeed,
    FetchResult,
    candidate_urls,
    import_candidates,
    load_candidates,
    load_search_hits,
    load_seeds,
    run_oneshot,
    write_candidates,
)
from jobbot.companies.registry import (
    CompanyRegistry,
    default_companies_path,
    load_companies,
)
from jobbot.portals.detect import AtsKind

FIXTURES = Path("tests/fixtures/companies")

CAREER_PAGE = (
    "<html><head><title>Trabaja con nosotros | Empresa</title></head>"
    "<body><h1>Trabaja con nosotros</h1><p>Vacantes disponibles</p>"
    "<a href='/postular'>Postula aquí</a></body></html>"
)
# 200 OK on an unrelated page: a user profile that happens to be named "empleos".
UNRELATED_PAGE = (
    "<html><head><title>Empleos · GitHub</title></head>"
    "<body><h1>Empleos</h1><p>empleos empleos empleos</p></body></html>"
)


@dataclass
class FakeFetcher:
    """Offline fetcher: exact-URL responses, everything else is a 404."""

    pages: dict[str, FetchResult]
    requested: list[str] = field(default_factory=list)

    def fetch(self, url: str) -> FetchResult:
        self.requested.append(url)
        return self.pages.get(url, FetchResult(url=url, status=404))


def _seeds(project_root: Path) -> list[CompanySeed]:
    return load_seeds(project_root / FIXTURES / "seeds.yaml")


def test_candidate_urls_start_with_official_domain_paths() -> None:
    seed = CompanySeed(name="Empresa", domain="empresa.cl", country="CL")
    urls = candidate_urls(seed)
    assert urls[0] == "https://empresa.cl/careers"
    assert "https://empresa.cl/trabaja-con-nosotros" in urls
    assert "https://careers.empresa.cl" in urls
    assert len(urls) == len(set(urls))


def test_known_portal_url_is_stored_as_hint_and_probed_first() -> None:
    """Once a portal URL is known, save it — do not rediscover via path patterns alone."""
    seed = CompanySeed(
        name="SQM",
        domain="sqm.com",
        country="CL",
        career_url_hints=["https://www.trabajaensqm.com/"],
    )
    urls = candidate_urls(seed)
    assert urls[0] == "https://trabajaensqm.com"
    assert "https://sqm.com/careers" in urls


def test_load_seeds_reads_career_url_hints(tmp_path: Path) -> None:
    path = tmp_path / "seeds.yaml"
    path.write_text(
        "companies:\n"
        "  - name: SQM\n"
        "    domain: sqm.com\n"
        "    career_url_hints:\n"
        "      - https://www.trabajaensqm.com/\n",
        encoding="utf-8",
    )
    seeds = load_seeds(path)
    assert seeds[0].career_url_hints == ["https://www.trabajaensqm.com/"]


def test_rejected_false_positive_stays_rejected_on_reimport() -> None:
    """companies reject is the FP fix; a later oneshot import must not revive it."""
    registry = CompanyRegistry()
    bad = "https://sqm.com/noticia/carreras-e-karts"
    registry.observe(
        company="SQM",
        company_id="sqm",
        url=bad,
        source=DiscoverySource.OFFICIAL_SITE,
        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
        status=KnowledgeStatus.CANDIDATE,
    )
    registry.reject("sqm", url=bad)
    from jobbot.companies.oneshot import CompanyPortalCandidate

    import_candidates(
        registry,
        [
            CompanyPortalCandidate(
                company="SQM",
                company_id="sqm",
                career_url=bad,
                site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
                ats=AtsKind.UNKNOWN,
                source=DiscoverySource.OFFICIAL_SITE,
                status=KnowledgeStatus.CANDIDATE,
            )
        ],
    )
    site = registry.find_company("sqm")
    assert site is not None
    assert site.find_site(bad) is not None
    assert site.find_site(bad).status == KnowledgeStatus.REJECTED


def test_oneshot_produces_candidates_and_reports_gaps(project_root: Path) -> None:
    html = (project_root / FIXTURES / "career_page_greenhouse.html").read_text(encoding="utf-8")
    fetcher = FakeFetcher(
        pages={
            # own corporate portal, ATS proven by the embedded board script
            "https://empresa.cl/trabaja-con-nosotros": FetchResult(
                url="https://empresa.cl/trabaja-con-nosotros", status=200, html=html
            ),
            # corporate career page that redirects to an external ATS
            "https://empresa-redirige.cl/careers": FetchResult(
                url="https://empresa-redirige.wd3.myworkdayjobs.com/en-US/External",
                status=200,
            ),
        }
    )
    report = run_oneshot(_seeds(project_root), fetcher)

    assert report.companies_seen == 3
    assert report.requests_made == len(fetcher.requested)
    assert report.companies_without_portal == ["Empresa Sin Portal"]

    by_company = {c.company_id: c for c in report.candidates}
    own = by_company["empresa-portal-propio"]
    assert own.career_url == "https://empresa.cl/trabaja-con-nosotros"
    assert own.site_type == CareerSiteType.COMPANY_CAREER_PORTAL
    assert own.ats == AtsKind.GREENHOUSE
    assert "html marker" in own.evidence
    assert own.source == DiscoverySource.OFFICIAL_SITE

    redirected = by_company["empresa-redirige-a-ats"]
    assert redirected.career_url == "https://empresa-redirige.wd3.myworkdayjobs.com/en-US/External"
    assert redirected.ats == AtsKind.WORKDAY
    assert redirected.reached_from == "https://empresa-redirige.cl/careers"
    assert "redirect" in redirected.evidence

    assert {c.status for c in report.candidates} == {KnowledgeStatus.CANDIDATE}


def test_http_200_alone_is_not_a_candidate(project_root: Path) -> None:
    """Regression: /empleos answering 200 on a user profile is not a career portal."""
    fetcher = FakeFetcher(
        pages={
            "https://empresa.cl/empleos": FetchResult(
                url="https://empresa.cl/empleos", status=200, html=UNRELATED_PAGE
            ),
            "https://empresa.cl/carreras": FetchResult(
                url="https://empresa.cl/carreras", status=200, html=""
            ),
        }
    )
    report = run_oneshot(_seeds(project_root), fetcher)
    assert report.candidates == []
    assert "Empresa Portal Propio" in report.companies_without_portal


def test_apex_that_refuses_the_connection_is_retried_on_www(project_root: Path) -> None:
    """Smoke test finding: many corporate sites only serve the www host."""
    fetcher = FakeFetcher(
        pages={
            # status 0 = the host closed the connection, so the path proves nothing
            "https://empresa.cl/careers": FetchResult(url="https://empresa.cl/careers", status=0),
            "https://www.empresa.cl/careers": FetchResult(
                url="https://www.empresa.cl/careers", status=200, html=CAREER_PAGE
            ),
        }
    )
    report = run_oneshot(_seeds(project_root), fetcher)

    found = [c for c in report.candidates if c.company_id == "empresa-portal-propio"]
    assert len(found) == 1
    # Stored www-free: the dedup key must not depend on which host answered.
    assert found[0].career_url == "https://empresa.cl/careers"
    assert "Empresa Portal Propio" not in report.companies_without_portal


def test_a_site_that_blocks_us_is_reported_as_blocked_not_as_absent(project_root: Path) -> None:
    """Being refused is not evidence that a company has no career portal."""

    @dataclass
    class BlockingFetcher:
        requested: list[str] = field(default_factory=list)

        def fetch(self, url: str) -> FetchResult:
            self.requested.append(url)
            return FetchResult(url=url, status=403)

    report = run_oneshot(_seeds(project_root)[:1], BlockingFetcher())

    assert report.candidates == []
    assert report.companies_blocked == ["Empresa Portal Propio"]
    assert report.companies_without_portal == []


def test_career_page_evidence_needs_several_concepts() -> None:
    from jobbot.companies.discovery import career_page_evidence

    assert career_page_evidence(UNRELATED_PAGE) == ""
    evidence = career_page_evidence(CAREER_PAGE)
    assert "page title" in evidence
    assert "career vocabulary" in evidence


def test_oneshot_uses_search_hits_after_official_domain(project_root: Path) -> None:
    hits = load_search_hits(project_root / FIXTURES / "search_hits.yaml")
    fetcher = FakeFetcher(
        pages={
            "https://sin-portal.recruitee.com": FetchResult(
                url="https://sin-portal.recruitee.com", status=200
            )
        }
    )
    report = run_oneshot(_seeds(project_root), fetcher, search_hits=hits)
    found = [c for c in report.candidates if c.company_id == "empresa-sin-portal"]
    assert len(found) == 1
    assert found[0].source == DiscoverySource.WEB_DISCOVERY
    assert found[0].ats == AtsKind.RECRUITEE


def test_oneshot_does_not_touch_the_canonical_registry(project_root: Path, tmp_path: Path) -> None:
    from jobbot.config import JobbotConfig

    config = JobbotConfig(root=tmp_path)
    fetcher = FakeFetcher(
        pages={
            "https://empresa.cl/careers": FetchResult(
                url="https://empresa.cl/careers", status=200, html=CAREER_PAGE
            )
        }
    )
    report = run_oneshot(_seeds(project_root), fetcher)
    out = write_candidates(report.candidates, tmp_path / "output/discovery/generated.yaml")
    assert out.is_file()
    assert not default_companies_path(config.root).exists()

    reloaded = load_candidates(out)
    assert [c.career_url for c in reloaded] == [c.career_url for c in report.candidates]
    assert all(c.status == KnowledgeStatus.CANDIDATE for c in reloaded)


def test_import_keeps_candidate_status_and_dedupes(project_root: Path, tmp_path: Path) -> None:
    fetcher = FakeFetcher(
        pages={
            "https://empresa.cl/careers": FetchResult(
                url="https://empresa.cl/careers", status=200, html=CAREER_PAGE
            )
        }
    )
    report = run_oneshot(_seeds(project_root), fetcher)
    path = write_candidates(report.candidates, tmp_path / "generated.yaml")
    candidates = load_candidates(path)

    registry = CompanyRegistry()
    first = import_candidates(registry, candidates)
    assert all(outcome.created_site for outcome in first)
    site = registry.companies[0].career_sites[0]
    assert site.status == KnowledgeStatus.CANDIDATE

    second = import_candidates(registry, candidates)
    assert all(outcome.merged for outcome in second)
    assert len(registry.companies[0].career_sites) == 1


def test_import_can_skip_candidates_on_review(project_root: Path, tmp_path: Path) -> None:
    fetcher = FakeFetcher(
        pages={
            "https://empresa.cl/careers": FetchResult(
                url="https://empresa.cl/careers", status=200, html=CAREER_PAGE
            )
        }
    )
    report = run_oneshot(_seeds(project_root), fetcher)
    write_candidates(report.candidates, tmp_path / "generated.yaml")
    registry = CompanyRegistry()
    outcomes = import_candidates(registry, report.candidates, confirm=lambda _c: False)
    assert outcomes == []
    assert registry.companies == []


def test_saved_registry_reloads_with_provenance(tmp_path: Path) -> None:
    from jobbot.companies.oneshot import CompanyPortalCandidate
    from jobbot.companies.registry import save_companies

    registry = CompanyRegistry()
    import_candidates(
        registry,
        [
            CompanyPortalCandidate(
                company="Empresa",
                company_id="empresa",
                career_url="https://empresa.cl/careers",
                site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
                evidence="HTTP 200 on https://empresa.cl/careers",
            )
        ],
    )
    path = tmp_path / "companies.yaml"
    save_companies(registry, path)
    site = load_companies(path).companies[0].career_sites[0]
    assert site.observations[0].source == DiscoverySource.WEB_DISCOVERY
    assert site.observations[0].evidence.startswith("HTTP 200")


ROBOTS_BLOCK_ALL = "User-agent: *\nDisallow: /\n"
ROBOTS_BLOCK_CAREERS = "User-agent: *\nDisallow: /careers\n"


def test_candidates_are_grouped_by_type_and_by_ats() -> None:
    """A raw list of URLs is not a map; the grouping is what makes it one."""
    from jobbot.companies.oneshot import CompanyPortalCandidate, group_candidates

    candidates = [
        CompanyPortalCandidate(
            company="Una",
            company_id="una",
            career_url="https://una.cl/careers",
            site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
            ats=AtsKind.GREENHOUSE,
        ),
        CompanyPortalCandidate(
            company="Otra",
            company_id="otra",
            career_url="https://otra.cl/trabaja",
            site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
            ats=AtsKind.UNKNOWN,
        ),
        CompanyPortalCandidate(
            company="Tercera",
            company_id="tercera",
            career_url="https://boards.greenhouse.io/tercera",
            site_type=CareerSiteType.ATS_INSTANCE,
            ats=AtsKind.GREENHOUSE,
        ),
    ]

    groups = group_candidates(candidates)

    assert len(groups.by_site_type[CareerSiteType.COMPANY_CAREER_PORTAL.value]) == 2
    assert len(groups.by_site_type[CareerSiteType.ATS_INSTANCE.value]) == 1
    assert len(groups.by_ats[AtsKind.GREENHOUSE.value]) == 2
    # Rows are ordered by size so the biggest group reads first, and are stable.
    rows = groups.rows()
    assert rows[0][2] >= rows[-1][2]
    assert {row[0] for row in rows} == {"site_type", "ats"}
    portals, companies = rows[0][2], rows[0][3]
    assert portals >= companies >= 1


def test_grouping_an_empty_run_is_empty_not_an_error() -> None:
    from jobbot.companies.oneshot import group_candidates

    groups = group_candidates([])
    assert groups.rows() == []


def test_more_public_career_paths_are_probed() -> None:
    seed = CompanySeed(name="Empresa", domain="empresa.cl")
    urls = candidate_urls(seed)

    for path in ("/careers", "/trabaja-con-nosotros", "/vacantes", "/empleo"):
        assert f"https://empresa.cl{path}" in urls
    assert "https://vacantes.empresa.cl" in urls
    # No duplicates survive normalisation.
    assert len(urls) == len(set(urls))


def test_robots_disallow_is_obeyed_and_not_read_as_absence(project_root: Path) -> None:
    """A path we are not allowed to read says nothing about whether it exists."""
    seeds = [CompanySeed(name="Empresa", domain="empresa.cl", country="CL")]
    fetcher = FakeFetcher(
        pages={
            "https://empresa.cl/robots.txt": FetchResult(
                url="https://empresa.cl/robots.txt", status=200, html=ROBOTS_BLOCK_ALL
            ),
            "https://empresa.cl/careers": FetchResult(
                url="https://empresa.cl/careers", status=200, html=CAREER_PAGE
            ),
        }
    )

    report = run_oneshot(seeds, fetcher)

    assert report.candidates == [], "a disallowed page must not become a candidate"
    assert report.companies_robots_skipped == ["Empresa"]
    assert report.companies_without_portal == []
    assert "https://empresa.cl/careers" not in fetcher.requested


def test_robots_allows_what_it_does_not_forbid() -> None:
    seeds = [CompanySeed(name="Empresa", domain="empresa.cl", country="CL")]
    fetcher = FakeFetcher(
        pages={
            "https://empresa.cl/robots.txt": FetchResult(
                url="https://empresa.cl/robots.txt", status=200, html=ROBOTS_BLOCK_CAREERS
            ),
            "https://empresa.cl/trabaja-con-nosotros": FetchResult(
                url="https://empresa.cl/trabaja-con-nosotros", status=200, html=CAREER_PAGE
            ),
        }
    )

    report = run_oneshot(seeds, fetcher)

    assert [c.career_url for c in report.candidates] == ["https://empresa.cl/trabaja-con-nosotros"]
    assert "https://empresa.cl/careers" not in fetcher.requested


def test_a_site_without_robots_is_probed_normally() -> None:
    seeds = [CompanySeed(name="Empresa", domain="empresa.cl")]
    fetcher = FakeFetcher(
        pages={
            "https://empresa.cl/careers": FetchResult(
                url="https://empresa.cl/careers", status=200, html=CAREER_PAGE
            )
        }
    )

    report = run_oneshot(seeds, fetcher)

    assert len(report.candidates) == 1


def test_robots_is_fetched_once_per_host() -> None:
    """Many paths on one host, one reading of its rules."""
    seeds = [CompanySeed(name="Empresa", domain="empresa.cl")]
    fetcher = FakeFetcher(pages={})

    run_oneshot(seeds, fetcher)

    apex = [url for url in fetcher.requested if url == "https://empresa.cl/robots.txt"]
    probed = [url for url in fetcher.requested if url.startswith("https://empresa.cl/")]
    assert len(apex) == 1, f"robots.txt fetched {len(apex)} times"
    assert len(probed) > 2, "several paths on that host were probed"


def test_a_host_that_hides_its_rules_is_a_refusal_not_an_omission() -> None:
    """Being unable to ask is not the same as being told no."""
    from jobbot.companies.oneshot import RobotsPolicy, RobotsVerdict

    @dataclass
    class RefusingFetcher:
        def fetch(self, url: str) -> FetchResult:
            return FetchResult(url=url, status=403)

    policy = RobotsPolicy(RefusingFetcher())

    assert policy.verdict("https://empresa.cl/careers") is RobotsVerdict.HOST_REFUSED
    assert not policy.allows("https://empresa.cl/careers")


def test_real_robots_file_opens_the_career_path_and_closes_the_rest(
    project_root: Path,
) -> None:
    """Read against a corporate robots.txt, not a hand-made one-liner."""
    from jobbot.companies.oneshot import RobotsPolicy

    body = (project_root / FIXTURES / "robots.txt").read_text(encoding="utf-8")

    @dataclass
    class RobotsFetcher:
        def fetch(self, url: str) -> FetchResult:
            return FetchResult(url=url, status=200, html=body)

    policy = RobotsPolicy(RobotsFetcher())

    assert policy.allows("https://empresa.cl/empleos")
    assert policy.allows("https://empresa.cl/careers")
    assert not policy.allows("https://empresa.cl/intranet")
    assert not policy.allows("https://empresa.cl/empleos/postular")


def test_import_skips_reserved_example_domains() -> None:
    from jobbot.companies.oneshot import CompanyPortalCandidate

    registry = CompanyRegistry()
    candidates = [
        CompanyPortalCandidate(
            company="Empresa Demo",
            company_id="empresa-demo",
            career_url="https://careers.example.com/empleos",
            site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
        ),
        CompanyPortalCandidate(
            company="Empresa Real Demo",
            company_id="empresa-real-demo",
            career_url="https://empresa-demo.cl/trabaja-con-nosotros",
            site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
        ),
    ]
    outcomes = import_candidates(registry, candidates)
    assert [o.site.url for o in outcomes if o.site] == [
        "https://empresa-demo.cl/trabaja-con-nosotros"
    ]
    assert registry.find_company("Empresa Demo") is None
