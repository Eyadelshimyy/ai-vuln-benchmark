"""Login tour: permanent sites plus active and candidate company portals (issue #56)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.browser.sessions import CdpEndpoint, ChromeProcess, SessionState, SessionStatus
from jobbot.companies.models import (
    CareerSite,
    CareerSiteType,
    CompanyRecord,
    KnowledgeStatus,
)
from jobbot.companies.registry import CompanyRegistry
from jobbot.companies.signup import AccountNeed
from jobbot.portals.detect import AtsKind


def _site(
    url: str,
    domain: str,
    ats: AtsKind,
    status: KnowledgeStatus,
    *,
    site_type: CareerSiteType = CareerSiteType.ATS_INSTANCE,
) -> CareerSite:
    return CareerSite(
        url=url,
        domain=domain,
        site_type=site_type,
        ats=ats,
        status=status,
    )


def _registry() -> CompanyRegistry:
    return CompanyRegistry(
        companies=[
            CompanyRecord(
                id="acme",
                name="Acme",
                career_sites=[
                    _site(
                        "https://acme.wd3.myworkdayjobs.com/careers",
                        "acme.wd3.myworkdayjobs.com",
                        AtsKind.WORKDAY,
                        KnowledgeStatus.ACTIVE,
                    ),
                    _site(
                        "https://boards.greenhouse.io/acme",
                        "boards.greenhouse.io",
                        AtsKind.GREENHOUSE,
                        KnowledgeStatus.ACTIVE,
                    ),
                ],
            ),
            CompanyRecord(
                id="northwind",
                name="Northwind",
                career_sites=[
                    _site(
                        "https://northwind.fa.oraclecloud.com/hcmUI/CandidateExperience",
                        "northwind.fa.oraclecloud.com",
                        AtsKind.ORACLE,
                        KnowledgeStatus.CANDIDATE,
                    )
                ],
            ),
            CompanyRecord(
                id="rejected-co",
                name="Rejected Co",
                career_sites=[
                    _site(
                        "https://rejected.wd3.myworkdayjobs.com/careers",
                        "rejected.wd3.myworkdayjobs.com",
                        AtsKind.WORKDAY,
                        KnowledgeStatus.REJECTED,
                    )
                ],
            ),
            CompanyRecord(
                id="stale-co",
                name="Stale Co",
                career_sites=[
                    _site(
                        "https://stale.wd3.myworkdayjobs.com/careers",
                        "stale.wd3.myworkdayjobs.com",
                        AtsKind.WORKDAY,
                        KnowledgeStatus.STALE,
                    )
                ],
            ),
            CompanyRecord(
                id="board-co",
                name="Board Co",
                career_sites=[
                    _site(
                        "https://example.test/jobs",
                        "example.test",
                        AtsKind.UNKNOWN,
                        KnowledgeStatus.ACTIVE,
                        site_type=CareerSiteType.JOB_BOARD,
                    )
                ],
            ),
        ]
    )


def _state(site: str, status: SessionStatus) -> SessionState:
    return SessionState(site=site, status=status, evidence=f"{site} fixture", hint=f"hint-{site}")


def _permanent(
    *,
    indeed: SessionStatus = SessionStatus.READY,
    linkedin: SessionStatus = SessionStatus.READY,
    gmail: SessionStatus = SessionStatus.READY,
    getonboard: SessionStatus = SessionStatus.READY,
) -> list[SessionState]:
    return [
        _state("indeed", indeed),
        _state("linkedin", linkedin),
        _state("gmail", gmail),
        _state("getonboard", getonboard),
    ]


def test_active_and_candidate_accounts_are_listed_and_the_rest_are_not(tmp_path: Path) -> None:
    from jobbot.browser.login_plan import LoginBucket, plan_logins

    plan = plan_logins(
        tmp_path,
        registry=_registry(),
        sessions=_permanent(),
        endpoints=[],
        processes=[],
    )
    by_name = {row.name: row for row in plan.rows}
    urls = [row.url for row in plan.rows]

    assert by_name["Acme"].bucket is LoginBucket.ACTIVE
    assert by_name["Acme"].need is AccountNeed.NEEDED
    assert by_name["Northwind"].bucket is LoginBucket.CANDIDATE
    assert by_name["Northwind"].need is AccountNeed.NEEDED
    assert sum(1 for row in plan.rows if row.name == "Acme") == 1
    assert not any("greenhouse" in url for url in urls)
    assert "Rejected Co" not in by_name
    assert "Stale Co" not in by_name
    assert "Board Co" not in by_name
    names = [row.name for row in plan.rows]
    assert names[:4] == ["indeed", "linkedin", "gmail", "getonboard"]
    assert names.index("Acme") < names.index("Northwind")


def test_permanent_needs_login_is_kept(tmp_path: Path) -> None:
    from jobbot.browser.login_plan import plan_logins

    plan = plan_logins(
        tmp_path,
        registry=CompanyRegistry(),
        sessions=_permanent(linkedin=SessionStatus.NEEDS_LOGIN),
        endpoints=[],
        processes=[],
    )
    linkedin = next(row for row in plan.rows if row.name == "linkedin")
    assert linkedin.status is SessionStatus.NEEDS_LOGIN
    assert linkedin.evidence == "linkedin fixture"
    gap = plan.next_to_open()
    assert gap is not None and gap.name == "linkedin"


def test_a_company_tab_is_never_a_ready_session(tmp_path: Path) -> None:
    from jobbot.browser.login_plan import plan_logins

    endpoints = [
        CdpEndpoint(
            url="http://127.0.0.1:9222",
            port=9222,
            page_urls=("https://acme.wd3.myworkdayjobs.com/en-US/careers/login",),
        )
    ]
    plan = plan_logins(
        tmp_path,
        registry=_registry(),
        sessions=_permanent(),
        endpoints=endpoints,
        processes=[],
    )
    acme = next(row for row in plan.rows if row.name == "Acme")
    assert acme.status is SessionStatus.UNKNOWN
    assert acme.status is not SessionStatus.READY
    assert acme.tab_open is True
    assert "not proven" in acme.evidence.casefold() or "not proven" in acme.evidence
    # Opening the host is not "done": the next unopened portal is still the gap.
    gap = plan.next_to_open()
    assert gap is not None and gap.name == "Northwind"


def test_login_port_is_never_one_held_by_another_chrome(tmp_path: Path) -> None:
    """Regression: every probed port taken made --apply fall back to the first one."""
    from jobbot.browser.login_plan import plan_logins

    endpoints = [
        CdpEndpoint(url=f"http://127.0.0.1:{port}", port=port) for port in (9222, 9223, 9224)
    ]
    plan = plan_logins(
        tmp_path,
        registry=_registry(),
        sessions=_permanent(),
        endpoints=endpoints,
        processes=[],
    )
    assert plan.port not in (9222, 9223, 9224)

    held = [ChromeProcess(pid=9, profile_dir=Path("/elsewhere/other"), cdp_port=9222)]
    plan = plan_logins(
        tmp_path,
        registry=_registry(),
        sessions=_permanent(),
        endpoints=[],
        processes=held,
        ports=(9222, 9223),
    )
    assert plan.port == 9223


def test_foreign_chrome_tab_does_not_count_as_an_open_company_tab(tmp_path: Path) -> None:
    from jobbot.browser.login_plan import plan_logins

    endpoints = [
        CdpEndpoint(
            url="http://127.0.0.1:9222",
            port=9222,
            page_urls=("https://acme.wd3.myworkdayjobs.com/en-US/careers",),
            foreign=True,
        )
    ]
    plan = plan_logins(
        tmp_path,
        registry=_registry(),
        sessions=_permanent(),
        endpoints=endpoints,
        processes=[],
    )
    acme = next(row for row in plan.rows if row.name == "Acme")
    assert acme.tab_open is False
    assert "myworkdayjobs" not in acme.evidence


def test_planner_does_not_launch_chrome_or_probe_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jobbot.browser.login_plan import plan_logins

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("chrome probed")

    monkeypatch.setattr("jobbot.browser.login_plan.list_chrome_processes", boom)
    monkeypatch.setattr("jobbot.browser.login_plan.fetch_local_json", boom)
    monkeypatch.setattr("subprocess.Popen", boom)

    plan_logins(
        tmp_path,
        registry=_registry(),
        sessions=_permanent(),
        endpoints=[],
        processes=[],
    )


def test_planning_does_not_promote_a_candidate(tmp_path: Path) -> None:
    from jobbot.browser.login_plan import plan_logins

    registry = _registry()
    plan_logins(
        tmp_path,
        registry=registry,
        sessions=_permanent(),
        endpoints=[],
        processes=[],
    )
    northwind = next(company for company in registry.companies if company.id == "northwind")
    assert northwind.career_sites[0].status is KnowledgeStatus.CANDIDATE


def test_busy_companies_profile_blocks_that_destination(tmp_path: Path) -> None:
    from jobbot.browser.login_plan import plan_logins

    profile = tmp_path / "browser-data" / "companies"
    profile.mkdir(parents=True)
    processes = [ChromeProcess(pid=4242, profile_dir=profile)]
    plan = plan_logins(
        tmp_path,
        registry=_registry(),
        sessions=_permanent(),
        endpoints=[],
        processes=processes,
    )
    acme = next(row for row in plan.rows if row.name == "Acme")
    assert acme.status is SessionStatus.PROFILE_BUSY
    assert acme.status is not SessionStatus.READY
    gap = plan.next_to_open()
    assert gap is not None and gap.name == "Acme"
    assert "4242" in acme.evidence
