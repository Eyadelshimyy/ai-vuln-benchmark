"""Standing presence plan: permanent ∪ active companies (issue #43)."""

from __future__ import annotations

from pathlib import Path

from jobbot.companies.models import (
    CareerSite,
    CareerSiteType,
    CompanyRecord,
    KnowledgeStatus,
)
from jobbot.companies.registry import CompanyRegistry
from jobbot.companies.signup import AccountNeed
from jobbot.config import JobbotConfig, PathsConfig
from jobbot.cv.sync import plan_active_companies, plan_sync, rows_to_open
from jobbot.models.candidate import Candidate
from jobbot.portals.detect import AtsKind
from tests.fixtures.profile import sample_profile_dict


def _candidate() -> Candidate:
    return Candidate.model_validate(sample_profile_dict())


def _config(tmp_path: Path, project_root: Path) -> JobbotConfig:
    return JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(templates=project_root / "templates", output=Path("output")),
    )


def _registry() -> CompanyRegistry:
    return CompanyRegistry(
        companies=[
            CompanyRecord(
                id="betterfly",
                name="Betterfly",
                career_sites=[
                    CareerSite(
                        url="https://betterfly.wd3.myworkdayjobs.com/careers",
                        domain="betterfly.wd3.myworkdayjobs.com",
                        site_type=CareerSiteType.ATS_INSTANCE,
                        ats=AtsKind.WORKDAY,
                        status=KnowledgeStatus.ACTIVE,
                    )
                ],
            ),
            CompanyRecord(
                id="buk",
                name="Buk",
                career_sites=[
                    CareerSite(
                        url="https://buk.cl/trabaja-con-nosotros",
                        domain="buk.cl",
                        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
                        ats=AtsKind.UNKNOWN,
                        status=KnowledgeStatus.CANDIDATE,
                    )
                ],
            ),
            CompanyRecord(
                id="rejected-co",
                name="Rejected Co",
                career_sites=[
                    CareerSite(
                        url="https://boards.greenhouse.io/rejected",
                        domain="boards.greenhouse.io",
                        site_type=CareerSiteType.ATS_INSTANCE,
                        ats=AtsKind.GREENHOUSE,
                        status=KnowledgeStatus.REJECTED,
                    )
                ],
            ),
        ]
    )


def test_active_company_appears_candidate_and_rejected_do_not(
    tmp_path: Path, project_root: Path
) -> None:
    rows = plan_active_companies(_config(tmp_path, project_root), registry=_registry())
    ids = {row.company_id for row in rows}
    assert ids == {"betterfly"}
    assert rows[0].action == "needs_account"
    assert rows[0].need is AccountNeed.NEEDED
    assert "companies signup" in rows[0].hint


def test_greenhouse_active_is_update_profile_not_account(
    tmp_path: Path, project_root: Path
) -> None:
    registry = CompanyRegistry(
        companies=[
            CompanyRecord(
                id="acme",
                name="Acme",
                career_sites=[
                    CareerSite(
                        url="https://boards.greenhouse.io/acme",
                        domain="boards.greenhouse.io",
                        site_type=CareerSiteType.ATS_INSTANCE,
                        ats=AtsKind.GREENHOUSE,
                        status=KnowledgeStatus.ACTIVE,
                    )
                ],
            )
        ]
    )
    rows = plan_active_companies(_config(tmp_path, project_root), registry=registry)
    assert len(rows) == 1
    assert rows[0].action == "update_profile"
    assert rows[0].need is AccountNeed.NOT_NEEDED


def test_plan_sync_includes_permanent_and_companies(
    tmp_path: Path, project_root: Path
) -> None:
    (tmp_path / "output").mkdir(exist_ok=True)
    plan = plan_sync(
        _config(tmp_path, project_root),
        _candidate(),
        registry=_registry(),
    )
    assert {p.target.value for p in plan.permanent} >= {"cv", "getonboard", "indeed", "linkedin"}
    assert [r.company_id for r in plan.companies] == ["betterfly"]


def test_signed_in_page_lets_an_account_portal_be_filled(
    tmp_path: Path, project_root: Path
) -> None:
    rows = plan_active_companies(
        _config(tmp_path, project_root),
        registry=_registry(),
        page_urls=("https://betterfly.wd3.myworkdayjobs.com/en-US/External/userHome",),
    )
    assert rows[0].company_id == "betterfly"
    assert rows[0].session_evidenced is True
    assert rows[0].action == "update_profile"
    assert rows_to_open(rows) == rows


def test_public_career_tab_or_login_wall_is_not_a_session(
    tmp_path: Path, project_root: Path
) -> None:
    config = _config(tmp_path, project_root)
    public = plan_active_companies(
        config,
        registry=_registry(),
        page_urls=("https://betterfly.wd3.myworkdayjobs.com/careers",),
    )
    login = plan_active_companies(
        config,
        registry=_registry(),
        page_urls=("https://betterfly.wd3.myworkdayjobs.com/login",),
    )
    assert public[0].session_evidenced is False
    assert public[0].action == "needs_account"
    assert login[0].session_evidenced is False
    assert login[0].action == "needs_account"
    assert rows_to_open(public) == []
    assert rows_to_open(login) == []


def test_unknown_ats_is_not_inferred_from_the_hostname(
    tmp_path: Path, project_root: Path
) -> None:
    registry = CompanyRegistry(
        companies=[
            CompanyRecord(
                id="acme",
                name="Acme",
                career_sites=[
                    CareerSite(
                        url="https://boards.greenhouse.io/acme",
                        domain="boards.greenhouse.io",
                        site_type=CareerSiteType.ATS_INSTANCE,
                        ats=AtsKind.UNKNOWN,
                        status=KnowledgeStatus.ACTIVE,
                    )
                ],
            )
        ]
    )
    rows = plan_active_companies(_config(tmp_path, project_root), registry=registry)
    assert rows[0].ats is AtsKind.UNKNOWN
    assert rows[0].need is AccountNeed.UNKNOWN
    assert rows[0].action == "blocked"
    assert rows_to_open(rows) == []


def test_planning_does_not_promote_or_open_candidates(
    tmp_path: Path, project_root: Path
) -> None:
    registry = _registry()
    rows = plan_active_companies(_config(tmp_path, project_root), registry=registry)
    assert {row.company_id for row in rows} == {"betterfly"}
    assert rows_to_open(rows) == []
    buk = registry.find_company("buk")
    assert buk is not None
    assert buk.career_sites[0].status is KnowledgeStatus.CANDIDATE
    rejected = registry.find_company("rejected-co")
    assert rejected is not None
    assert rejected.career_sites[0].status is KnowledgeStatus.REJECTED
