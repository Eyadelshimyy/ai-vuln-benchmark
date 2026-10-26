"""Inside recon: ATS markers + form questions from HTML (issue #45)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.companies.models import (
    CareerSite,
    CareerSiteType,
    CompanyRecord,
    KnowledgeStatus,
)
from jobbot.companies.recon import (
    ReconError,
    plan_recon,
    recon_from_html,
    resolve_recon_site,
)
from jobbot.companies.registry import (
    CompanyRegistry,
    default_companies_path,
    load_companies,
    save_companies,
)
from jobbot.config import JobbotConfig
from jobbot.portals.detect import AtsKind
from jobbot.portals.form_learn import default_form_knowledge_path, load_form_knowledge

GH_FIXTURE = Path("tests/fixtures/forms/career_inside_greenhouse.html")
UNKNOWN_FIXTURE = Path("tests/fixtures/forms/career_inside_unknown.html")


def _registry_unknown_active() -> CompanyRegistry:
    return CompanyRegistry(
        companies=[
            CompanyRecord(
                id="acme",
                name="Acme",
                career_sites=[
                    CareerSite(
                        url="https://careers.acme-demo.cl/join",
                        domain="careers.acme-demo.cl",
                        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
                        ats=AtsKind.UNKNOWN,
                        status=KnowledgeStatus.ACTIVE,
                    )
                ],
            )
        ]
    )


def test_greenhouse_embed_sets_ats_with_evidence(tmp_path: Path, project_root: Path) -> None:
    registry = _registry_unknown_active()
    record, site = resolve_recon_site(registry, "acme")
    html = (project_root / GH_FIXTURE).read_text(encoding="utf-8")
    companies_path = default_companies_path(tmp_path)
    forms_path = default_form_knowledge_path(tmp_path)
    (tmp_path / "data").mkdir()

    report = recon_from_html(
        registry,
        record,
        site,
        html,
        apply=True,
        companies_path=companies_path,
        forms_path=forms_path,
    )

    assert report.detected_ats == AtsKind.GREENHOUSE
    assert "greenhouse" in report.evidence.casefold()
    assert report.wrote_companies is True
    assert report.form_field_count >= 3
    assert "When could you start?" in report.field_labels
    saved = load_companies(companies_path)
    assert saved.find_company("acme") is not None
    site_saved = saved.find_company("acme").find_site(site.url)  # type: ignore[union-attr]
    assert site_saved is not None
    assert site_saved.ats == AtsKind.GREENHOUSE
    forms = load_form_knowledge(forms_path)
    assert forms and forms[0].readable
    blob = forms_path.read_text(encoding="utf-8")
    assert "ada@" not in blob.casefold()
    assert "+56" not in blob


def test_page_without_markers_stays_unknown(tmp_path: Path, project_root: Path) -> None:
    registry = _registry_unknown_active()
    record, site = resolve_recon_site(registry, "acme")
    html = (project_root / UNKNOWN_FIXTURE).read_text(encoding="utf-8")
    companies_path = default_companies_path(tmp_path)
    (tmp_path / "data").mkdir()

    report = recon_from_html(
        registry,
        record,
        site,
        html,
        apply=True,
        companies_path=companies_path,
        forms_path=default_form_knowledge_path(tmp_path),
    )

    assert report.detected_ats == AtsKind.UNKNOWN
    assert report.wrote_companies is False
    assert not companies_path.is_file() or load_companies(companies_path).companies == []


def test_dry_run_writes_nothing(tmp_path: Path, project_root: Path) -> None:
    registry = _registry_unknown_active()
    record, site = resolve_recon_site(registry, "acme")
    html = (project_root / GH_FIXTURE).read_text(encoding="utf-8")
    companies_path = default_companies_path(tmp_path)
    forms_path = default_form_knowledge_path(tmp_path)
    (tmp_path / "data").mkdir()

    report = recon_from_html(
        registry,
        record,
        site,
        html,
        apply=False,
        companies_path=companies_path,
        forms_path=forms_path,
    )

    assert report.detected_ats == AtsKind.GREENHOUSE
    assert report.wrote_companies is False
    assert report.wrote_forms is False
    assert not companies_path.is_file()
    assert not forms_path.is_file()


def test_plan_recon_preview_without_html(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    save_companies(_registry_unknown_active(), default_companies_path(tmp_path))
    config = JobbotConfig(root=tmp_path)
    report = plan_recon(config, "acme", html=None, apply=False)
    assert report.detected_ats == AtsKind.UNKNOWN
    assert "fixture" in report.hint.casefold() or "cdp" in report.hint.casefold()


def test_unknown_company_raises(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    save_companies(CompanyRegistry(), default_companies_path(tmp_path))
    with pytest.raises(ReconError, match="Unknown company"):
        plan_recon(JobbotConfig(root=tmp_path), "missing-co")


def test_recon_refuses_to_store_a_reserved_example_domain(
    tmp_path: Path, project_root: Path
) -> None:
    registry = CompanyRegistry(
        companies=[
            CompanyRecord(
                id="acme",
                name="Acme",
                career_sites=[
                    CareerSite(
                        url="https://careers.acme.example/join",
                        domain="careers.acme.example",
                        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
                        ats=AtsKind.UNKNOWN,
                        status=KnowledgeStatus.CANDIDATE,
                    )
                ],
            )
        ]
    )
    record, site = resolve_recon_site(registry, "acme")
    html = (project_root / GH_FIXTURE).read_text(encoding="utf-8")
    companies_path = default_companies_path(tmp_path)

    report = recon_from_html(
        registry, record, site, html, apply=True, companies_path=companies_path
    )

    assert report.detected_ats == AtsKind.GREENHOUSE
    assert report.wrote_companies is False
    assert report.refused is not None and "reserved" in report.refused
    assert not companies_path.exists()
