"""BCI career portal (trabajaenbci.cl): seed hint + Cornerstone evidence (#250)."""

from __future__ import annotations

from pathlib import Path

import yaml

from jobbot.companies.discovery import classify_url
from jobbot.companies.learn import learn_from_url
from jobbot.companies.models import CareerSiteType, DiscoverySource
from jobbot.companies.oneshot import load_seeds
from jobbot.companies.registry import CompanyRegistry
from jobbot.companies.urls import company_hint_from_url
from jobbot.portals.detect import JOB_BOARD_KINDS, AtsKind, detect_ats, detect_ats_in_html
from jobbot.workspace import repo_root

PORTAL = "https://trabajaenbci.cl/"
OFERTAS = "https://trabajaenbci.cl/ofertas/"
CSOD = "https://bci.csod.com/ux/ats/careersite/1/home?c=bci"
WORKDAY_HOST = "https://bci.wd3.myworkdayjobs.com/en-US/External"


def test_seed_lists_bci_career_hint() -> None:
    seeds = {s.name: s for s in load_seeds(repo_root() / "data" / "companies-cl.example.yaml")}
    bci = seeds["BCI"]
    assert PORTAL.rstrip("/") in {u.rstrip("/") for u in bci.career_url_hints}
    raw = yaml.safe_load((repo_root() / "data" / "companies-cl.example.yaml").read_text())
    row = next(c for c in raw["companies"] if c["name"] == "BCI")
    hints = row.get("career_url_hints") or []
    assert any(h.rstrip("/") == PORTAL.rstrip("/") for h in hints)


def test_trabajaenbci_is_company_portal_not_workday() -> None:
    found = classify_url(OFERTAS)
    assert found.is_company_specific
    assert found.site_type == CareerSiteType.COMPANY_CAREER_PORTAL
    assert found.ats == AtsKind.UNKNOWN
    assert detect_ats(OFERTAS) == AtsKind.UNKNOWN
    assert detect_ats(WORKDAY_HOST) == AtsKind.WORKDAY


def test_learn_stores_trabajaenbci_without_inventing_workday() -> None:
    registry = CompanyRegistry()
    outcome = learn_from_url(
        registry,
        company="BCI",
        url=PORTAL,
        source=DiscoverySource.USER_OBSERVATION,
        country="CL",
    )
    assert outcome is not None and outcome.site is not None
    assert "trabajaenbci.cl" in outcome.site.url
    assert outcome.site.ats == AtsKind.UNKNOWN
    assert outcome.site.site_type == CareerSiteType.COMPANY_CAREER_PORTAL
    assert "trabajaenbci.cl" in outcome.company.domains


def test_csod_host_is_cornerstone() -> None:
    assert detect_ats(CSOD) == AtsKind.CORNERSTONE
    assert AtsKind.CORNERSTONE not in JOB_BOARD_KINDS
    assert company_hint_from_url(CSOD) == "bci"


def test_bci_portal_fixture_marks_cornerstone_from_csp(project_root: Path) -> None:
    html = (project_root / "tests/fixtures/bci_trabajaen_portal.html").read_text(encoding="utf-8")
    kind, evidence = detect_ats_in_html(html)
    assert kind == AtsKind.CORNERSTONE
    assert "csod" in evidence.casefold()
    found = classify_url(OFERTAS, html=html)
    assert found.is_company_specific
    assert found.site_type == CareerSiteType.COMPANY_CAREER_PORTAL
    assert found.ats == AtsKind.CORNERSTONE
    assert "html marker" in found.evidence
    assert found.ats != AtsKind.WORKDAY
