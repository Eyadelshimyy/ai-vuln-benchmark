"""Game-condition LinkedIn feed scrape: real Playwright + LinkedIn-shaped DOM."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jobbot.adapters.linkedin.posts_source import (
    LinkedInPostJobSource,
    collect_jobs_from_feed_page,
)
from jobbot.config import JobbotConfig, PathsConfig
from jobbot.jobs.sources import JobSearchQuery
from jobbot.portals.detect import AtsKind

pytestmark = pytest.mark.integration

playwright = pytest.importorskip("playwright.sync_api")


def test_browser_feed_dom_detects_email_apply_like_live_sweep(
    project_root: Path,
) -> None:
    """
    Same scrape path as search_live after the page is open:
    Playwright reads feed-shared-update-v2 cards → parse_post_blob → email mailto.
    """
    html_path = project_root / "tests/fixtures/linkedin_feed_email.html"
    query = JobSearchQuery(query="data scientist", limit=20)

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            # file:// origin so absolute hrefs resolve like a real page
            page.goto(html_path.as_uri(), wait_until="domcontentloaded")
            jobs = collect_jobs_from_feed_page(
                page, query, resolve_short_links=False
            )
        finally:
            browser.close()

    assert len(jobs) >= 2
    email_jobs = [j for j in jobs if j.ats_kind == "email"]
    assert len(email_jobs) == 1
    job = email_jobs[0]
    assert job.ats_url == "mailto:seleccion@empresa.cl"
    assert "Enviar CV a seleccion@empresa.cl" in job.description
    assert job.description == job.raw_description
    greenhouse = [j for j in jobs if j.ats_kind == "greenhouse"]
    assert greenhouse and "greenhouse" in (greenhouse[0].ats_url or "")
    # Hiking noise must not become a job
    assert all("hiking" not in j.description.casefold() for j in jobs)


def test_browser_feed_dom_splits_a_post_that_lists_several_vacancies(
    project_root: Path,
) -> None:
    """One card, several roles: the live scrape must yield one appliable job per role."""
    html_path = project_root / "tests/fixtures/linkedin_feed_multi_vacancy.html"
    query = JobSearchQuery(query="data scientist", limit=20)

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(html_path.as_uri(), wait_until="domcontentloaded")
            jobs = collect_jobs_from_feed_page(page, query, resolve_short_links=False)
        finally:
            browser.close()

    assert [j.title for j in jobs] == [
        "Lead Agentic AI Consultant",
        "Lead Data Scientist",
        "Senior Computer Vision Engineer",
    ]
    assert {j.company for j in jobs} == {"Northwind Labs"}
    assert len({j.ats_url for j in jobs}) == 3
    assert all("about-us" not in (j.ats_url or "") for j in jobs)


@pytest.mark.skipif(
    os.environ.get("JOBBOT_LIVE_LINKEDIN") != "1",
    reason="Set JOBBOT_LIVE_LINKEDIN=1 for real LinkedIn HITL sweep (opens browser)",
)
def test_live_linkedin_sweep_opens_browser_and_returns_jobs(
    project_root: Path, tmp_path: Path
) -> None:
    """True game conditions: navigates linkedin.com (login/scroll HITL)."""
    config = JobbotConfig(
        root=project_root,
        paths=PathsConfig(output=tmp_path / "output", database=tmp_path / "t.sqlite"),
    )
    source = LinkedInPostJobSource(config)
    jobs = source.search_live(
        JobSearchQuery(query="hiring data scientist", limit=5)
    )
    # After HITL, we at least exercised navigation; jobs depend on what's on screen
    assert isinstance(jobs, list)
    for job in jobs:
        if job.ats_kind == AtsKind.EMAIL.value:
            assert job.ats_url and job.ats_url.startswith("mailto:")
            assert job.description
