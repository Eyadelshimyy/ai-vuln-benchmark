"""Career-page HTML → JobPosting (Phenom-style markers, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.jobs.career_page import (
    CareerPageParseError,
    ClosedPostingError,
    job_from_career_html,
    looks_like_career_job_html,
)
from jobbot.portals.detect import AtsKind

OPEN = Path("tests/fixtures/jobs/career_phenom_open.html")
FILLED = Path("tests/fixtures/jobs/career_phenom_filled.html")
URL = "https://careers.example-corp.test/global/en/job/abc123/role"


def test_open_phenom_fixture_extracts_fields(project_root: Path) -> None:
    html = (project_root / OPEN).read_text(encoding="utf-8")
    job = job_from_career_html(html, url=URL)
    assert job.title == "Applied Research Analyst"
    assert job.company == "Northwind Labs"
    assert "SQL" in job.description
    assert job.source == "career_page"
    assert job.ats_kind == AtsKind.UNKNOWN.value
    assert job.url == URL
    assert looks_like_career_job_html(html)


def test_hidden_expire_template_does_not_block_ingest(project_root: Path) -> None:
    html = (project_root / OPEN).read_text(encoding="utf-8")
    job = job_from_career_html(html, url=URL)
    assert "filled" not in job.description.casefold()


def test_visible_filled_banner_raises(project_root: Path) -> None:
    html = (project_root / FILLED).read_text(encoding="utf-8")
    with pytest.raises(ClosedPostingError) as exc:
        job_from_career_html(html, url=URL)
    assert "filled" in exc.value.evidence.casefold()
    assert looks_like_career_job_html(html)


def test_empty_html_is_not_a_career_page() -> None:
    with pytest.raises(CareerPageParseError):
        job_from_career_html("<html><body><p>hello</p></body></html>", url=URL)
    assert not looks_like_career_job_html("<html><body></body></html>")
