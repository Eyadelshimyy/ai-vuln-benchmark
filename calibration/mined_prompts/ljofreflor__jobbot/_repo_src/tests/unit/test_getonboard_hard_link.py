"""Get on Board hard-link HTML → JobPosting (offline fixture)."""

from __future__ import annotations

from pathlib import Path

from jobbot.adapters.getonboard.jobs import (
    job_from_hard_link,
    parse_job_html,
    slug_from_url,
)
from jobbot.portals.detect import AtsKind

FIXTURE = Path("tests/fixtures/jobs/getonboard_applied_scientist.html")
URL = (
    "https://www.getonbrd.com/empleos/data-science-analytics/"
    "applied-scientist-neuralworks-santiago-e3c8"
)


def test_slug_from_empleos_and_jobs_paths() -> None:
    assert slug_from_url(URL) == "applied-scientist-neuralworks-santiago-e3c8"
    assert (
        slug_from_url(
            "https://www.getonbrd.com/jobs/data-science-analytics/"
            "applied-scientist-neuralworks-santiago-e3c8"
        )
        == "applied-scientist-neuralworks-santiago-e3c8"
    )
    assert slug_from_url("https://www.getonbrd.com/companies/neuralworks") is None


def test_parse_job_html_reads_title_company_and_sections() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    job = parse_job_html(html, url=URL)

    assert job.source == "getonboard"
    assert job.source_job_id == "applied-scientist-neuralworks-santiago-e3c8"
    assert job.title == "Applied Scientist"
    assert job.company == "NeuralWorks"
    assert job.ats_kind == AtsKind.GETONBOARD.value
    assert job.ats_url == URL
    assert "Calificaciones clave" in job.description
    assert "Applied Scientist" in job.description
    assert job.skills
    assert job.remote_type == "hybrid"
    assert job.location and "Santiago" in job.location


def test_job_from_hard_link_uses_fixture_without_network() -> None:
    html = FIXTURE.read_text(encoding="utf-8")

    def _boom(_url: str) -> str:
        raise AssertionError("must not fetch when html= is given")

    job = job_from_hard_link(URL, html=html, fetch=_boom)
    assert job.company == "NeuralWorks"
