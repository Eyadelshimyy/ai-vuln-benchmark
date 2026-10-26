"""Organic learning must fire from the discovery commands, not only from manual `learn`."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.companies.registry import default_companies_path, load_companies
from jobbot.exit_codes import SUCCESS
from jobbot.jobs.sources import JobSearchQuery
from jobbot.models.job import JobPosting


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


class FakeIndeedJobSource:
    """Returns one posting that applies on the company's own career portal."""

    base = "cl.indeed.com"

    def __init__(self, _config: object, *, cdp_url: str | None = None) -> None:
        self.cdp_url = cdp_url

    def _jobs(self) -> list[JobPosting]:
        return [
            JobPosting.model_validate(
                {
                    "id": "J0001",
                    "source": "indeed",
                    "source_job_id": "abc123",
                    "title": "Senior Data Scientist",
                    "company": "Empresa Retail",
                    "url": "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97",
                    "ats_url": "https://empresa-retail.cl/trabaja-con-nosotros",
                }
            )
        ]

    def search_cards(self, _query: JobSearchQuery) -> list[JobPosting]:
        return self._jobs()

    def search_jobs(self, _query: JobSearchQuery) -> list[JobPosting]:
        return self._jobs()


def test_jobs_search_feeds_the_company_registry(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`jobs search` stored postings without ever recording who hires where."""
    _workspace(tmp_path, project_root, monkeypatch)

    import jobbot.adapters.indeed.jobs as indeed_jobs
    from jobbot.cli import run_cli

    monkeypatch.setattr(indeed_jobs, "IndeedJobSource", FakeIndeedJobSource)

    assert run_cli(["jobs", "search", "data scientist"], standalone_mode=False) == SUCCESS

    registry = load_companies(default_companies_path(tmp_path))
    sites = {site.url for record in registry.companies for site in record.career_sites}
    assert "https://empresa-retail.cl/trabaja-con-nosotros" in sites


def test_board_only_results_do_not_create_a_registry(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An Indeed-hosted posting is a job board: no company knowledge to store."""
    _workspace(tmp_path, project_root, monkeypatch)

    import jobbot.adapters.indeed.jobs as indeed_jobs
    from jobbot.cli import run_cli

    class BoardOnly(FakeIndeedJobSource):
        def _jobs(self) -> list[JobPosting]:
            jobs = super()._jobs()
            jobs[0].ats_url = None
            return jobs

    monkeypatch.setattr(indeed_jobs, "IndeedJobSource", BoardOnly)

    assert run_cli(["jobs", "search", "data scientist"], standalone_mode=False) == SUCCESS
    assert not default_companies_path(tmp_path).exists()
