"""Offline Indeed job audit (#211)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from jobbot.applications.manager import ApplicationRepository
from jobbot.cli import app
from jobbot.db.engine import make_engine, make_session_factory
from jobbot.jobs.audit import audit_indeed_jobs
from jobbot.jobs.repository import JobRepository
from jobbot.models.application import ApplicationStatus
from jobbot.models.job import JobPosting


def _repos(tmp_path: Path) -> tuple[JobRepository, ApplicationRepository]:
    engine = make_engine(tmp_path / "t.sqlite")
    session = make_session_factory(engine)()
    return JobRepository(session), ApplicationRepository(session)


def _indeed(
    job_id: str,
    jk: str,
    *,
    title: str = "Senior Data Scientist",
    company: str = "Acme Analytics",
    location: str = "Santiago",
) -> JobPosting:
    return JobPosting(
        id=job_id,
        source="indeed",
        source_job_id=jk,
        url=f"https://cl.indeed.com/viewjob?jk={jk}",
        title=title,
        company=company,
        location=location,
        discovered_at=datetime.now(UTC),
    )


def test_audit_dry_run_lists_decoy_without_writing(tmp_path: Path) -> None:
    jobs, apps = _repos(tmp_path)
    jobs.save(_indeed("J0039", "123456789abcdef0"))
    jobs.save(_indeed("J0040", "8a5fab1a7c476a97"))

    findings = audit_indeed_jobs(jobs, apps, apply=False)
    assert any(f.job_id == "J0039" and "jk inválido" in f.reason for f in findings)
    assert jobs.get("J0039") is not None
    assert jobs.get("J0039").invalid_reason is None


def test_audit_apply_marks_decoy_and_hides_from_list(tmp_path: Path) -> None:
    jobs, apps = _repos(tmp_path)
    jobs.save(_indeed("J0039", "123456789abcdef0"))
    jobs.save(_indeed("J0040", "8a5fab1a7c476a97"))

    findings = audit_indeed_jobs(jobs, apps, apply=True)
    assert any(f.job_id == "J0039" for f in findings)
    assert jobs.get("J0039").invalid_reason is not None
    active_ids = {job.id for job in jobs.list_all()}
    assert active_ids == {"J0040"}


def test_audit_apply_skips_submitted(tmp_path: Path) -> None:
    jobs, apps = _repos(tmp_path)
    jobs.save(_indeed("J0039", "123456789abcdef0"))
    apps.upsert_for_job("J0039", status=ApplicationStatus.APPLIED)

    findings = audit_indeed_jobs(jobs, apps, apply=True)
    assert findings[0].protected is True
    assert jobs.get("J0039").invalid_reason is None


def test_audit_reports_title_company_duplicate(tmp_path: Path) -> None:
    jobs, apps = _repos(tmp_path)
    jobs.save(_indeed("J0040", "8a5fab1a7c476a97"))
    jobs.save(_indeed("J0041", "25db4649b81bfb9d"))  # same title/company/location

    findings = audit_indeed_jobs(jobs, apps, apply=False)
    assert any(f.job_id == "J0041" and "duplicado de J0040" in f.reason for f in findings)


def test_cli_unknown_source_is_validation_failure() -> None:
    runner = CliRunner()
    outcome = runner.invoke(app, ["jobs", "audit", "--source", "linkedin"])
    assert outcome.exit_code == 2
    assert "Unknown audit source" in outcome.output
