"""Backfill posted_at for jobs swept before JobBot knew how to date a post."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jobbot.db.engine import make_engine, make_session_factory
from jobbot.jobs.backfill import backfill_posted_at
from jobbot.jobs.repository import JobRepository
from jobbot.models.job import JobPosting

J0048_URL = (
    "https://es.linkedin.com/posts/dcarrenom_nttdata-datascience-machinelearning-"
    "activity-7391867181325733888-y7P9"
)


def _repo(tmp_path: Path) -> JobRepository:
    engine = make_engine(tmp_path / "test.sqlite")
    return JobRepository(make_session_factory(engine)())


def _job(source_id: str, url: str | None, posted_at: datetime | None = None) -> JobPosting:
    return JobPosting(
        id="PENDING",
        source="linkedin_post",
        source_job_id=source_id,
        url=url,
        title="Data Scientist",
        company="Someone",
        description="Buscamos Data Scientist",
        posted_at=posted_at,
    )


def test_backfill_dates_rows_that_carry_an_activity_id(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    stale = repo.upsert_external(_job("post-with-activity", J0048_URL))
    profile_only = repo.upsert_external(
        _job("post-without-link", "https://www.linkedin.com/in/rodrigo-fernandez-4272437/")
    )

    assert backfill_posted_at(repo) == 1

    dated = repo.get(stale.id)
    assert dated is not None
    assert dated.posted_at is not None
    assert dated.posted_at.astimezone(UTC).date() == datetime(2025, 11, 5, tzinfo=UTC).date()

    undated = repo.get(profile_only.id)
    assert undated is not None
    assert undated.posted_at is None


def test_backfill_leaves_known_dates_alone(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    known = datetime(2026, 1, 2, tzinfo=UTC)
    job = repo.upsert_external(_job("already-dated", J0048_URL, posted_at=known))

    assert backfill_posted_at(repo) == 0

    loaded = repo.get(job.id)
    assert loaded is not None
    assert loaded.posted_at is not None
    assert loaded.posted_at.astimezone(UTC) == known
