"""SQLite job repository tests."""

from __future__ import annotations

from pathlib import Path

from jobbot.db.engine import make_engine, make_session_factory
from jobbot.jobs.repository import JobRepository


def test_add_and_get_job(tmp_path: Path) -> None:
    engine = make_engine(tmp_path / "test.sqlite")
    session = make_session_factory(engine)()
    repo = JobRepository(session)
    text = Path("tests/fixtures/jobs/senior_ds_retail.txt").read_text(encoding="utf-8")
    job = repo.add_from_text(text)
    assert job.id == "J0001"
    loaded = repo.get("J0001")
    assert loaded is not None
    assert loaded.company == "Sodimac"
    job2 = repo.add_from_text(text)
    assert job2.id == "J0002"


def test_posted_at_survives_the_roundtrip(tmp_path: Path) -> None:
    """The sweep learns when a post was published; the DB must keep it."""
    from datetime import UTC, datetime

    from jobbot.models.job import JobPosting

    engine = make_engine(tmp_path / "test.sqlite")
    session = make_session_factory(engine)()
    repo = JobRepository(session)
    posted = datetime(2025, 11, 5, 13, 1, 10, tzinfo=UTC)
    job = JobPosting(
        id="PENDING",
        title="Data Scientist Senior",
        company="NTT Data",
        description="Buscamos Data Scientist Senior",
        posted_at=posted,
    )
    job.source_job_id = "abc123"
    stored = repo.upsert_external(job)

    loaded = repo.get(stored.id)
    assert loaded is not None
    assert loaded.posted_at is not None
    assert loaded.posted_at.astimezone(UTC) == posted


def test_posted_at_column_is_added_to_an_existing_database(tmp_path: Path) -> None:
    """Regression: older SQLite files have no posted_at; make_engine must migrate."""
    from sqlalchemy import text

    db_path = tmp_path / "legacy.sqlite"
    engine = make_engine(db_path)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE jobs DROP COLUMN posted_at"))
    engine.dispose()

    engine = make_engine(db_path)
    with engine.begin() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(jobs)"))}
    assert "posted_at" in columns
