"""Batch assisted apply: queue selection, never auto-submit."""

from __future__ import annotations

from jobbot.applications.batch import BatchApplyItem, select_batch_apply_jobs
from jobbot.models.application import Application, ApplicationStatus
from jobbot.models.job import JobPosting


def _job(
    job_id: str,
    *,
    score: float | None = None,
    url: str | None = "https://boards.example.com/jobs/1",
) -> JobPosting:
    return JobPosting(
        id=job_id,
        title=f"Role {job_id}",
        company="Acme",
        url=url,
        ats_url=url,
        match_score=score,
    )


def _app(job_id: str, status: ApplicationStatus) -> Application:
    return Application(id="A1", job_id=job_id, status=status)


def test_select_orders_by_match_and_skips_applied() -> None:
    jobs = [
        _job("J0002", score=40.0),
        _job("J0001", score=90.0),
        _job("J0003", score=70.0),
        _job("J0004", score=99.0, url=None),
    ]
    jobs[3].ats_url = None
    apps = [
        _app("J0003", ApplicationStatus.APPLIED),
        _app("J0002", ApplicationStatus.PREPARED),
    ]
    items = select_batch_apply_jobs(jobs, apps)
    assert [i.job.id for i in items] == ["J0001", "J0002"]
    assert items[0].status is None
    assert items[1].status is ApplicationStatus.PREPARED


def test_limit_caps_the_queue() -> None:
    jobs = [_job("J0001", score=10.0), _job("J0002", score=20.0)]
    items = select_batch_apply_jobs(jobs, [], limit=1)
    assert len(items) == 1
    assert items[0].job.id == "J0002"


def test_batch_item_label_is_readable() -> None:
    item = BatchApplyItem(job=_job("J0001", score=33.3), status=ApplicationStatus.PREPARED)
    assert "J0001" in item.label
    assert "33%" in item.label
    assert "prepared" in item.label
