"""ATS adapter registry tests."""

from jobbot.adapters.ats.registry import adapter_for_job, adapter_for_kind
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind


def test_adapter_for_kind_greenhouse() -> None:
    adapter = adapter_for_kind(AtsKind.GREENHOUSE)
    assert adapter is not None
    assert adapter.name == "greenhouse"


def test_adapter_for_job_from_ats_url() -> None:
    job = JobPosting(
        id="J1",
        title="DS",
        company="Co",
        ats_url="https://jobs.lever.co/acme/role",
    )
    adapter = adapter_for_job(job)
    assert adapter is not None
    assert adapter.name == "lever"


def test_adapter_for_getonboard() -> None:
    job = JobPosting(
        id="J2",
        title="DS",
        company="Co",
        ats_url="https://www.getonbrd.com/jobs/data-scientist-x",
        ats_kind="getonboard",
    )
    adapter = adapter_for_job(job)
    assert adapter is not None
    assert adapter.name == "getonboard"


def test_adapter_for_workday() -> None:
    job = JobPosting(
        id="J3",
        title="DS",
        company="Co",
        ats_url="https://company.wd1.myworkdayjobs.com/en-US/careers/job/1",
    )
    adapter = adapter_for_job(job)
    assert adapter is not None
    assert adapter.name == "workday"
