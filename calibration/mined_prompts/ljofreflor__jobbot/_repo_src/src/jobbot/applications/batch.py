"""Queue of jobs for assisted apply — one at a time, human submits."""

from __future__ import annotations

from dataclasses import dataclass

from jobbot.models.application import Application, ApplicationStatus
from jobbot.models.job import JobPosting

# Already past assisted open, or terminal — do not reopen in --all.
_SKIP_STATUSES: frozenset[ApplicationStatus] = frozenset(
    {
        ApplicationStatus.APPLIED,
        ApplicationStatus.SCREENING,
        ApplicationStatus.INTERVIEW,
        ApplicationStatus.TECHNICAL_INTERVIEW,
        ApplicationStatus.OFFER,
        ApplicationStatus.REJECTED,
        ApplicationStatus.WITHDRAWN,
    }
)


@dataclass(frozen=True)
class BatchApplyItem:
    """One job in the --all queue (plan-only until the CLI opens it)."""

    job: JobPosting
    status: ApplicationStatus | None

    @property
    def label(self) -> str:
        score = (
            f"{self.job.match_score:.0f}%"
            if self.job.match_score is not None
            else "—"
        )
        st = self.status.value if self.status is not None else "no-app"
        return f"{self.job.id}  {score}  {st}  {self.job.company}  {self.job.title}"


def select_batch_apply_jobs(
    jobs: list[JobPosting],
    applications: list[Application],
    *,
    limit: int | None = None,
) -> list[BatchApplyItem]:
    """Jobs with a portal URL, not already past apply, highest match first."""
    by_job = {app.job_id: app for app in applications}
    items: list[BatchApplyItem] = []
    for job in jobs:
        if not (job.ats_url or job.url):
            continue
        app = by_job.get(job.id)
        if app is not None and app.status in _SKIP_STATUSES:
            continue
        items.append(
            BatchApplyItem(job=job, status=app.status if app is not None else None)
        )
    items.sort(
        key=lambda item: (
            item.job.match_score is not None,
            item.job.match_score or 0.0,
            item.job.id,
        ),
        reverse=True,
    )
    if limit is not None:
        if limit < 0:
            msg = "limit must be >= 0"
            raise ValueError(msg)
        items = items[:limit]
    return items
