"""Offline audit of stored jobs (Indeed decoy keys and title/company dupes)."""

from __future__ import annotations

from dataclasses import dataclass

from jobbot.applications.manager import ApplicationRepository
from jobbot.jobs.indeed_url import indeed_jk_rejection
from jobbot.jobs.repository import JobRepository
from jobbot.models.application import ApplicationStatus

_SUBMITTED = frozenset(
    {
        ApplicationStatus.APPLIED,
        ApplicationStatus.SCREENING,
        ApplicationStatus.INTERVIEW,
        ApplicationStatus.TECHNICAL_INTERVIEW,
        ApplicationStatus.OFFER,
    }
)


@dataclass(frozen=True)
class AuditFinding:
    job_id: str
    reason: str
    protected: bool = False


def audit_indeed_jobs(
    jobs: JobRepository,
    applications: ApplicationRepository,
    *,
    apply: bool = False,
) -> list[AuditFinding]:
    """List (and optionally mark) Indeed rows that look like decoys or duplicates."""
    stored = jobs.list_all(include_invalid=True)
    indeed = [job for job in stored if job.source == "indeed" and not job.invalid_reason]
    by_fingerprint: dict[tuple[str, str, str], str] = {}
    findings: list[AuditFinding] = []

    for job in indeed:
        jk = job.source_job_id or ""
        rejection = indeed_jk_rejection(jk) if jk else "formato"
        if rejection is not None:
            findings.append(
                AuditFinding(
                    job_id=job.id,
                    reason=f"jk inválido ({rejection})",
                    protected=_is_protected(applications, job.id),
                )
            )
            continue
        key = (
            (job.title or "").casefold().strip(),
            (job.company or "").casefold().strip(),
            (job.location or "").casefold().strip(),
        )
        prior = by_fingerprint.get(key)
        if prior is not None:
            findings.append(
                AuditFinding(
                    job_id=job.id,
                    reason=f"duplicado de {prior}",
                    protected=_is_protected(applications, job.id),
                )
            )
            continue
        by_fingerprint[key] = job.id

    if apply:
        for finding in findings:
            if finding.protected:
                continue
            jobs.mark_invalid(finding.job_id, finding.reason)

    return findings


def _is_protected(applications: ApplicationRepository, job_id: str) -> bool:
    app = applications.get_for_job(job_id)
    return app is not None and app.status in _SUBMITTED
