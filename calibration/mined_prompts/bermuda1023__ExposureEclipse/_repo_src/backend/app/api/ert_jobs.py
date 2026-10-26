"""ERT background-job endpoints (API_SPEC.md §ERT Job APIs).

Thin router → :class:`~app.services.jobs.JobRegistry`. Per BACKGROUND_JOBS_SPEC.md:

* `POST /run` returns ``202 Accepted`` with the jobId immediately.
* `GET /status/{jobId}` reports the current state (and the technical-error
  block once the job has failed). A failed job is *not* an HTTP error — the
  failure rides in the body (ERROR_HANDLING.md §Domain outcomes).
* `POST /{jobId}/cancel` is optional in v1.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..models.enums import ErrorCode, JobStatus
from ..models.jobs import (
    ErtJobAcceptedResponse,
    ErtJobCancelResponse,
    ErtJobRunRequest,
    ErtJobStatusResponse,
)
from ..services.jobs import JobRegistry, get_registry

router = APIRouter(prefix="/ert-jobs", tags=["ert-jobs"])


@router.post("/run", status_code=202, response_model=ErtJobAcceptedResponse)
async def run(
    payload: ErtJobRunRequest,
    registry: JobRegistry = Depends(get_registry),
) -> ErtJobAcceptedResponse:
    """Start (or re-attach to) an ERT generation job."""
    job_id = registry.submit(payload)
    record = registry.get(job_id)
    status = record.status if record else JobStatus.QUEUED
    return ErtJobAcceptedResponse(job_id=job_id, status=status)


@router.get("/status/{job_id}", response_model=ErtJobStatusResponse)
def status(
    job_id: str,
    registry: JobRegistry = Depends(get_registry),
) -> ErtJobStatusResponse:
    """Poll status for a job."""
    record = registry.get(job_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": ErrorCode.JOB_NOT_FOUND.value,
                "message": f"Job '{job_id}' was not found.",
                "details": {"jobId": job_id},
            },
        )
    return record.to_response()


@router.post("/{job_id}/cancel", response_model=ErtJobCancelResponse)
def cancel(
    job_id: str,
    registry: JobRegistry = Depends(get_registry),
) -> ErtJobCancelResponse:
    """Cancel a queued or running job (no-op if already terminal)."""
    record = registry.get(job_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": ErrorCode.JOB_NOT_FOUND.value,
                "message": f"Job '{job_id}' was not found.",
                "details": {"jobId": job_id},
            },
        )
    registry.cancel(job_id)
    return ErtJobCancelResponse(job_id=job_id, status=record.status)


__all__ = ["router"]
