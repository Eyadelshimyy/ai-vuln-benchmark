"""``/api/dashboard`` — queue state and per-stage progress.

Read-only, like every other console route: it calls
:mod:`womblex.ui.dashboard`, which reads the job queue and the checkpoints
the pipeline already writes. Nothing here requeues, cancels or claims — the
dashboard names a stalled job, a worker recovers it.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from womblex.ui import dashboard
from womblex.ui.deps import UISettings, get_settings

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("")
def get_dashboard(
    run_id: str | None = None,
    stale_after: float = Query(dashboard.DEFAULT_STALE_AFTER, gt=0),
    window_seconds: float = Query(dashboard.DEFAULT_THROUGHPUT_WINDOW, gt=0),
    job_limit: int = Query(200, gt=0, le=1000),
    settings: UISettings = Depends(get_settings),  # noqa: B008 - FastAPI DI idiom
) -> dict:
    """Everything the Dashboard screen renders, for the whole queue or one run.

    Without ``run_id`` the queue views span every run and ``stages`` is
    empty — checkpoints are per-run artefacts, so there is no cross-run
    answer to give. A missing run is not a 404: an enqueued run whose first
    batch has not landed yet has no directory, and the queue counts are
    exactly what the operator is watching for at that moment.
    """
    return dashboard.get_dashboard(
        settings, run_id=run_id, stale_after=stale_after,
        window_seconds=window_seconds, job_limit=job_limit,
    )
