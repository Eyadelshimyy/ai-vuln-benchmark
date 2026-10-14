"""``/api/resources`` — connection cards for the Resources Console
(docs/ui-plan.md merge 10).

``GET`` returns the three cards cheaply — configuration and masked
credentials only, no network call. Each card's live reachability check is a
separate ``POST /test/*`` action, matching the plan's "test actions" and
letting a slow or dead connection block only the card the operator clicked,
not the page load.

Locations are deploy-time configuration (flags / env); nothing here writes them.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from womblex.ui import dashboard, resources
from womblex.ui.deps import UISettings, get_settings

router = APIRouter(prefix="/api/resources", tags=["resources"])


@router.get("")
def get_resources(settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """The store, ingest, queue and Isaacus cards, cheap enough to load with the screen."""
    return resources.get_resources(settings)


@router.post("/test/store")
def post_test_store(settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """Live reachability: a local dir check, or `RemoteStore.list_dirs("runs")`."""
    return resources.test_store(settings)


@router.post("/test/ingest")
def post_test_ingest(settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """Live reachability of the configured ingest location, if any."""
    return resources.test_ingest(settings)


@router.post("/test/queue")
def post_test_queue(
    stale_after: float = Query(dashboard.DEFAULT_STALE_AFTER, gt=0),
    window_seconds: float = Query(dashboard.DEFAULT_THROUGHPUT_WINDOW, gt=0),
    job_limit: int = Query(200, gt=0, le=1000),
    settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """Connect to the queue and read fleet + queue-depth state in one go."""
    return resources.test_queue(
        settings, stale_after=stale_after, window_seconds=window_seconds, job_limit=job_limit,
    )
