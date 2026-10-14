"""Read the shared run store. HTML stays on disk; this returns the rows."""

from typing import Optional

from fastapi import APIRouter

from tracebi.state import install_extra, list_runs

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("")
def get_runs(kind: Optional[str] = None, target: Optional[str] = None,
             limit: int = 50):
    """Newest first. ``kind`` and ``target`` narrow the list. Viewer may read."""
    with install_extra("web"):
        return list_runs(kind=kind or None, target=target or None, limit=limit)
