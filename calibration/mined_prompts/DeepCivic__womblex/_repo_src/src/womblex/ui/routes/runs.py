"""``/api/runs`` — the run selector's data source.

Read-only: calls nothing but ``womblex.ui.readers``, which calls nothing but
existing ``store/`` readers. No pipeline logic lives here.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from womblex.store.retention import STAGE_SUFFIXES
from womblex.ui import readers
from womblex.ui.deps import UISettings, get_settings
from womblex.ui.readers import StoreUnreachable

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _unreachable(e: StoreUnreachable) -> HTTPException:
    """Map a store fault to 503 with its cause — not an opaque 500."""
    return HTTPException(status_code=503, detail=f"run store unreachable: {e}")


@router.get("")
def list_runs(settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008 - FastAPI DI idiom
    """All runs visible under the console's configured run source.

    A store that cannot be reached (e.g. ``womblex[cloud]`` / ``s3fs`` not
    installed, or a dead endpoint) surfaces as 503 with the underlying cause,
    not an opaque 500 — the same fault the Resources card's *Test connection*
    names.
    """
    try:
        runs = readers.list_run_summaries(settings)
    except StoreUnreachable as e:
        raise _unreachable(e) from e
    return {"runs": [asdict(r) for r in runs]}


@router.get("/{run_id}/manifest")
def get_manifest(run_id: str, settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """The run's documents table (``MANIFEST_SCHEMA``), one row per document."""
    try:
        rows = readers.get_manifest_rows(settings, run_id)
    except StoreUnreachable as e:
        raise _unreachable(e) from e
    if rows is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return {"run_id": run_id, "documents": rows}


@router.get("/{run_id}/stage-presence/{stage}")
def get_stage_presence(
    run_id: str, stage: str, settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """``source_hash`` values with a `stage` sidecar — the checkpoint switcher's data."""
    if stage not in STAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"unknown stage: {stage}")
    try:
        hashes = readers.get_stage_presence(settings, run_id, stage)
    except StoreUnreachable as e:
        raise _unreachable(e) from e
    if hashes is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return {"run_id": run_id, "stage": stage, "source_hashes": hashes}


@router.get("/{run_id}/audit")
def get_audit(run_id: str, settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """Shard-directory integrity audit — the Corpus Inspector's verify-shards action."""
    try:
        report = readers.get_shard_audit(settings, run_id)
    except StoreUnreachable as e:
        raise _unreachable(e) from e
    if report is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return report


@router.get("/{run_id}/chunks/{source_hash}")
def get_chunk_detail(
    run_id: str, source_hash: str, settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """One document's chunks plus entity/PII/money overlays — the Chunk Inspector's data."""
    try:
        detail = readers.get_chunk_detail(settings, run_id, source_hash)
    except StoreUnreachable as e:
        raise _unreachable(e) from e
    if detail is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return {"run_id": run_id, "source_hash": source_hash, **detail}


@router.get("/{run_id}/logs")
def list_logs(run_id: str, settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """Batch logs published for this run (name, size, modified), newest first.

    An existing run with no logs (one written before workers published them)
    lists empty rather than 404-ing — the frontend renders that as an explained
    empty state, not an error.
    """
    try:
        logs = readers.list_run_logs(settings, run_id)
    except StoreUnreachable as e:
        raise _unreachable(e) from e
    if logs is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return {"run_id": run_id, "logs": logs}


@router.get("/{run_id}/logs/{name}")
def get_log(
    run_id: str,
    name: str,
    download: int = Query(0),
    settings: UISettings = Depends(get_settings),  # noqa: B008
) -> PlainTextResponse:
    """One batch log as ``text/plain`` (``?download=1`` forces a file download).

    A *name* that fails the ``batch-NNNN.log`` pattern and a *name* that passes
    but is not present return the **same 404**, both carrying the ``available``
    list. That is deliberate: the operator gets
    "that log is not here — these are" in one round trip, and a rejected name is
    indistinguishable from an absent one, so the endpoint cannot be used to probe
    for what exists outside the run's ``logs/`` prefix. Containment happens first
    — the pattern check runs before any path join, so a malformed name never
    reaches the filesystem or store.
    """
    try:
        text = readers.read_run_log(settings, run_id, name)
    except StoreUnreachable as e:
        raise _unreachable(e) from e
    if text is None:
        # A run that does not exist at all is a plain 404; a run that exists but
        # lacks this log gets the available-list payload so the picker can
        # self-correct. `list_run_logs` distinguishes the two by returning None.
        try:
            available = readers.list_run_logs(settings, run_id)
        except StoreUnreachable as e:
            raise _unreachable(e) from e
        if available is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        raise HTTPException(
            status_code=404,
            detail={"message": f"log not found: {name}", "available": available},
        )
    headers = (
        {"Content-Disposition": f'attachment; filename="{name}"'} if download else None
    )
    return PlainTextResponse(text, headers=headers)
