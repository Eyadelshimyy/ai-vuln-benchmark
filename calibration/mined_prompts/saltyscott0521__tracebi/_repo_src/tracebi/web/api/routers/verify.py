"""Offline file verification — the ``tracebi verify --file`` check over HTTP.

A reviewer with a report ``.html`` and its ``.manifest.json`` receipt can
confirm — with no model, no warehouse, and no account — that the numbers
embedded in the file are exactly what was fingerprinted at build. This is the
same rehash-and-compare :func:`tracebi.verify.verify_file` runs on the CLI,
served here so the browser UI can do it too. The data is posted to this box
and never leaves it; nothing is persisted.
"""

import json

from fastapi import APIRouter, Body, HTTPException

from tracebi.verify import verify_file
from tracebi.web.api.errors import error_detail as _error_detail

router = APIRouter(prefix="/verify", tags=["verify"])


@router.post("/file")
def verify_uploaded_file(payload: dict = Body(...)):
    """Rehash the data embedded in an uploaded report ``.html`` against its
    manifest receipt. Body: ``{"html": "<full html>", "manifest": {...}}`` —
    ``manifest`` may be the parsed object or its raw JSON text. Returns the
    same ``verdict`` / ``verdict_detail`` / ``ok`` / ``figures`` shape the CLI
    reports; the HTTP status is 200 whenever the check *ran* (the verdict —
    INTACT or ALTERED — lives in the body), 4xx only for a malformed request.
    """
    html = payload.get("html")
    manifest = payload.get("manifest")

    if not isinstance(html, str) or not html.strip():
        raise HTTPException(
            status_code=400,
            detail="Provide the report HTML as 'html'.",
        )

    if isinstance(manifest, str):
        try:
            manifest = json.loads(manifest)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=_error_detail("The manifest is not valid JSON", exc),
            )
    if not isinstance(manifest, dict):
        raise HTTPException(
            status_code=400,
            detail="Provide the .manifest.json contents as 'manifest'.",
        )

    try:
        return verify_file(html, manifest)
    except Exception as exc:  # noqa: BLE001 — surfaced structured to the UI
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Verification failed", exc),
        )
