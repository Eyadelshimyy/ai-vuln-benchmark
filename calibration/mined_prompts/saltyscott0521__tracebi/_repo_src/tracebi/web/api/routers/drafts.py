"""Drafts over HTTP: the person's view of what an agent (or they) are writing.

Reading is for the draft's owner or an admin; starting, writing, publishing and
deleting also need ``analyst`` (the unlisted-write default in
``auth._required_role``). With no authorization configured every principal is
``admin``, as everywhere else.
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from tracebi import drafts
from tracebi.audit import get_actor
from tracebi.web.api.errors import error_detail

router = APIRouter(prefix="/drafts", tags=["drafts"])


class PublishBody(BaseModel):
    note: str = ""


class StartBody(BaseModel):
    kind: Literal["reports", "models"]
    path: str
    from_published: bool = True


class FileBody(BaseModel):
    content: str


def _who() -> "tuple[Optional[str], bool]":
    """(acting user, may see everyone's drafts)."""
    user, role = get_actor()
    return user, (role or "admin") == "admin"


def _allow(owner: str) -> None:
    user, admin = _who()
    if not admin and drafts.slug_owner(user) != owner:
        raise HTTPException(status_code=403, detail={
            "message": "That draft belongs to someone else.",
            "role": get_actor()[1], "required_role": "admin"})


def _refused(exc: drafts.DraftError) -> HTTPException:
    status = 404 if isinstance(exc, drafts.DraftNotFound) else 422
    return HTTPException(status_code=status, detail=error_detail("Draft refused", exc))


@router.get("")
def list_drafts():
    user, admin = _who()
    found = drafts.list_drafts(None if admin else drafts.slug_owner(user))
    return {"drafts": [
        {**d, "url": drafts.draft_url(d["owner"], d["kind"], d["path"])}
        for d in found]}


@router.post("")
def start_draft(body: StartBody, response: Response):
    """Start the acting user's draft of a published report or model. One they
    already have is returned as it is (200), so "edit this" is safe to repeat;
    a new one is 201. A Python model or a report that runs code is refused
    (422), and the message says to edit it from a laptop."""
    owner = drafts.slug_owner(_who()[0])
    try:
        have = next((d for d in drafts.list_drafts(owner)
                     if (d["kind"], d["path"]) == (body.kind, body.path)), None)
        draft = have or drafts.start_draft(owner, body.kind, body.path, body.from_published)
    except drafts.DraftError as exc:
        raise _refused(exc)
    response.status_code = 200 if have else 201
    return {**draft, "url": drafts.draft_url(owner, body.kind, body.path),
            "created": have is None}


@router.get("/{owner}/{kind}/{path:path}/version")
def draft_version(owner: str, kind: str, path: str):
    _allow(owner)
    try:
        return {"version": drafts.draft_version(owner, kind, path)}
    except drafts.DraftError as exc:
        raise _refused(exc)


@router.get("/{owner}/{kind}/{path:path}/files")
def draft_files(owner: str, kind: str, path: str):
    _allow(owner)
    try:
        return drafts.read_draft(owner, kind, path)
    except drafts.DraftError as exc:
        raise _refused(exc)


@router.put("/{owner}/{kind}/{path:path}/files/{file}")
def write_draft_file(owner: str, kind: str, path: str, file: str, body: FileBody):
    """Write one of the draft's files. The answer carries the draft's new
    version, so a page can follow it without waiting for its next poll."""
    _allow(owner)
    try:
        written = drafts.write_draft_file(owner, kind, path, file, body.content)
        return {**written, "version": drafts.draft_version(owner, kind, path)}
    except drafts.DraftError as exc:
        raise _refused(exc)


@router.get("/{owner}/{kind}/{path:path}/preview", response_class=HTMLResponse)
def draft_preview(owner: str, kind: str, path: str):
    """The draft as it is right now, rendered in memory (exploration blocks
    kept). Not a build: no file is written, no receipt minted."""
    from tracebi.web.api.routers.desk import _loaded_models

    _allow(owner)
    try:
        page = drafts.render_preview(owner, kind, path,
                                     drafts.preview_models(owner, _loaded_models()))
    except drafts.DraftError as exc:
        raise _refused(exc)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 — a broken package is shown, not a 500
        raise HTTPException(status_code=422, detail=error_detail(
            "The report package failed to render", exc))
    if page is None:
        raise HTTPException(status_code=422, detail={
            "message": "A model draft has no page to preview.",
            "exception_type": "DraftError", "traceback": ""})
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


@router.post("/{owner}/{kind}/{path:path}/publish")
def publish_draft(owner: str, kind: str, path: str, body: Optional[PublishBody] = None):
    from tracebi.web.api.routers.desk import _loaded_models

    _allow(owner)
    try:
        return drafts.publish_draft(
            owner, kind, path, _who()[0], (body.note if body else "") or None,
            _loaded_models())
    except drafts.DraftError as exc:
        raise _refused(exc)


@router.delete("/{owner}/{kind}/{path:path}")
def delete_draft(owner: str, kind: str, path: str):
    _allow(owner)
    try:
        if not drafts.delete_draft(owner, kind, path):
            raise drafts.DraftNotFound(f"no draft {kind}/{path} for {owner}")
    except drafts.DraftError as exc:
        raise _refused(exc)
    return {"deleted": True}
