"""FastAPI app factory for the `/v1` service API (`docs/service-api.md`).

Separate from the console: it binds one shared store and queue for its
lifetime, authenticates every `/v1` route with a service token
(:mod:`womblex.api.auth`), and scopes runs to the caller that submitted them —
a run another client owns is a 404, and ``admin`` sees all. A run is known by
the queue, so the CLI- and console-submitted runs (no owner) are admin-only.

Uploads (``POST /v1/uploads``) store documents under the caller's own
``<client_id>/<upload_id>/`` folder of the ingest location, which is the
``input_prefix`` a submission then names.

Submission writes queue rows only, through the console's dispatch
(:mod:`womblex.cloud.dispatch`): the run id is minted here, a non-admin
caller's ``input_prefix`` must sit under its own client id, and the config is
validated before any row is written. The config selects downstream stages; the
workers run them under their own config, as for ``womblex enqueue-stages``.
"""
from __future__ import annotations

import logging
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import Literal, cast

import psycopg
import sqlalchemy.exc
from fastapi import Depends, FastAPI, HTTPException, Response, UploadFile

from womblex.api import readers
from womblex.api.auth import Caller, ClientRegistry, caller_dependency, require_scope
from womblex.api.models import (
    DocumentText,
    Health,
    Ready,
    RunFile,
    RunFiles,
    RunList,
    RunManifest,
    RunMetrics,
    RunRequest,
    RunStatus,
    RunSubmitted,
    UploadAccepted,
)
from womblex.cli._shared import SUPPORTED_EXTENSIONS, normalise_prefix, select_supported
from womblex.cloud import dispatch
from womblex.cloud.dispatch import QUEUE_CONNECT_TIMEOUT
from womblex.cloud.jobs import RunBoard, RunSummary
from womblex.cloud.workflow_order import StageOrderError
from womblex.store.contract import read_footer_contract
from womblex.store.remote import RemoteStore
from womblex.store.retention import generate_run_id
from womblex.ui import readers as ui_readers
from womblex.ui import resources
from womblex.ui.deps import UISettings

logger = logging.getLogger(__name__)

#: How many runs one caller's listing returns.
RUN_LIMIT = 1000

#: The default cap on one upload request's total size.
MAX_UPLOAD_BYTES = 256 * 1024 * 1024


def run_state(counts: dict[str, int]) -> str:
    """Collapse a status rollup into one state; any failed job fails the run."""
    if counts.get("failed"):
        return "failed"
    if counts.get("running") or (counts.get("pending") and counts.get("done")):
        return "running"
    return "pending" if counts.get("pending") else "done"


def mint_run_id() -> str:
    """A server-side run id: the sortable timestamp plus a random suffix.

    The timestamp alone is per-second, and two callers submitting in the same
    second would otherwise share a run.
    """
    return f"{generate_run_id()}-{secrets.token_hex(4)}"


def confine_prefix(caller: Caller, input_prefix: str) -> str:
    """*input_prefix* normalised; a non-admin caller may only name its own ``<client_id>/`` tree."""
    try:
        prefix = normalise_prefix(input_prefix)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    owner = caller.owner
    if owner is not None and prefix != owner and not prefix.startswith(f"{owner}/"):
        raise HTTPException(status_code=403, detail=f"input_prefix must be under {owner}/")
    return prefix


def upload_names(files: list[UploadFile]) -> list[str]:
    """Each upload's bare file name; 400 if one is unusable, repeated or unsupported.

    Only the final path segment is kept, so a client-side path cannot place a
    file outside the upload folder or nest it where a run would refuse it.
    """
    names = []
    for f in files:
        name = PurePosixPath((f.filename or "").replace("\\", "/")).name
        if name in ("", ".", "..") or "\x00" in name:
            raise HTTPException(status_code=400, detail=f"unusable file name: {f.filename!r}")
        names.append(name)
    if len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="file names must be unique within an upload")
    unsupported = sorted(set(names) - set(select_supported(names, location="upload")))
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type: {unsupported}; supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    return names


def upload_size(f: UploadFile) -> int:
    """The spooled upload's size in bytes, measured rather than taken from a header."""
    f.file.seek(0, 2)
    size = f.file.tell()
    f.file.seek(0)
    return size


def _status(run: RunSummary) -> RunStatus:
    return RunStatus(
        run_id=run.run_id, state=run_state(run.counts), counts=run.counts,  # type: ignore[arg-type]
        total=run.total, created_at=run.created_at, updated_at=run.updated_at,
    )


def create_api_app(
    *,
    store_uri: str,
    db_dsn: str,
    registry: ClientRegistry | None,
    ingest_uri: str | None = None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> FastAPI:
    """Build the API, bound to one store and queue for its lifetime.

    ``registry=None`` is the ``--insecure-no-auth`` development mode; the
    ``serve`` command, not this factory, refuses an empty registry. Without
    *ingest_uri* the reads serve and submission and uploads answer 503.
    """
    settings = UISettings(
        output_root=None, store_uri=store_uri, db_dsn=db_dsn, ingest_uri=ingest_uri,
    )
    app = FastAPI(title="Womblex API", version="1")
    get_caller = caller_dependency(registry)
    can_read = require_scope(get_caller, "read")
    can_submit = require_scope(get_caller, "submit")

    @contextmanager
    def queue() -> Iterator[RunBoard]:
        try:
            q = RunBoard(db_dsn, connect_timeout=QUEUE_CONNECT_TIMEOUT)
        except Exception as e:
            logger.warning("api: queue unreachable: %s", e)
            raise HTTPException(status_code=503, detail="job queue unreachable") from e
        with q:
            yield q

    def find_run(q: RunBoard, caller: Caller, run_id: str) -> RunSummary:
        found = q.runs(caller.owner, run_id=run_id, limit=1)
        if found:
            return found[0]
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")

    @app.get("/v1/health", response_model=Health, tags=["ops"])
    def health() -> Health:
        return Health()

    @app.get("/v1/ready", response_model=Ready, tags=["ops"])
    def ready(response: Response) -> Ready:
        store_ok = bool(resources.test_store(settings)["reachable"])
        queue_ok = bool(resources.test_queue(settings)["reachable"])
        ok = store_ok and queue_ok
        if not ok:
            response.status_code = 503
        return Ready(ready=ok, store=store_ok, queue=queue_ok)

    @app.get("/v1/runs", response_model=RunList, tags=["runs"])
    def list_runs(caller: Caller = Depends(can_read)) -> RunList:  # noqa: B008
        with queue() as q:
            return RunList(runs=[_status(r) for r in q.runs(caller.owner, limit=RUN_LIMIT)])

    @app.post("/v1/runs", response_model=RunSubmitted, status_code=201, tags=["runs"])
    def submit_run(body: RunRequest, caller: Caller = Depends(can_submit)) -> RunSubmitted:  # noqa: B008
        prefix = confine_prefix(caller, body.input_prefix)
        config = body.config
        if body.preset is not None:
            preset = ui_readers.get_any_preset(settings, body.preset)
            if preset is None:
                raise HTTPException(status_code=400, detail=f"unknown preset: {body.preset}")
            config = preset.config
        run_id = mint_run_id()
        written = False
        try:
            stages = dispatch.downstream_stages(settings, config) if config is not None else ()
            if stages:
                dispatch.check_stage_order(settings, cast(dict, config), stages, run_id)
            run = dispatch.enqueue_extraction(
                settings, input_prefix=prefix, run_id=run_id,
                batch_size=body.batch_size, owner=caller.owner,
            )
            written = True
            if stages:
                dispatch.enqueue_downstream_stages(
                    settings, run_id=run_id, config=cast(dict, config), owner=caller.owner,
                    order_checked=True,
                )
        except dispatch.ExecutionDisabled as e:
            raise HTTPException(status_code=503, detail=e.detail) from e
        except StageOrderError as e:  # stable `code` per issue; checked before any row is written
            raise HTTPException(
                status_code=400,
                detail={"message": str(e), "issues": [i.as_dict() for i in e.issues]},
            ) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except (psycopg.Error, sqlalchemy.exc.SQLAlchemyError) as e:
            # Batch rows already committed are not rolled back: name the run so
            # an operator can find it (or a caller can re-list it) after the outage.
            logger.warning(
                "api submit: queue unreachable: run_id=%s batches_written=%s: %s",
                run_id, written, e,
            )
            raise HTTPException(status_code=503, detail="job queue unreachable") from e
        logger.info("api submit: run_id=%s owner=%s stages=%s", run.run_id, caller.owner, stages)
        return RunSubmitted(
            run_id=run.run_id, document_count=run.document_count,
            batch_count=run.batch_count, stages=list(stages),
        )

    @app.post("/v1/uploads", response_model=UploadAccepted, status_code=201, tags=["uploads"])
    def upload(files: list[UploadFile], caller: Caller = Depends(can_submit)) -> UploadAccepted:  # noqa: B008
        if not ingest_uri:
            raise HTTPException(status_code=503, detail="no ingest location configured")
        names = upload_names(files)
        total = sum(upload_size(f) for f in files)
        if total > max_upload_bytes:
            raise HTTPException(
                status_code=413, detail=f"upload is {total} bytes; the limit is {max_upload_bytes}",
            )
        if "/" in caller.client_id or confine_prefix(caller, caller.client_id) != caller.client_id:
            raise HTTPException(status_code=400, detail="client id is not usable as a folder name")
        upload_id = mint_run_id()
        prefix = f"{caller.client_id}/{upload_id}"
        try:
            store = RemoteStore.from_uri(ingest_uri)
            for name, f in zip(names, files, strict=True):
                store.write_stream(f.file, f"{prefix}/{name}")
        except Exception as e:
            logger.warning("api upload: ingest unreachable: upload_id=%s: %s", upload_id, e)
            raise HTTPException(status_code=503, detail=f"ingest location unreachable: {e}") from e
        logger.info("api upload: %s files=%d bytes=%d", prefix, len(names), total)
        return UploadAccepted(upload_id=upload_id, input_prefix=prefix, files=names, bytes=total)

    @app.get("/v1/runs/{run_id}", response_model=RunStatus, tags=["runs"])
    def get_run(run_id: str, caller: Caller = Depends(can_read)) -> RunStatus:  # noqa: B008
        with queue() as q:
            return _status(find_run(q, caller, run_id))

    @app.get("/v1/runs/{run_id}/manifest", response_model=RunManifest, tags=["runs"])
    def get_manifest(run_id: str, caller: Caller = Depends(can_read)) -> RunManifest:  # noqa: B008
        with queue() as q:
            find_run(q, caller, run_id)
        try:
            rows = ui_readers.get_manifest_rows(settings, run_id)
        except ui_readers.StoreUnreachable as e:
            raise HTTPException(status_code=503, detail=f"run store unreachable: {e}") from e
        return RunManifest(run_id=run_id, documents=rows or [])

    @app.get("/v1/runs/{run_id}/files", response_model=RunFiles, tags=["runs"])
    def get_files(run_id: str, caller: Caller = Depends(can_read)) -> RunFiles:  # noqa: B008
        with queue() as q:
            find_run(q, caller, run_id)
        try:
            store = RemoteStore.from_uri(store_uri)
            keys = sorted(store.find(f"runs/{run_id}"))
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"run store unreachable: {e}") from e
        files = []
        for key in keys:
            entry = RunFile(key=key)
            footer = store.read_parquet_footer(key) if key.endswith(".parquet") else None
            if footer is not None:
                entry = RunFile(key=key, rows=footer[1], **read_footer_contract(footer[0]))
            files.append(entry)
        return RunFiles(run_id=run_id, files=files)

    @app.get(
        "/v1/runs/{run_id}/documents/{source_hash}/text", response_model=DocumentText, tags=["runs"],
    )
    def get_document_text(
        run_id: str, source_hash: str,
        layer: Literal["masked", "chunks", "elements"] = "masked",
        caller: Caller = Depends(can_read),  # noqa: B008
    ) -> DocumentText:
        spec = readers.LAYERS[layer]
        if spec.sensitivity == "raw" and not caller.has("read_raw"):
            raise HTTPException(status_code=403, detail="requires the 'read_raw' scope")
        with queue() as q:
            find_run(q, caller, run_id)
        try:
            rows = readers.document_rows(store_uri, run_id, source_hash, spec)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"run store unreachable: {e}") from e
        if not rows:
            raise HTTPException(
                status_code=404, detail=f"no {layer} text for document {source_hash} in {run_id}",
            )
        return DocumentText(
            run_id=run_id, source_hash=source_hash, layer=layer,
            sensitivity=spec.sensitivity, rows=rows,
        )

    @app.get("/v1/runs/{run_id}/metrics", response_model=RunMetrics, tags=["runs"])
    def get_metrics(run_id: str, caller: Caller = Depends(can_read)) -> RunMetrics:  # noqa: B008
        with queue() as q:
            find_run(q, caller, run_id)
            return RunMetrics(
                run_id=run_id,
                stats=q.stats(run_id, owner=caller.owner),
                workers=[w.__dict__ for w in q.workers(run_id, owner=caller.owner)],
                throughput=q.throughput(run_id, owner=caller.owner).__dict__,
            )

    return app
