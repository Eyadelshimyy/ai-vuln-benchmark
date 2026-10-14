"""``/api/execute`` — the Execution Controls (docs/ui-plan.md merge 11).

The one writable-to-a-run surface in the console. ``GET /status`` is a
cheap, network-free read of whether this deployment can dispatch work (and
if not, why); ``POST /enqueue`` plans an extraction run into the queue and
``POST /stages`` dispatches that run's downstream stages. All delegate to
:mod:`womblex.ui.execute`, which enforces the store+ingest+queue requirement
before touching anything — this router only maps its
:class:`~womblex.ui.execute.ExecutionDisabled` reasons onto HTTP status
codes.

The run/log feed is the queue's own view the Dashboard already serves
(``/api/dashboard``); nothing is duplicated here.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError

from womblex.ui import execute
from womblex.ui.deps import UISettings, get_settings

router = APIRouter(prefix="/api/execute", tags=["execute"])

#: `ExecutionDisabled.reason` -> HTTP status. A missing store, ingest location
#: or queue is a wiring gap on an otherwise willing console (409 Conflict — the
#: request cannot be served in the console's current state). Kept as data so a
#: new reason must be mapped explicitly rather than defaulting to a misleading
#: status.
_REASON_STATUS: dict[str, int] = {
    "no_store": 409,
    "no_ingest": 409,
    "no_queue": 409,
}


class EnqueueRequest(BaseModel):
    """The configure-and-run form: which documents, at what run id and batching.

    ``input_prefix`` is ingest-relative (the enqueue lists
    ``<ingest_uri>/<prefix>``), matching ``womblex enqueue --input-prefix`` and
    ``GET /ingest``'s query parameter — the composer previews a prefix there
    and posts that same string here. Optional: omitted means the whole
    configured ingest root, which is the normal case.
    ``run_id`` omitted mints a fresh timestamped id; supplying an existing
    one resumes it, since enqueue is idempotent on ``(run_id, batch_num)``.
    """

    input_prefix: str | None = None
    run_id: str | None = None
    batch_size: int = Field(50, ge=1)
    max_attempts: int = Field(3, ge=1)


@router.get("/status")
def get_status(settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008 - FastAPI DI idiom
    """Whether the console can dispatch work, and which piece is missing if not."""
    return execute.execution_status(settings).as_dict()


@router.get("/ingest")
def get_ingest(
    input_prefix: str | None = None, settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """Reachability + document count of the ingest location a prefix names.

    Feeds the composer's "N documents ready" line before it enqueues.
    ``input_prefix`` is the same ingest-relative prefix
    :class:`EnqueueRequest` takes, and the response echoes it back, so the
    composer posts the prefix it previewed and the two cannot disagree about
    which documents a press will run. Omitted, the whole ingest root, as
    before.

    400 on a prefix that escapes the ingest root — the refusal the enqueue
    applies to the same string.
    """
    try:
        return execute.ingest_preflight(settings, input_prefix=input_prefix)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/enqueue")
def post_enqueue(
    body: EnqueueRequest, settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """Plan an extraction run into the queue; workers the platform brings up run it.

    409 when no store, ingest location or queue is configured (the states
    :class:`ExecutionDisabled` distinguishes), 400 on bad input (no documents
    under the prefix, an unsafe run_id).
    """
    try:
        result = execute.enqueue_extraction(
            settings,
            input_prefix=body.input_prefix,
            run_id=body.run_id,
            batch_size=body.batch_size,
            max_attempts=body.max_attempts,
        )
    except execute.ExecutionDisabled as e:
        raise HTTPException(status_code=_REASON_STATUS[e.reason], detail=e.detail) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return result.as_dict()


class StageDispatchRequest(BaseModel):
    """The "run downstream stages" press: which run, and the composed config.

    ``config`` is the composer's current form state — the same whole config
    ``POST /api/composer/validate`` takes — because *it* is what decides which
    stages a run wants. The queue still carries no config: the rows name only
    the stage, and the workers read their own ``--config`` at launch, exactly
    as ``womblex enqueue-stages --config`` behaves.

    ``run_id`` is required and never minted here. Stages run over shards that
    already exist, so there is no such thing as dispatching them for a run
    nobody extracted.
    """

    run_id: str = Field(..., min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)
    max_attempts: int = Field(3, ge=1)


@router.post("/stages")
def post_stages(
    body: StageDispatchRequest, settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """Dispatch this run's downstream stages; workers claim them once it drains.

    Same status shapes as ``POST /enqueue`` — 409 store/ingest/queue unwired,
    400 bad input — plus 400 carrying Pydantic's own error list when
    the posted config would not load, matching what the composer's preset save
    returns for the same fault.
    """
    try:
        result = execute.enqueue_downstream_stages(
            settings,
            run_id=body.run_id,
            config=body.config,
            max_attempts=body.max_attempts,
        )
    except execute.ExecutionDisabled as e:
        raise HTTPException(status_code=_REASON_STATUS[e.reason], detail=e.detail) from e
    except ValidationError as e:
        detail = e.errors(include_url=False, include_context=False, include_input=False)
        raise HTTPException(status_code=400, detail=detail) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return result.as_dict()
