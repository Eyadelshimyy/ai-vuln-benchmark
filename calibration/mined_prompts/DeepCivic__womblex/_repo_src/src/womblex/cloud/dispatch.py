"""Queue dispatch shared by the console and the service API (`docs/service-api.md`).

The two write actions that put work on the queue — plan an extraction run
(:func:`enqueue_extraction`) and dispatch a run's downstream stages
(:func:`enqueue_downstream_stages`) — plus the guard that refuses either when
a deployment lacks the store, ingest location or DSN they need. Extracted from
``womblex.ui.execute`` unchanged so a second caller reuses the same rules;
``ui.execute`` re-exports every name here.

Dispatch is always the queue. Nothing here shells out or runs a batch
in-process: it writes queue rows, and the workers a platform brings up do the
work. Execution needs both a remote store (to enqueue keys from and publish
shards to) and a job queue (to dispatch through); :func:`execution_status`
reports which piece is missing rather than half-working.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

from womblex.cli._shared import normalise_prefix, select_supported
from womblex.cloud.dbos_app import ensure_schema
from womblex.store.retention import generate_run_id, is_safe_run_id
from womblex.ui.deps import UISettings

#: Seconds to wait for the queue connection before reporting it unreachable. A
#: polled dashboard or a request thread would otherwise sit on a routable-but-dead
#: host until the OS gave up.
QUEUE_CONNECT_TIMEOUT = 5.0

logger = logging.getLogger(__name__)

class ExecutionDisabled(Exception):
    """A write action was attempted on a console that cannot execute.

    Carries a machine-readable ``reason`` the route maps to a 409 (store,
    ingest or queue not configured), so the frontend can tell which piece is
    unwired without parsing a message.
    """

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ExecutionCapability:
    """Whether this console can dispatch work, and if not, precisely why.

    All three must hold: an output location a ``RemoteStore`` can publish
    shards to, an ingest location to enqueue documents from, and a job queue
    to dispatch through (see the module docstring). ``can_execute`` is their
    conjunction; the individual flags are surfaced so the screen can name the
    missing piece rather than a bare "disabled".

    It reports no stage list. It used to carry ``STAGE_NAMES``, which read as
    "the stages this console dispatches" while nothing here could dispatch
    one — and no caller ever read it. The pipeline's shape is the composer's
    ``get_stage_graph()``, which serves it from the contracts.
    """

    has_store: bool
    has_ingest: bool
    has_queue: bool
    ingest_uri: str | None
    output_uri: str | None

    @property
    def can_execute(self) -> bool:
        return self.has_store and self.has_ingest and self.has_queue

    def as_dict(self) -> dict:
        return {
            "can_execute": self.can_execute,
            "has_store": self.has_store,
            "has_ingest": self.has_ingest,
            "has_queue": self.has_queue,
            "ingest_uri": self.ingest_uri,
            "output_uri": self.output_uri,
        }


def execution_status(settings: UISettings) -> ExecutionCapability:
    """What the console is allowed and able to dispatch — a cheap, network-free read.

    The screen loads this to decide whether to show the run/dispatch controls
    at all, and which explanation to show when it cannot. No store or queue
    connection is made here; reachability is the Resources Console's separate
    ``test`` actions.
    """
    return ExecutionCapability(
        has_store=settings.is_remote,
        has_ingest=bool(settings.ingest_uri),
        has_queue=bool(settings.db_dsn),
        ingest_uri=settings.ingest_uri,
        output_uri=settings.store_uri or (str(settings.output_root) if settings.output_root else None),
    )


def _guard(settings: UISettings, *, needs_ingest: bool = True) -> ExecutionCapability:
    """Refuse any write action the console is not configured to perform.

    Ordered so the operator sees the most actionable failure first: a missing
    store, ingest location or queue is a wiring gap (409), checked in that
    order. Every write path calls this before touching either.

    *needs_ingest* is false for dispatching the downstream stages: they read
    the shards extraction already published to the store and never look at the
    ingest location, so a deployment whose ingest has since been unset can
    still finish a run it started.
    """
    cap = execution_status(settings)
    if not cap.has_store:
        raise ExecutionDisabled(
            "no_store",
            "Execution dispatches through a shared object store; this console reads a "
            "local output_root. Point it at a --store to enqueue work.",
        )
    if needs_ingest and not cap.has_ingest:
        raise ExecutionDisabled(
            "no_ingest",
            "Execution enqueues documents from a configured ingest location; none is "
            "set. Set one (--ingest / $WOMBLEX_INGEST_URI) to enqueue work.",
        )
    if not cap.has_queue:
        raise ExecutionDisabled(
            "no_queue",
            "Execution dispatches through the job queue; no DSN is configured. Set one "
            "(--dsn / $WOMBLEX_DB_DSN) to enqueue work.",
        )
    return cap


def _supported_under(settings: UISettings, prefix: str) -> tuple[str, list[str]]:
    """The location *prefix* names, and the store keys a run there would ingest.

    One listing for both write and preview, so the count the composer shows is
    the count its press enqueues rather than a second walk that agrees today.
    Raises :class:`~womblex.cli._shared.NestedCorpusError` on a layout a run
    refuses.
    """
    from womblex.store.remote import RemoteStore

    ingest_uri = cast(str, settings.ingest_uri)
    location = f"{ingest_uri}/{prefix}".rstrip("/")
    ingest_store = RemoteStore.from_uri(ingest_uri)
    all_keys = ingest_store.list_files(prefix, "*", recursive=True)
    # Store-relative keys: strip the prefix for the nesting check, restore after.
    scope = f"{prefix}/" if prefix else ""
    names = select_supported((k.removeprefix(scope) for k in all_keys), location=location)
    return location, [f"{scope}{n}" for n in names]


@dataclass(frozen=True)
class EnqueueResult:
    """The outcome of an enqueue, for the screen to report and then poll.

    ``newly_enqueued`` distinguishes a fresh run from a resume: enqueue is
    idempotent on ``(run_id, batch_num)`` (`RunBoard.enqueue`), so
    re-dispatching a run that partly ran inserts only its missing batches and
    this counts them. ``run_id`` is what the Dashboard and Corpus Inspector
    are then pointed at to watch it drain.
    """

    run_id: str
    document_count: int
    batch_count: int
    newly_enqueued: int
    shard_prefix: str

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "document_count": self.document_count,
            "batch_count": self.batch_count,
            "newly_enqueued": self.newly_enqueued,
            "shard_prefix": self.shard_prefix,
        }


def enqueue_extraction(
    settings: UISettings,
    *,
    input_prefix: str | None = None,
    run_id: str | None = None,
    batch_size: int = 50,
    max_attempts: int = 3,
    owner: str | None = None,
) -> EnqueueResult:
    """Plan an extraction run into the queue — the "configure-and-run" action.

    The same three steps ``womblex enqueue`` does, against the configured
    ingest location: list supported documents under *input_prefix* (the whole
    ingest root when omitted), split them into ``batch_size`` batches, and
    write one idempotent queue row each, stamped with the ingest root.

    *input_prefix* is ingest-relative and goes through the listing
    :func:`ingest_preflight` previews, so a prefix the operator saw a count for
    enqueues that count.

    *owner* is the service caller the run belongs to (``None`` for the CLI
    and console); see `RunBoard.enqueue`.

    Raises :class:`ExecutionDisabled` when the console cannot dispatch (the
    route maps it to 409) and ``ValueError`` on bad input (→ 400) — an unsafe
    run id or prefix, a nested layout, or no documents under the prefix.
    """
    _guard(settings)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    resolved_run_id = run_id or generate_run_id()
    if not is_safe_run_id(resolved_run_id):
        raise ValueError(f"unsafe run_id: {resolved_run_id!r}")

    from womblex.cloud.jobs import JobSpec, RunBoard

    ingest_uri = cast(str, settings.ingest_uri)
    location, keys = _supported_under(settings, normalise_prefix(input_prefix))
    if not keys:
        raise ValueError(f"no supported documents under {location}")

    output_prefix = f"runs/{resolved_run_id}"
    shard_prefix = f"{output_prefix}/documents"
    specs = [
        JobSpec(
            batch_num=batch_idx,
            input_keys=keys[i : i + batch_size],
            shard_prefix=shard_prefix,
            max_attempts=max_attempts,
            ingest_root=ingest_uri,
        )
        for batch_idx, i in enumerate(range(0, len(keys), batch_size), start=1)
    ]

    ensure_schema(settings.db_dsn)
    with RunBoard(settings.db_dsn, connect_timeout=QUEUE_CONNECT_TIMEOUT) as board:
        newly = board.enqueue(resolved_run_id, specs, owner=owner)

    logger.info(
        "console enqueue: run_id=%s, %d doc(s) -> %d batch(es), %d newly enqueued",
        resolved_run_id, len(keys), len(specs), newly,
    )
    return EnqueueResult(
        run_id=resolved_run_id,
        document_count=len(keys),
        batch_count=len(specs),
        newly_enqueued=newly,
        shard_prefix=shard_prefix,
    )


def downstream_stages(settings: UISettings, config: dict) -> tuple[str, ...]:
    """The downstream stages *config* enables, in `PIPELINE_ORDER` — the shared gate.

    Validates *config* before anything is written, so a caller that enqueues
    extraction and stages together can refuse a bad config with no row on the
    queue. Raises ``pydantic.ValidationError`` on a config that would not load,
    and ``ValueError`` listing the known names when it names a model that is not
    registered — registration only, since this host may not carry the models the
    workers do. An empty result is the caller's to judge.
    """
    from womblex.config import WomblexConfig
    from womblex.pipeline_order import enabled_downstream_stages
    from womblex.ui.composer import deployment_paths
    from womblex.utils.model_check import check_registered

    # Built through the same `WomblexConfig(**{**raw, "paths": …})` construction
    # the composer validates and renders YAML with, so the stage list dispatched
    # is the one the config *as the CLI would load it* asks for — defaults and
    # coercions applied, not whatever the browser happened to send.
    #
    # `dataset` is filled in only when absent, the way `cli.cloud._runner_config`
    # does: `WomblexConfig` requires it but no stage gate reads it (stages run
    # over a shard prefix, not a dataset), so a config posted without one is a
    # dispatchable config, not an invalid one.
    paths, _ = deployment_paths(settings)
    raw = {"dataset": {"name": "console"}, **config, "paths": paths}
    validated = WomblexConfig(**raw)
    check_registered(validated)
    return enabled_downstream_stages(validated)


@dataclass(frozen=True)
class StageDispatchResult:
    """The outcome of dispatching a run's downstream stages.

    ``stages`` is what the composed config asked for, in `PIPELINE_ORDER` —
    reported back because the list is *derived*, not typed by the operator, and
    a press that quietly dispatched nothing (or dispatched embed when they
    thought embedding was off) should be visible on the screen that pressed it.

    ``newly_enqueued`` separates a first press from a repeat the same way
    :class:`EnqueueResult` does: `RunBoard.enqueue_stages` is idempotent
    per ``(run_id, stage)``, so pressing twice re-dispatches nothing and this
    reads 0.
    """

    run_id: str
    stages: tuple[str, ...]
    newly_enqueued: int
    shard_prefix: str

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "stages": list(self.stages),
            "newly_enqueued": self.newly_enqueued,
            "shard_prefix": self.shard_prefix,
        }


def check_stage_order(
    settings: UISettings, config: dict, stages: tuple[str, ...], run_id: str,
) -> None:
    """Refuse a stage list whose strict inputs no earlier stage or stored copy supplies.

    Writes nothing, so a submit can refuse with no row on the queue. Raises
    :class:`~womblex.cloud.workflow_order.StageOrderError` (a ``ValueError``, so 400).
    """
    from womblex.cloud.workflow_order import check_dispatch_order
    from womblex.config import WomblexConfig
    from womblex.store.remote import RemoteStore
    from womblex.ui.composer import deployment_paths

    paths, _ = deployment_paths(settings)
    validated = WomblexConfig(**{"dataset": {"name": "console"}, **config, "paths": paths})
    store = RemoteStore.from_uri(settings.store_uri) if settings.store_uri else None
    check_dispatch_order(stages, validated, store, f"runs/{run_id}/documents")


def enqueue_downstream_stages(
    settings: UISettings,
    *,
    run_id: str,
    config: dict,
    max_attempts: int = 3,
    owner: str | None = None,
    order_checked: bool = False,
) -> StageDispatchResult:
    """Dispatch *run_id*'s downstream stages — the console's second write action.

    Exactly what ``womblex enqueue-stages --run-id … --config …`` does, reached
    from the screen that composed the config: the stages *config* enables
    (:func:`~womblex.pipeline_order.enabled_downstream_stages`), written as
    queue rows for the workers to claim. The console still only writes rows —
    execution stays on the fleet, with its retry and crash recovery, and no
    request here can become a command because there is no command.

    Deliberately a separate press from :func:`enqueue_extraction`, per the same
    reasoning the CLI split them: a worker will not start a stage until the
    run's batches settle, but a *bad* extraction should not spend Isaacus
    budget on enrich and embed either. The operator enqueues, watches it drain,
    looks at it, then presses this.

    *owner* is checked against the run's owner as in `RunBoard.enqueue_stages`. The stage
    order is checked (:func:`check_stage_order`) unless the caller already did
    (*order_checked*, the API's submit, which checks before writing any row).

    ``pii`` and ``quality`` are never dispatched — that bound lives in
    ``DOWNSTREAM_STAGES``, not here, so the console cannot widen it. Both stay
    reachable through ``womblex run-stage``.

    Raises :class:`ExecutionDisabled` when the console cannot dispatch (→
    409), ``pydantic.ValidationError`` on a config that would not load (→
    400) and ``ValueError`` on an unsafe run id or a config that enables no
    stage at all (→ 400).
    """
    _guard(settings, needs_ingest=False)
    if not is_safe_run_id(run_id):
        raise ValueError(f"unsafe run_id: {run_id!r}")

    from womblex.cloud.jobs import RunBoard

    stages = downstream_stages(settings, config)
    if not stages:
        raise ValueError(
            "This config enables no downstream stages — nothing to dispatch. Turn on "
            "a stage (chunking, enrichment, embedding, money, linking) and press again."
        )

    shard_prefix = f"runs/{run_id}/documents"
    if not order_checked:
        check_stage_order(settings, config, stages, run_id)
    ensure_schema(settings.db_dsn)
    with RunBoard(settings.db_dsn, connect_timeout=QUEUE_CONNECT_TIMEOUT) as board:
        newly = board.enqueue_stages(
            run_id, list(stages), shard_prefix, max_attempts=max_attempts, owner=owner,
        )

    logger.info(
        "console stage dispatch: run_id=%s, stages %s, %d newly enqueued",
        run_id, " -> ".join(stages), newly,
    )
    return StageDispatchResult(
        run_id=run_id, stages=stages, newly_enqueued=newly, shard_prefix=shard_prefix,
    )
