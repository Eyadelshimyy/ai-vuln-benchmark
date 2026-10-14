"""The Dashboard's read model (docs/ui-plan.md merge 8).

Two sources, both already written by the pipeline:

- **The run board**, when a DSN is configured. Counts are exact, the job list
  is the DBOS workflow rows, the fleet is the executor holding each running
  workflow, and throughput is derived from completion times.
- **Per-stage checkpoints**, always. Every shard stage writes its
  ``CheckpointState`` to a dot-directory *inside the run*
  (``<run>/.chunk-checkpoint/`` and friends), which ``STAGE_CONTRACTS``
  already names via ``checkpoint_dirname``. That makes stage progress
  readable from the run the console is already pointed at, in both
  deployments, with no configuration of its own.

Separate from :mod:`womblex.ui.readers` because none of this is a parquet
sidecar read: the queue is Postgres and the checkpoints are JSON.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

from womblex.cloud.dispatch import QUEUE_CONNECT_TIMEOUT
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.pipeline_order import sort_by_pipeline
from womblex.store.checkpoint import CHECKPOINT_GLOB, CheckpointProgress, read_checkpoints
from womblex.store.retention import is_safe_run_id
from womblex.ui.deps import UISettings

if TYPE_CHECKING:
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

#: Default staleness threshold, in seconds. A ``running`` workflow with no
#: update for longer than this is named on the dashboard; DBOS recovers it when
#: its executor restarts.
DEFAULT_STALE_AFTER = 900.0

#: Default trailing window for the throughput tile, in seconds.
DEFAULT_THROUGHPUT_WINDOW = 3600.0

#: Stage name -> the dot-directory its checkpoint lands in, taken from the
#: contracts rather than re-typed here so a renamed directory cannot drift.
#: Stages with no checkpoint of their own (``quality``, whose scope is the
#: whole run) are absent by construction.
#:
#: Ordered by `sort_by_pipeline`, because `_stage_progress` reports "per-stage
#: progress, in pipeline order" and `STAGE_CONTRACTS`'s own dict order is an
#: inventory, not a sequence — it lists `chunk` before `enrich`, the reverse
#: of the order a run executes them in.
CHECKPOINT_DIRNAMES: dict[str, str] = sort_by_pipeline({
    name: contract.checkpoint_dirname
    for name, contract in STAGE_CONTRACTS.items()
    if contract.checkpoint_dirname is not None
})


def _EMPTY_QUEUE_SECTION(window_seconds: float) -> dict:
    """The queue payload a reachable-but-empty queue produces.

    Same shape a live read returns with zero rows, so the frontend renders the
    tiles/lists as empty rather than special-casing it. Used when the
    DBOS system tables do not exist yet (a fresh database before ``init``
    or the first enqueue) — an empty queue, not a fault.
    """
    return {
        "stats": {},
        "total": 0,
        "jobs": [],
        "workers": [],
        "stale": [],
        "throughput": {
            "window_seconds": window_seconds,
            "completed": 0,
            "per_minute": 0.0,
            "last_completed_at": None,
        },
    }


def _is_missing_jobs_table(exc: Exception) -> bool:
    """Whether *exc* says the DBOS system tables do not exist yet.

    A reachable database no worker or dispatcher has written to has no schema,
    and the console does not create it. Matched on the message because the
    driver error arrives wrapped (SQLAlchemy over psycopg or sqlite): Postgres
    says the ``workflow_status`` relation does not exist, SQLite that there is
    no such table. Anything else (a real connection failure, a permissions
    error) is not swallowed; it propagates to the ``queue_error``.
    """
    text = str(exc).lower()
    return "workflow_status" in text and ("exist" in text or "no such table" in text)


def get_dashboard(
    settings: UISettings,
    *,
    run_id: str | None = None,
    stale_after: float = DEFAULT_STALE_AFTER,
    window_seconds: float = DEFAULT_THROUGHPUT_WINDOW,
    job_limit: int = 200,
) -> dict:
    """Queue state and per-stage progress, scoped to *run_id* when given.

    ``queue`` is ``None`` whenever there is no queue to read — no DSN
    configured, or the connection failed — and ``queue_error`` says which.
    That is a normal local deployment, not a fault, so the checkpoint half
    of the payload is unaffected by it.
    """
    queue, queue_error = queue_section(
        settings, run_id, stale_after=stale_after, window_seconds=window_seconds,
        job_limit=job_limit,
    )
    return {
        "run_id": run_id,
        "stale_after_seconds": stale_after,
        "queue": queue,
        "queue_error": queue_error,
        "stages": _stage_progress(settings, run_id) if run_id else [],
    }


def queue_section(
    settings: UISettings,
    run_id: str | None,
    *,
    stale_after: float,
    window_seconds: float,
    job_limit: int,
) -> tuple[dict | None, str | None]:
    """Read every queue view in one connection, or explain why there is none.

    A queue that cannot be reached is reported rather than raised: the
    dashboard's other half still renders, and an operator seeing "queue
    unreachable" next to live checkpoint progress has more to go on than a
    500.

    A *reachable* database whose DBOS system tables do not exist yet is not
    an error: it is a fresh Postgres that ``init`` (or the first enqueue) has
    not created the schema in, and its honest state is an empty queue. Reading
    it as "unreachable" was a reported symptom — the console gates the whole
    dashboard/execution surface on the queue, so a fresh cluster read as a
    fault until the table happened to appear. We report the empty queue and
    never create the table (the console is read-only; ``init``/enqueue own
    creation).

    Not prefixed private: the Resources Console reuses this as its queue
    connectivity test and fleet/queue-depth read (docs/ui-plan.md merge 10)
    rather than reimplementing the same connect-and-read.
    """
    if not settings.db_dsn:
        return None, None
    try:
        from womblex.cloud.jobs import RunBoard

        with RunBoard(settings.db_dsn, connect_timeout=QUEUE_CONNECT_TIMEOUT) as board:
            try:
                stats = board.stats(run_id)
                jobs = board.list_jobs(run_id, limit=job_limit)
                workers = board.workers(run_id)
                stale = board.stale_jobs(stale_after, run_id)
                throughput = board.throughput(run_id, window_seconds=window_seconds)
            except Exception as e:
                if _is_missing_jobs_table(e):
                    return _EMPTY_QUEUE_SECTION(window_seconds), None
                raise
    except Exception as e:
        logger.warning("dashboard: queue unavailable: %s", e)
        return None, str(e)
    return {
        "stats": stats,
        "total": sum(stats.values()),
        "jobs": [asdict(j) for j in jobs],
        "workers": [asdict(w) for w in workers],
        # Whole rows, not ids: `jobs` is capped by `job_limit` ordered on
        # `updated_at DESC`, and a stale row's `updated_at` is by definition
        # among the oldest — so it is the first thing to fall outside that
        # window, and ids alone would be unresolvable exactly when they
        # matter. Bounded by the running set, so it stays small.
        "stale": [asdict(j) for j in stale],
        "throughput": asdict(throughput),
    }, None


def _stage_progress(settings: UISettings, run_id: str) -> list[dict]:
    """Per-stage checkpoint progress for one run, in pipeline order.

    Only stages with a checkpoint on disk appear — an absent directory means
    the stage has not run for this run, which the Corpus Inspector's stage
    presence already reports from the sidecars themselves.

    ``run_id`` arrives as a *query* parameter here, not a path segment, so
    nothing has already rejected ``../``: unlike the other console reads,
    this one must contain the join itself. A run_id that is not a single
    path segment reads as "no such run", which is the same answer the
    remote branch's ``list_dirs`` check already gives — the two deployments
    must not disagree about what is reachable.
    """
    if not is_safe_run_id(run_id):
        return []
    if settings.is_remote:
        found = _remote_checkpoints(cast(str, settings.store_uri), run_id)
    else:
        run_dir = cast(Path, settings.output_root) / run_id
        found = {
            stage: read_checkpoints(run_dir / dirname)
            for stage, dirname in CHECKPOINT_DIRNAMES.items()
        }
    return [
        {"stage": stage, **asdict(progress)}
        for stage, entries in found.items()
        for progress in entries
    ]


def _remote_checkpoints(store_uri: str, run_id: str) -> dict[str, list[CheckpointProgress]]:
    """Stage each run's checkpoint JSONs into a temp dir and read them locally.

    Checkpoints are a few KB apiece, so this stays the same
    stage-then-reuse-the-local-reader pattern :mod:`womblex.ui.readers` uses
    for manifests — one definition of what a checkpoint file means.
    """
    from womblex.store.remote import RemoteStore

    store: RemoteStore = RemoteStore.from_uri(store_uri)
    if run_id not in store.list_dirs("runs"):
        return {}
    found: dict[str, list[CheckpointProgress]] = {}
    with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
        for stage, dirname in CHECKPOINT_DIRNAMES.items():
            keys = store.list_files(f"runs/{run_id}/{dirname}", CHECKPOINT_GLOB)
            if not keys:
                continue
            stage_dir = Path(tmp) / dirname
            stage_dir.mkdir()
            store.download_to_dir(keys, stage_dir)
            found[stage] = read_checkpoints(stage_dir)
    return found
