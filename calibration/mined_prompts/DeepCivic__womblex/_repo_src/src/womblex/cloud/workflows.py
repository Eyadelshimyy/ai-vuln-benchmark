"""The DBOS workflows a worker runs: an extraction batch, a stage, and the coordinator.

An extraction batch stages its inputs from the ingest store, runs the shared
``womblex.batch.process_batch`` body (identical to ``womblex run``) and
publishes the shards. A stage plans its units (one per batch base, or one for
the whole run), then runs each as its own step, so a unit that finished is
recorded and a re-run starts after it. The coordinator waits for the run's
extraction to settle, then runs the requested stages in pipeline order.

Step results are storage keys, never data. The worker's config and stores are
process state (:func:`set_context`), not workflow arguments: arguments are
stored with the workflow, so they stay plain JSON-shaped values.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dbos import DBOS, SetWorkflowAttributes, SetWorkflowID, StepOptions

from womblex.batch import process_batch
from womblex.cloud import dbos_app
from womblex.cloud.stage_contracts import PRODUCER_OF, STAGE_CONTRACTS
from womblex.cloud.stage_runner import (
    InputContractError,
    NotReady,
    plan_units,
    prepare_stage_context,
    run_unit,
)
from womblex.store.run_stamp import RunStamp, workflow_context
from womblex.store.source_provenance import IngestProvenance
from womblex.utils.log_format import log_context
from womblex.utils.run_log import capture_batch_log

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.config import WomblexConfig
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

#: Seconds between looks at whether a run's extraction has settled.
SETTLE_POLL_SECONDS = 30.0
_UNSETTLED = ["ENQUEUED", "DELAYED", "PENDING"]


class StageNotReady(Exception):
    """An upstream sidecar the stage reads is absent: the stage was run out of order."""


@dataclass
class WorkerContext:
    """What this worker process runs with; set once before DBOS launches."""

    config: WomblexConfig
    store: RemoteStore
    ingest: RemoteStore
    ingest_root: str


_ctx: WorkerContext | None = None


def set_context(ctx: WorkerContext | None) -> None:
    global _ctx
    _ctx = ctx


def _context() -> WorkerContext:
    if _ctx is None:
        raise RuntimeError("no worker context: call set_context() before launching DBOS")
    return _ctx


def _not_a_failure_to_retry(e: BaseException) -> bool:
    """A missing input is a wiring fact, not a transient fault: retrying cannot fix it."""
    return not isinstance(e, NotReady | StageNotReady | InputContractError)


def _retry(name: str, max_attempts: int) -> StepOptions:
    return StepOptions(
        name=name, retries_allowed=True, max_attempts=max_attempts,
        interval_seconds=5.0, backoff_rate=2.0, should_retry=_not_a_failure_to_retry,
    )


def _output_prefix(shard_prefix: str) -> str:
    """The run's output dir — ``shard_prefix`` (``…/documents``) less its leaf."""
    return shard_prefix.rsplit("/", 1)[0]


def _logged[T](
    name: str, run_id: str, shard_prefix: str, stage: str | None,
    body: Callable[[Path], T],
) -> T:
    """Run *body* with its log captured, and publish the log on success or failure.

    A failed job still leaves its ``logs/<name>.log`` in the store — that is
    the case the operator most needs it. A failed *upload* of the log never
    masks the body's own error.
    """
    store = _context().store
    job_id = DBOS.workflow_id
    with tempfile.TemporaryDirectory(prefix="womblex-job-") as tmp:
        root = Path(tmp)
        log_path = root / f"{name}.log"
        try:
            with log_context(run_id=run_id, job_id=job_id, stage=stage), \
                    capture_batch_log(log_path):
                return body(root)
        finally:
            try:
                store.upload_file(log_path, f"{_output_prefix(shard_prefix)}/logs/{name}.log")
            except Exception:  # publishing the log must never mask the job's own error
                logger.exception("[%s] failed to publish job log", name)


# --- extraction --------------------------------------------------------------


def _extract(
    run_id: str, batch_num: int, input_keys: list[str], shard_prefix: str, ingest_root: str,
) -> dict[str, str]:
    """Stage this batch's inputs, extract them, publish the shards; return key -> SHA-256."""
    ctx = _context()

    def body(root: Path) -> dict[str, str]:
        shards_dir = root / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        # Nested: a job's keys come from a recursive listing, so two documents
        # under different prefixes routinely share a basename.
        files = ctx.ingest.download_to_dir(input_keys, root / "inputs", nested=True)
        declared = ingest_root or ctx.ingest_root
        provenance = (
            IngestProvenance.declare(
                declared, ctx.config.dataset.name,
                relpaths=dict(zip(files, input_keys, strict=True)),
            )
            if declared else None
        )
        stamp = RunStamp.declare(run_id, ctx.config, stage="extract") if run_id.strip() else None
        outcome = process_batch(
            files, ctx.config, batch_num=batch_num, shard_dir=shards_dir,
            provenance=provenance, stamp=stamp,
        )
        # Glob off the shard path the batch reported, so the naming scheme lives
        # only in womblex.batch.
        published = ctx.store.upload_glob(shards_dir, f"{outcome.shard_path.stem}.*", shard_prefix)
        logger.info(
            "[batch %d] %d ok, %d failed -> %s",
            batch_num, outcome.batch.succeeded, outcome.batch.failed, shard_prefix,
        )
        return published

    return _logged(f"batch-{batch_num:04d}", run_id, shard_prefix, None, body)


@DBOS.workflow(name=dbos_app.EXTRACT_WORKFLOW)
def extract_batch(
    run_id: str, batch_num: int, input_keys: list[str], shard_prefix: str,
    ingest_root: str, max_attempts: int,
) -> dict[str, Any]:
    recorded = DBOS.run_step(
        _retry("extract", max_attempts), _extract,
        run_id, batch_num, input_keys, shard_prefix, ingest_root,
    )
    return {"published": len(recorded), "checksums": _digests(recorded)}


def _digests(recorded: Any) -> dict[str, str]:
    """A step's published keys with digests; one recorded before checksums
    returned a key list, so its files carry none (the board names that gap)."""
    return recorded if isinstance(recorded, dict) else {}


# --- stages ------------------------------------------------------------------


def _stage_config(workflow: dict[str, Any] | None) -> WomblexConfig:
    """The config a stage runs under: the worker's own, or a dispatched workflow's recipe.

    A workflow's overlay is built exactly as the local runner builds it
    (`config_from_overlay`), keeping this worker's `paths`.
    """
    ctx = _context()
    if workflow is None:
        return ctx.config
    from womblex.cloud.workflow_runner import config_from_overlay

    return config_from_overlay(
        workflow["preset"], workflow["config"], ctx.config.paths.model_dump(mode="json"),
    )


def _plan(stage: str, shard_prefix: str, workflow: dict[str, Any] | None = None) -> list[list[str]]:
    ctx = _context()
    return plan_units(STAGE_CONTRACTS[stage], ctx.store, shard_prefix, _stage_config(workflow))


def _unit(
    stage: str, run_id: str, shard_prefix: str, unit: list[str],
    workflow: dict[str, Any] | None = None,
) -> dict[str, str]:
    ctx = _context()
    config = _stage_config(workflow)
    contract = STAGE_CONTRACTS[stage]
    label = unit[0] if len(unit) == 1 else f"{len(unit)}-bases"

    def body(_root: Path) -> dict[str, str]:
        # Built per unit: the client holds a connection, and a recovered
        # workflow resumes in a process that has none.
        run_ctx = prepare_stage_context(contract, config)
        try:
            with workflow_context(workflow["digest"]) if workflow else nullcontext():
                return run_unit(contract, config, run_ctx, ctx.store, shard_prefix, unit, ctx.ingest)
        except NotReady as nr:
            producer = PRODUCER_OF.get(nr.suffix, "the upstream stage")
            raise StageNotReady(
                f"{stage}: {nr.stem} is missing {nr.suffix} — {producer} has not produced it"
            ) from nr

    return _logged(f"stage-{stage}-{label}", run_id, shard_prefix, stage, body)


@DBOS.workflow(name=dbos_app.STAGE_WORKFLOW)
def run_stage(
    run_id: str, stage: str, shard_prefix: str, max_attempts: int,
    workflow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    units = DBOS.run_step({"name": "plan"}, _plan, stage, shard_prefix, workflow)
    published = 0
    checksums: dict[str, str] = {}
    for unit in units:
        label = unit[0] if len(unit) == 1 else "run"
        recorded = DBOS.run_step(
            _retry(f"unit:{label}", max_attempts), _unit, stage, run_id, shard_prefix, unit,
            workflow,
        )
        published += len(recorded)
        checksums.update(_digests(recorded))
    return {"stage": stage, "units": len(units), "published": published, "checksums": checksums}


# --- coordinator -------------------------------------------------------------


def _unsettled_batches(run_id: str) -> int:
    return len(DBOS.list_workflows(
        workflow_id_prefix=f"{run_id}:batch:", status=_UNSETTLED,
        load_input=False, load_output=False,
    ))


@DBOS.workflow(name=dbos_app.DOWNSTREAM_WORKFLOW)
def run_downstream(
    run_id: str, stages: list[str], shard_prefix: str, max_attempts: int, owner: str | None,
    workflow: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run *stages* over the run once its extraction has settled, in the order given.

    A failed batch does not hold the run: its absence surfaces as the
    not-ready or missing-input error of the stage that needed it. A failed
    stage ends the coordinator, since every later stage reads what it wrote.
    """
    while DBOS.run_step({"name": "await-extraction"}, _unsettled_batches, run_id):
        DBOS.sleep(SETTLE_POLL_SECONDS)

    results: list[dict[str, Any]] = []
    for stage in stages:
        attrs: dict[str, Any] = {"run_id": run_id, "kind": dbos_app.KIND_STAGE, "stage": stage,
                                 "max_attempts": max_attempts}
        if owner is not None:
            attrs["owner"] = owner
        if workflow is not None:
            attrs["workflow_digest"] = workflow["digest"]
        # Enqueued by name, not registered: the workers that serve the stage
        # own the queue's settings.
        with SetWorkflowID(dbos_app.stage_id(run_id, stage)), SetWorkflowAttributes(attrs):
            handle = DBOS.enqueue_workflow(
                dbos_app.stage_queue(stage), run_stage, run_id, stage, shard_prefix, max_attempts,
                workflow,
            )
        results.append(handle.get_result())
    return results
