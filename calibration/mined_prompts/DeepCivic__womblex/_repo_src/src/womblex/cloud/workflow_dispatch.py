"""Object-store execution of a stored workflow: dispatch its steps to the fleet.

The fleet's counterpart to `run_workflow_local`, through the coordinator path
(`RunBoard.enqueue_stages` -> `run_downstream`). Everything that can refuse does so
here, before a queue row exists, with the local runner's refusal type and codes:
the spec, preset and config (`workflow_runner`), the step order against the store's
listing (`workflow_order.check_dispatch_order`), and three things only a fleet has to
say: the stage is not dispatchable, the list is not in pipeline order (the coordinator
sorts, so another order would not be the order the workflow declares), or the run
already ran a stage under a different workflow. The workflow's resolved overlay and
digest travel with the dispatch (`enqueue_stages(workflow=…)`), so workers run the
stages under that recipe and stamp its digest.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

from womblex.cloud.workflow_order import StageOrderError, check_dispatch_order
from womblex.cloud.workflow_runner import (
    RefusalCode,
    WorkflowRefusedError,
    config_from_overlay,
    not_acted_on,
    workflow_overlay,
)
from womblex.cloud.workflow_store import StoredWorkflow, check_spec
from womblex.pipeline_order import DOWNSTREAM_STAGES, in_pipeline_order
from womblex.store.retention import is_safe_run_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.ui.presets import Preset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowDispatch:
    run_id: str
    workflow_id: str
    digest: str
    stages: tuple[str, ...]
    shard_prefix: str
    newly_enqueued: int
    dry_run: bool


def dispatch_workflow(
    stored: StoredWorkflow,
    *,
    resolve_preset: Callable[[str], Preset | None],
    run_id: str,
    store_uri: str | None,
    dsn: str | None,
    output_prefix: str | None = None,
    owner: str | None = None,
    max_attempts: int = 3,
    dry_run: bool = False,
) -> WorkflowDispatch:
    """Check *stored* against the store and the board, then enqueue it for *run_id*.

    Raises `WorkflowRefusedError` before any row is written. *dry_run* runs every check and
    writes nothing.
    """
    def refuse(message: str, code: RefusalCode, **kw: object) -> WorkflowRefusedError:
        return WorkflowRefusedError(f"workflow {stored.id!r}: {message}", code=code, **kw)  # type: ignore[arg-type]

    if not is_safe_run_id(run_id):
        raise refuse(f"unsafe run_id {run_id!r}", RefusalCode.CONFIG_INVALID)
    try:
        check_spec(stored.spec)
    except ValueError as e:
        raise refuse(f"is no longer valid: {e}", RefusalCode.SPEC_INVALID) from e
    overlay = workflow_overlay(stored, resolve_preset)
    try:
        config = config_from_overlay(stored.spec.preset, overlay)
    except ValueError as e:
        raise refuse(f"its config does not load: {e}", RefusalCode.CONFIG_INVALID) from e

    stages = tuple(s.stage for s in stored.spec.steps)
    if bad := [s for s in stages if s not in DOWNSTREAM_STAGES]:
        raise refuse(
            f"the fleet does not dispatch {', '.join(bad)} (run them with `run-stage`, or "
            "`run-workflow` locally)", RefusalCode.STAGE_NOT_DISPATCHABLE,
            steps=tuple((s, "not dispatchable") for s in bad),
        )
    if stages != tuple(in_pipeline_order(stages)):
        raise refuse(
            f"steps {list(stages)} are not in pipeline order {list(in_pipeline_order(stages))}; "
            "the coordinator runs stages in pipeline order, so reorder the steps",
            RefusalCode.ORDER_NOT_PIPELINE,
        )
    from womblex.utils.model_check import check_registered

    try:
        check_registered(config)  # registration only: this host may not carry the workers' models
    except ValueError as e:
        raise refuse(str(e), RefusalCode.CONFIG_INVALID) from e

    shard_prefix = f"{(output_prefix or f'runs/{run_id}').strip('/')}/documents"
    from womblex.store.remote import RemoteStore

    try:
        check_dispatch_order(
            list(stages), config, RemoteStore.from_uri(store_uri) if store_uri else None, shard_prefix,
        )
    except StageOrderError as e:
        raise refuse(f"step order cannot complete: {e}", RefusalCode.STEP_ORDER) from e

    if stored.spec.conditions:
        import tempfile
        from pathlib import Path

        from womblex.cloud.workflow_conditions import (
            ConditionRefused,
            manifest_from_store,
            require_conditions,
        )

        try:
            if not store_uri:
                raise ConditionRefused("condition_unevaluable", "no --store to read manifests from")
            with tempfile.TemporaryDirectory(prefix="womblex-cond-") as tmp:
                manifest = manifest_from_store(RemoteStore.from_uri(store_uri), shard_prefix, Path(tmp))
            require_conditions(stored.spec.conditions, manifest)
        except ConditionRefused as e:
            raise refuse(str(e), RefusalCode(e.code)) from e

    from womblex.cloud.dbos_app import ensure_schema, local_database_path
    from womblex.cloud.jobs import RunBoard

    if not dry_run:
        ensure_schema(dsn)
    # A dry run does not create the local database just to find it empty.
    unread = dry_run and not dsn and not local_database_path().exists()
    with nullcontext() if unread else RunBoard(dsn) as board:
        clash = {
            s: sorted(str(d) for d in ds - {stored.digest})
            for s, ds in (board.stage_workflow_digests(run_id).items() if board else ())
            if s in stages and ds - {stored.digest}
        }
        if clash:
            raise refuse(
                f"run {run_id!r} already dispatched {', '.join(sorted(clash))} under a different "
                "workflow (or none); a finished stage would return its recorded result, not run "
                "this one. Use a new run id.", RefusalCode.RUN_CONFLICT,
                steps=tuple((s, ", ".join(d)) for s, d in sorted(clash.items())),
            )
        newly = 0 if dry_run or board is None else board.enqueue_stages(
            run_id, list(stages), shard_prefix, max_attempts=max_attempts, owner=owner,
            workflow={"id": stored.id, "digest": stored.digest,
                      "preset": stored.spec.preset, "config": overlay},
        )
    if left := not_acted_on(stored):
        logger.warning("workflow %s: %s stored but not acted on by dispatch", stored.id, ", ".join(left))
    return WorkflowDispatch(
        run_id, stored.id, stored.digest, stages, shard_prefix, newly, dry_run,
    )


__all__ = ["WorkflowDispatch", "dispatch_workflow"]
