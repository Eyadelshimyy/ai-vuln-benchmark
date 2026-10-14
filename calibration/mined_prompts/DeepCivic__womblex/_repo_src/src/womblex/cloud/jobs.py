"""The run board: enqueue work and read its progress, over DBOS workflow rows.

Replaces the ``womblex_jobs`` table. One workflow is one row of the old
board: an extraction batch (``kind='batch'``), or the coordinator that runs a
run's downstream stages in order (``kind='downstream'``) with one child
workflow per stage (``kind='stage'``). Workflow ids are deterministic
(``<run_id>:batch:0001``), so enqueueing twice attaches to the existing row
and a re-enqueue resumes a run rather than duplicating it.

Ownership travels as the ``owner`` workflow attribute. DBOS's own statuses are
mapped back to the board's vocabulary (``pending`` / ``running`` / ``done`` /
``failed``), which the API and console already speak.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Self, TypeGuard

import sqlalchemy.exc

from womblex.cloud import dbos_app

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dbos import EnqueueOptions, WorkflowStatus

logger = logging.getLogger(__name__)

_STATUS = {
    "ENQUEUED": "pending",
    "DELAYED": "pending",
    "PENDING": "running",
    "SUCCESS": "done",
    "ERROR": "failed",
    "CANCELLED": "failed",
    "MAX_RECOVERY_ATTEMPTS_EXCEEDED": "failed",
}


@dataclass
class JobSpec:
    """A batch to enqueue: which input keys, and where its shards go."""

    batch_num: int
    input_keys: list[str]
    shard_prefix: str
    ingest_root: str
    max_attempts: int = 3


@dataclass(frozen=True)
class JobRow:
    """One workflow as read — the job list's grain. Timestamps are ISO-8601 strings."""

    id: str
    run_id: str
    batch_num: int
    status: str
    attempts: int
    max_attempts: int
    locked_by: str | None
    locked_at: str | None
    error: str | None
    created_at: str | None
    updated_at: str | None
    kind: str = "batch"
    stage: str | None = None


@dataclass(frozen=True)
class RunSummary:
    """One run's rows rolled up by status — the run list's grain."""

    run_id: str
    owner: str | None
    counts: dict[str, int]
    created_at: str | None
    updated_at: str | None

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class RunOwnedError(Exception):
    """A run id is already owned by a different owner."""


@dataclass(frozen=True)
class WorkerState:
    """A worker's live hold on the board, derived from the workflow's executor.

    Not liveness: a stopped worker's unfinished workflows stay here until DBOS
    recovers them, so an old entry means orphaned work, not a busy worker.
    """

    worker_id: str
    running: int
    oldest_locked_at: str | None
    newest_locked_at: str | None


@dataclass(frozen=True)
class Throughput:
    """Completions inside a trailing window — the dashboard's rate tile."""

    window_seconds: float
    completed: int
    per_minute: float
    last_completed_at: str | None


def _iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat() if ms else None


def _attr(status: WorkflowStatus, key: str, default: Any = None) -> Any:
    return (status.attributes or {}).get(key, default)


def _error_text(error: BaseException | None) -> str | None:
    """The failure as the board shows it, with the cause DBOS wraps on exhausted retries."""
    if error is None:
        return None
    causes = getattr(error, "errors", None)
    last = f" — last error: {type(causes[-1]).__name__}: {causes[-1]}" if causes else ""
    return f"{error}{last}"[:2000]


def _schema_missing(e: Exception) -> bool:
    """The DBOS system tables have not been created yet (Postgres or SQLite)."""
    orig = getattr(e, "orig", None)
    return getattr(orig, "sqlstate", None) in ("42P01", "3F000") or "no such table" in str(orig)


def _board_status(status: WorkflowStatus) -> str:
    return _STATUS.get(status.status, "failed")


def _job_row(s: WorkflowStatus) -> JobRow:
    running = _board_status(s) == "running"
    return JobRow(
        id=s.workflow_id,
        run_id=_attr(s, "run_id", ""),
        batch_num=int(_attr(s, "batch_num", 0)),
        status=_board_status(s),
        attempts=0 if _board_status(s) == "pending" else 1,
        max_attempts=int(_attr(s, "max_attempts", 3)),
        locked_by=s.executor_id if running else None,
        locked_at=_iso(s.updated_at) if running else None,
        error=_error_text(s.error),
        created_at=_iso(s.created_at),
        updated_at=_iso(s.updated_at),
        kind=_attr(s, "kind", "batch"),
        stage=_attr(s, "stage"),
    )


@dataclass(frozen=True)
class FileChecksums:
    """A run's published files with their digests, and the workflows that may
    have published files without one (failed, unfinished, or recorded before
    checksums existed)."""

    checksums: dict[str, str]
    undigested: list[str]


def _is_digest_map(value: Any) -> TypeGuard[dict[str, str]]:
    return isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    )


class RunBoard:
    """Enqueue and read runs through one DBOS client."""

    def __init__(self, dsn: str | None, *, connect_timeout: float | None = None):
        self._client = dbos_app.open_client(dsn, connect_timeout=connect_timeout)

    def close(self) -> None:
        self._client.destroy()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- writes ---------------------------------------------------------------

    def _list(
        self, run_id: str | None = None, *, owner: str | None = None,
        status: list[str] | str | None = None, **kwargs: Any,
    ) -> list[WorkflowStatus]:
        kwargs.setdefault("load_input", False)
        kwargs.setdefault("load_output", False)
        if run_id:
            kwargs["workflow_id_prefix"] = dbos_app.run_prefix(run_id)
        try:
            rows = self._client.list_workflows(status=status, **kwargs)
        except sqlalchemy.exc.DBAPIError as e:
            # Readers do not migrate: a database no writer has touched holds no runs.
            if not _schema_missing(e):
                raise
            return []
        # Filtered here, not by DBOS: filtering on attributes is Postgres-only.
        return rows if owner is None else [s for s in rows if _attr(s, "owner") == owner]

    def _claim_run_owner(self, run_id: str, owner: str | None) -> str | None:
        """Resolve the owner new rows of *run_id* carry; refuse a different owner.

        An unscoped caller (``owner=None``: the CLI, the console) is not
        checked and inherits the run's existing owner. Two submitters racing to
        create the same run can both pass this check; the first row written
        names the owner and the second attaches to its deterministic ids.
        """
        existing = self._list(run_id, limit=1)
        if not existing:
            return owner
        current = _attr(existing[0], "owner")
        if owner is not None and current != owner:
            raise RunOwnedError(f"run {run_id!r} is owned by another owner")
        return owner if owner is not None else current

    def _enqueue(self, sends: list[tuple[EnqueueOptions, tuple]]) -> int:
        """Enqueue each workflow; return how many were not already on the board.

        The workflow id is deterministic, so enqueueing one that exists attaches
        to it rather than starting a second.
        """
        ids = [options["workflow_id"] for options, _ in sends]
        present = {s.workflow_id for s in self._client.list_workflows(
            workflow_ids=ids, load_input=False, load_output=False)}
        for options, args in sends:
            self._client.enqueue(options, *args)
        return len(set(ids) - present)

    def enqueue(self, run_id: str, jobs: list[JobSpec], *, owner: str | None = None) -> int:
        """Enqueue *jobs* for *run_id*; idempotent per batch number.

        Returns the number of batches that were not already on the board.
        Raises :class:`RunOwnedError` if *owner* is named and the run already
        belongs to someone else, including a run with no owner.
        """
        row_owner = self._claim_run_owner(run_id, owner)
        sends = []
        for spec in jobs:
            attrs = {"run_id": run_id, "kind": dbos_app.KIND_BATCH, "batch_num": spec.batch_num,
                     "max_attempts": spec.max_attempts}
            if row_owner is not None:
                attrs["owner"] = row_owner
            options: EnqueueOptions = {
                "workflow_name": dbos_app.EXTRACT_WORKFLOW,
                "queue_name": dbos_app.extract_queue(spec.ingest_root),
                "workflow_id": dbos_app.batch_id(run_id, spec.batch_num),
                "attributes": attrs,
            }
            sends.append((options, (run_id, spec.batch_num, spec.input_keys, spec.shard_prefix,
                                    spec.ingest_root, spec.max_attempts)))
        inserted = self._enqueue(sends)
        logger.info("Enqueued %d new job(s) for run %s (of %d submitted)",
                    inserted, run_id, len(jobs))
        return inserted

    def enqueue_stages(
        self, run_id: str, stages: list[str], shard_prefix: str, *,
        max_attempts: int = 3, owner: str | None = None, workflow: dict[str, Any] | None = None,
    ) -> int:
        """Enqueue the coordinator that runs *stages* in pipeline order for *run_id*.

        *workflow* (`id`, `digest`, `preset`, `config`: the resolved overlay) makes the fleet
        run the stages under that config and stamp its digest on what they write, rather
        than under the worker's own config; it is part of the coordinator's identity.

        Order is not taken from the list: the coordinator sorts by
        ``stage_rank``, so an out-of-order list still runs in pipeline order and
        a duplicated name collapses. The same stage list dispatched twice is the
        same coordinator, so pressing the button again re-runs nothing; a
        different list starts a new coordinator whose already-finished stages
        return their recorded result. Returns the number of stages newly
        dispatched. *owner* is checked as in :meth:`enqueue`.
        """
        from womblex.pipeline_order import in_pipeline_order

        ordered = list(in_pipeline_order(stages))
        row_owner = self._claim_run_owner(run_id, owner)
        attrs: dict[str, Any] = {"run_id": run_id, "kind": dbos_app.KIND_DOWNSTREAM,
                                 "max_attempts": max_attempts, "stages": ordered}
        if row_owner is not None:
            attrs["owner"] = row_owner
        if workflow is not None:
            attrs["workflow_digest"] = workflow["digest"]
        options: EnqueueOptions = {
            "workflow_name": dbos_app.DOWNSTREAM_WORKFLOW,
            "queue_name": dbos_app.DOWNSTREAM_QUEUE,
            "workflow_id": dbos_app.downstream_id(
                run_id, ordered, workflow["digest"] if workflow else None),
            "attributes": attrs,
        }
        fresh = self._enqueue(
            [(options, (run_id, ordered, shard_prefix, max_attempts, row_owner, workflow))],
        )
        newly = len(ordered) if fresh else 0
        logger.info("Enqueued %d new stage job(s) for run %s (of %d submitted)",
                    newly, run_id, len(stages))
        return newly

    def stage_workflow_digests(self, run_id: str) -> dict[str, set[str | None]]:
        """Each stage already dispatched for *run_id*, with the workflow digests it is under.

        `None` is a stage dispatched without a workflow (`enqueue-stages`, the console). A
        queued coordinator counts before it has enqueued its stages: it names them in its
        `stages` attribute (a coordinator enqueued before that attribute is seen through
        its stage rows only).
        """
        found: dict[str, set[str | None]] = {}
        for s in self._list(run_id):
            kind, digest = _attr(s, "kind"), _attr(s, "workflow_digest")
            names = [_attr(s, "stage")] if kind == dbos_app.KIND_STAGE else (
                _attr(s, "stages") or [] if kind == dbos_app.KIND_DOWNSTREAM else [])
            for name in names:
                found.setdefault(name, set()).add(digest)
        return found

    # --- reads ----------------------------------------------------------------

    def _rows(self, run_id: str | None, owner: str | None, **kwargs: Any) -> list[WorkflowStatus]:
        return self._list(run_id, owner=owner, **kwargs)

    def stats(self, run_id: str | None = None, *, owner: str | None = None) -> dict[str, int]:
        """Count jobs by status (optionally for one run, one owner)."""
        return dict(Counter(_board_status(s) for s in self._rows(run_id, owner)))

    def runs(
        self, owner: str | None = None, *, run_id: str | None = None, limit: int = 100,
    ) -> list[RunSummary]:
        """Runs with their jobs rolled up by status, latest activity first.

        *owner* narrows to one caller's runs; ``None`` lists every run,
        including CLI- and console-submitted ones that have no owner.
        """
        grouped: dict[str, list[WorkflowStatus]] = {}
        for s in self._rows(run_id, owner):
            grouped.setdefault(_attr(s, "run_id", ""), []).append(s)
        summaries = [
            RunSummary(
                rid, _attr(rows[0], "owner"),
                dict(Counter(_board_status(s) for s in rows)),
                _iso(min(s.created_at or 0 for s in rows)),
                _iso(max(s.updated_at or 0 for s in rows)),
            )
            for rid, rows in grouped.items() if rid
        ]
        return sorted(summaries, key=lambda r: r.updated_at or "", reverse=True)[:limit]

    def list_jobs(
        self, run_id: str | None = None, *, status: str | None = None, limit: int = 200,
        owner: str | None = None,
    ) -> list[JobRow]:
        """Recent jobs, newest activity first, optionally scoped by run and status."""
        rows = self._rows(run_id, owner, load_output=True)
        rows.sort(key=lambda s: s.updated_at or 0, reverse=True)
        jobs = [_job_row(s) for s in rows]
        return [j for j in jobs if status is None or j.status == status][:limit]

    def file_checksums(self, run_id: str) -> FileChecksums:
        """Every file the run's extraction and stage workflows published, key -> SHA-256.

        A finished workflow's digests come from its output; one that failed or
        has not finished contributes the steps it completed. Read oldest first,
        so a key a later workflow rewrote (an in-place stage, a re-dispatched
        stage) carries the newest digest.
        """
        rows = [s for s in self._rows(run_id, None, load_output=True)
                if _attr(s, "kind", dbos_app.KIND_BATCH) != dbos_app.KIND_DOWNSTREAM]
        rows.sort(key=lambda s: s.created_at or 0)
        checksums: dict[str, str] = {}
        undigested: list[str] = []
        for s in rows:
            digests: dict[str, str] = {}
            if _board_status(s) == "done":
                out = s.output if isinstance(s.output, dict) else {}
                if _is_digest_map(recorded := out.get("checksums")):
                    digests = recorded
                # A list output, or a count above the digests, is a workflow
                # recorded before checksums: its files carry none.
                if not isinstance(s.output, dict) or out.get("published", 0) > len(digests):
                    undigested.append(s.workflow_id)
            else:
                for step in self._client.list_workflow_steps(s.workflow_id):
                    if step["error"] is None and _is_digest_map(output := step["output"]):
                        digests.update(output)
                # A step that stopped part-way may have published without returning.
                undigested.append(s.workflow_id)
            checksums.update(digests)
        return FileChecksums(checksums, sorted(undigested))

    def stale_jobs(
        self, older_than_seconds: float, run_id: str | None = None, *,
        owner: str | None = None,
    ) -> list[JobRow]:
        """Running jobs with no update for longer than the threshold, oldest first.

        Read-only: DBOS recovers a dead executor's workflows when it restarts;
        this only names what has gone quiet.
        """
        cutoff = (datetime.now(UTC) - timedelta(seconds=older_than_seconds)).timestamp() * 1000
        quiet = [s for s in self._rows(run_id, owner, status="PENDING")
                 if (s.updated_at or 0) < cutoff]
        quiet.sort(key=lambda s: s.updated_at or 0)
        return [_job_row(s) for s in quiet]

    def workers(
        self, run_id: str | None = None, *, owner: str | None = None,
    ) -> list[WorkerState]:
        """Which executors hold which running workflows right now — the fleet view."""
        held: dict[str, list[int]] = {}
        for s in self._rows(run_id, owner, status="PENDING"):
            if s.executor_id:
                held.setdefault(s.executor_id, []).append(s.updated_at or 0)
        return [
            WorkerState(worker_id=w, running=len(t), oldest_locked_at=_iso(min(t)),
                        newest_locked_at=_iso(max(t)))
            for w, t in sorted(held.items())
        ]

    def throughput(
        self, run_id: str | None = None, *, window_seconds: float = 3600.0,
        owner: str | None = None,
    ) -> Throughput:
        """Jobs completed in the trailing window, as a rate."""
        since = (datetime.now(UTC) - timedelta(seconds=window_seconds)).isoformat()
        done = self._rows(run_id, owner, status="SUCCESS", completed_after=since)
        completed = len(done)
        return Throughput(
            window_seconds=window_seconds,
            completed=completed,
            per_minute=completed / (window_seconds / 60.0) if window_seconds > 0 else 0.0,
            last_completed_at=_iso(max((s.updated_at or 0 for s in done), default=0)),
        )


__all__ = [
    "JobRow",
    "JobSpec",
    "RunBoard",
    "RunOwnedError",
    "RunSummary",
    "Throughput",
    "WorkerState",
]
