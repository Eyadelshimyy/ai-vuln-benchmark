"""A pipeline run, recorded: a ``pipeline_run`` row in the shared state store and
a log file under ``data/logs/pipelines/``.

Every way of running a whole pipeline leaves the same two things (the app's Run
button, in the background, and ``tracebi run-pipeline``), so the Refresh page
lists them together. The row holds the outcome; the file holds what the steps
printed, as it was printed. Polling reads the row first and the file second: a
terminal status is written only after the log is closed, so a poller that sees
one has the whole log.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Iterator, Optional

from tracebi.pipeline import runlog

KIND = "pipeline_run"

# The reports the run in progress has rebuilt. A ContextVar rather than a module
# global, for the reason the audit actor and the log sinks are: each run holds
# its own list, so concurrent runs (the app's worker threads) never share one.
_BUILT: ContextVar[Optional[list]] = ContextVar("tracebi_run_built_reports", default=None)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def begin(pipeline: str, order: list[str]) -> int:
    """Record a run of *order* (layer names) starting; the row exists on return."""
    from tracebi.state import record_run

    # Fail here, before a row exists, if the log has nowhere to go.
    os.makedirs(os.path.dirname(runlog.log_path(pipeline, 0)), exist_ok=True)
    return record_run(kind=KIND, target=pipeline, status="running",
                      started=_now(), detail={"layers": order})


def built_report(name: str) -> None:
    """Say that the step in progress just rebuilt report *name*; the run lists it.

    Does nothing outside a recorded run."""
    built = _BUILT.get()
    if built is not None and name not in built:
        built.append(name)


class Recording:
    """What a block run under :func:`recording` can tell its run."""

    def __init__(self, log: Optional[runlog.RunLog]) -> None:
        self.log = log                  # None: this run is not being recorded
        self.error: Optional[dict] = None

    def fail(self, exc: BaseException) -> None:
        """Mark the run failed without leaving the block: a step failed and the
        rest still ran. The first failure is the one recorded."""
        if self.error is not None:
            return
        trace = "".join(traceback.format_exception(exc)) \
            if os.environ.get("TRACEBI_DEV_MODE") == "1" else ""
        for line in trace.splitlines():
            if self.log is not None:
                self.log.line(line)
        self.error = {"message": f"Run failed: {str(exc) or type(exc).__name__}",
                      "exception_type": type(exc).__name__, "traceback": trace}


@contextlib.contextmanager
def recording(pipeline: str, run_id: Optional[int], order: list[str], *,
              echo: bool = True) -> Iterator[Recording]:
    """Run the block as run *run_id* of *pipeline* (from :func:`begin`).

    What the block prints on this thread goes to the run's log (``echo=False``
    keeps it off the terminal too). When the block ends the log is closed and
    the row says how it went. An exception leaving the block fails the run and
    is raised again, for the caller to decide what a failed run means to it.
    With ``run_id=None`` nothing is recorded and the block just runs.
    """
    if run_id is None:
        yield Recording(None)
        return
    from tracebi.state import update_run

    log = runlog.RunLog(runlog.log_path(pipeline, run_id))
    rec = Recording(log)
    built: list[str] = []
    token = _BUILT.set(built)
    began = time.monotonic()
    try:
        with runlog.capture(log.write, echo=echo):
            log.line(f"Pipeline {pipeline}: {' → '.join(order)}")
            yield rec
    except BaseException as exc:      # noqa: BLE001 — a KeyboardInterrupt ends the run too
        rec.fail(exc)
        raise
    finally:
        _BUILT.reset(token)
        log.line(f"{'Run failed' if rec.error else 'Run succeeded'} in {time.monotonic() - began:.1f}s")
        log.close()
        detail: dict = {"layers": order}
        if rec.error is not None:
            detail["error"] = rec.error
        if built:
            detail["reports"] = built
        finished = _now()
        try:
            update_run(None, run_id, status="failed" if rec.error else "succeeded",
                       finished=finished, completed_at=finished, output_path=log.path, detail=detail)
        finally:
            runlog.prune(pipeline)


def summary(row: dict) -> dict:
    """A run row as the app and the API show it."""
    detail = row.get("detail")
    if isinstance(detail, str) and detail:
        try:
            detail = json.loads(detail)
        except json.JSONDecodeError:
            detail = {}
    detail = detail if isinstance(detail, dict) else {}
    return {
        "run_id": str(row["id"]),
        "pipeline": row.get("target") or row.get("layer_name"),
        "status": row.get("status"),
        "started_at": row.get("started_at") or row.get("started"),
        "finished_at": row.get("finished") or row.get("completed_at"),
        "actor": row.get("actor"),
        "actor_role": row.get("actor_role"),
        "layers": detail.get("layers") or [],
        "reports": detail.get("reports") or [],
        "error": detail.get("error"),
    }


def get(pipeline: str, run_id: str) -> Optional[dict]:
    """One run's summary, or ``None`` when it is not a run of this pipeline."""
    from tracebi.state import get_run

    if not str(run_id).isdigit():
        return None
    row = get_run(int(run_id))
    if row is None or row.get("kind") != KIND or row.get("target") != pipeline:
        return None
    return summary(row)


def history(pipeline: str, limit: int = 10) -> list[dict]:
    """Recent runs of a pipeline, newest first."""
    from tracebi.state import list_runs

    return [summary(r) for r in list_runs(kind=KIND, target=pipeline, limit=limit)]


def read_log(pipeline: str, run_id: str, after: int = 0) -> Optional[dict]:
    """The log from byte *after*, and where to ask from next; ``None`` for an unknown run."""
    run = get(pipeline, run_id)
    if run is None:
        return None
    # The row first, then the file: see the module docstring.
    done = run["status"] != "running"
    try:
        text, nxt = runlog.read_from(runlog.log_path(pipeline, run_id), after)
        expired = False
    except FileNotFoundError:
        text, nxt, expired = "", after, done
    return {"run_id": run["run_id"], "status": run["status"], "done": done,
            "text": text, "next": nxt, "expired": expired}
