"""Pipeline runs in the background.

A run is recorded the same way wherever it starts (see
``tracebi.pipeline.run_record``); what is web-specific is here: a worker pool
that runs it off the request, and joining a second click to the run already going.
"""

from __future__ import annotations

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from tracebi.pipeline import run_record
from tracebi.state import get_run, install_extra

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tracebi-pipeline")
_active: dict[str, int] = {}                 # pipeline -> the run in progress in this process
_active_lock = threading.Lock()


def start(pipeline: str, runner, order: list[str]) -> dict:
    """Begin running *order* (layer names) of *runner*; the row exists on return.

    A pipeline already running in this process returns that run instead of
    starting a second (a double click, or two tabs).
    """
    with _active_lock:
        if pipeline in _active:
            with install_extra("web"):
                current = get_run(_active[pipeline])
            if current["status"] == "running":
                return {**run_record.summary(current), "already_running": True}
            del _active[pipeline]           # finished; its worker has not cleared it yet
        with install_extra("web"):
            run_id = run_record.begin(pipeline, order)
            row = get_run(run_id)
        _active[pipeline] = run_id
    # Carry the audit actor into the worker thread, so each layer's run is
    # attributed to whoever started this one.
    _pool.submit(contextvars.copy_context().run, _execute, pipeline, run_id, runner, order)
    return {**run_record.summary(row), "already_running": False}


def _execute(pipeline: str, run_id: int, runner, order: list[str]) -> None:
    try:
        with install_extra("web"), run_record.recording(pipeline, run_id, order):
            for name in order:
                runner.execute_layer(name)
    except Exception:  # noqa: BLE001 — recorded as the run's outcome; a failed run is the payload
        pass
    finally:
        with _active_lock:
            _active.pop(pipeline, None)


def get(pipeline: str, run_id: str) -> Optional[dict]:
    """One run's summary, or ``None`` when it is not a run of this pipeline."""
    with install_extra("web"):
        return run_record.get(pipeline, run_id)


def history(pipeline: str, limit: int = 10) -> list[dict]:
    """Recent runs of a pipeline, newest first."""
    with install_extra("web"):
        return run_record.history(pipeline, limit)


def read_log(pipeline: str, run_id: str, after: int = 0) -> Optional[dict]:
    """The log from byte *after*, and where to ask from next; ``None`` for an unknown run."""
    with install_extra("web"):
        return run_record.read_log(pipeline, run_id, after)
