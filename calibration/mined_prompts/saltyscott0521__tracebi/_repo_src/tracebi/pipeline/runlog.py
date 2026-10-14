"""A running log for a pipeline run: everything its steps print, with the time
each line arrived, in a file you can ``tail`` and the web app can follow.

``capture`` sends what the current thread prints (stdout and stderr) to a sink
as well as where it was going, so a transform's ``print`` and the runner's own
``[layer] Running...`` lines land in one place. Other threads are untouched: the
sink lives in a ContextVar, the same way ``tracebi.audit`` carries the actor.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from contextvars import ContextVar
from typing import Callable, Iterator

_SINKS: ContextVar[tuple] = ContextVar("tracebi_output_sinks", default=())
_ECHO: ContextVar[bool] = ContextVar("tracebi_output_echo", default=True)

_lock = threading.Lock()
_installed = 0


class _Router:
    """Stands in for sys.stdout / sys.stderr: feeds this thread's sinks, then
    the stream it replaced (unless the capture asked for silence)."""

    def __init__(self, base) -> None:
        self._base = base

    def write(self, text: str) -> int:
        for sink in _SINKS.get():
            sink(text)
        if _ECHO.get():
            return self._base.write(text)
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        if _ECHO.get():
            self._base.flush()

    def isatty(self) -> bool:
        return False if _SINKS.get() else self._base.isatty()

    def __getattr__(self, name):
        return getattr(self._base, name)


def _install() -> None:
    global _installed
    with _lock:
        # Whenever a capture begins, not only the first: something else (a
        # redirect, pytest's capture) may have swapped the stream since.
        if not isinstance(sys.stdout, _Router):
            sys.stdout = _Router(sys.stdout)
        if not isinstance(sys.stderr, _Router):
            sys.stderr = _Router(sys.stderr)
        _installed += 1


def _release() -> None:
    global _installed
    with _lock:
        _installed -= 1
        if _installed == 0:
            # Put back what was there; a stream someone else swapped in meanwhile stays.
            if isinstance(sys.stdout, _Router):
                sys.stdout = sys.stdout._base
            if isinstance(sys.stderr, _Router):
                sys.stderr = sys.stderr._base


@contextlib.contextmanager
def capture(sink: Callable[[str], object], *, echo: bool = True) -> Iterator[None]:
    """Send what this thread prints to ``sink(text)``.

    ``echo=False`` also keeps it off the terminal, for output that was never
    shown there (a step's captured output). Captures nest: an outer sink still
    sees what an inner one captures.
    """
    _install()
    sinks = _SINKS.set(_SINKS.get() + (sink,))
    shown = _ECHO.set(echo and _ECHO.get())
    try:
        yield
    finally:
        _ECHO.reset(shown)
        _SINKS.reset(sinks)
        _release()


def _clock(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes):02d}:{rest:04.1f}"


class RunLog:
    """An append-only log file. Each complete line is stamped with the time
    since the run began (``mm:ss.s``), so the file shows where the time went
    and reads the same in any timezone."""

    #: A chatty script cannot fill the disk: past this, the log says so and stops.
    MAX_BYTES = 2_000_000

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._file = open(path, "w", encoding="utf-8")
        self._start = time.monotonic()
        self._partial = ""
        self._bytes = 0
        self._full = False
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        """Take printed text; whole lines are written, a partial one is held."""
        with self._lock:
            self._partial += text
            *lines, self._partial = self._partial.split("\n")
            for line in lines:
                self._line(line)

    def line(self, text: str) -> None:
        with self._lock:
            self._line(text)

    def _line(self, text: str) -> None:
        if self._file is None or self._full:
            return
        text = text.rsplit("\r", 1)[-1].rstrip()      # a progress bar keeps its last frame
        stamped = f"{_clock(time.monotonic() - self._start)}  {text}\n"
        if self._bytes + len(stamped) > self.MAX_BYTES:
            self._full = True
            stamped = f"{_clock(time.monotonic() - self._start)}  [log full: output stops here]\n"
        self._file.write(stamped)
        self._file.flush()
        self._bytes += len(stamped)

    def close(self) -> None:
        with self._lock:
            if self._partial:
                self._line(self._partial)
                self._partial = ""
            if self._file is not None:
                self._file.close()
                self._file = None


def log_path(pipeline: str, run_id) -> str:
    """Where a pipeline run's log lives, from names alone (never from a stored path)."""
    return os.path.join("data", "logs", "pipelines", pipeline, f"{int(run_id)}.log")


def read_from(path: str, offset: int = 0) -> tuple[str, int]:
    """The complete lines at or after byte *offset*, and the offset to ask from next.

    Raises ``FileNotFoundError`` when the log is gone.
    """
    with open(path, "rb") as f:
        f.seek(max(int(offset), 0))
        data = f.read()
    end = data.rfind(b"\n") + 1                 # never hand back half a line
    return data[:end].decode("utf-8", errors="replace"), max(int(offset), 0) + end


def prune(pipeline: str, keep: int = 30) -> None:
    """Delete all but the newest *keep* logs of a pipeline."""
    folder = os.path.dirname(log_path(pipeline, 0))
    if not os.path.isdir(folder):
        return
    logs = sorted((n for n in os.listdir(folder) if n.endswith(".log") and n[:-4].isdigit()),
                  key=lambda n: int(n[:-4]), reverse=True)
    for name in logs[keep:]:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(folder, name))
