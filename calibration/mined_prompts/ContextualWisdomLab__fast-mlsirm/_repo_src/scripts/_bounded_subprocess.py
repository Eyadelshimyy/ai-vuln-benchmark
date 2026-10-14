"""Bound subprocess stdout/stderr in memory while preserving a hard deadline."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

_READ_CHUNK_BYTES = 64 * 1024
_DATA_ERROR_RETURN_CODE = 65
_PROCESS_REAP_TIMEOUT_SECONDS = 5.0


class BoundedSubprocessOutputError(RuntimeError):
    """Raised when a captured subprocess stream exceeds its configured limit."""

    def __init__(self, stream: str, limit_bytes: int) -> None:
        self.stream = stream
        self.limit_bytes = limit_bytes
        super().__init__(f"{stream} exceeded bounded capture limit of {limit_bytes} bytes")


class BoundedSubprocessDecodeError(UnicodeError):
    """Describe machine-readable subprocess stdout that is not valid UTF-8."""

    def __init__(self, stream: str) -> None:
        self.stream = stream
        super().__init__(f"{stream} was not valid UTF-8")


def _drain_bounded(
    stream: object,
    *,
    limit_bytes: int,
    buffer: bytearray,
    overflow: threading.Event,
) -> None:
    """Drain one binary pipe without retaining more than ``limit_bytes + 1`` bytes.

    ``BufferedReader.read`` may wait for the requested chunk size even when a
    smaller payload is already available. ``read1`` observes the pipe promptly,
    which keeps overflow detection independent from a descendant holding the
    write end open.
    """
    if hasattr(stream, "read1"):
        read = stream.read1  # type: ignore[attr-defined]
    else:
        read = stream.read  # type: ignore[attr-defined]
    while True:
        try:
            chunk = read(_READ_CHUNK_BYTES)
        except (OSError, ValueError):
            return
        if not chunk:
            return
        remaining = (limit_bytes + 1) - len(buffer)
        if remaining > 0:
            buffer.extend(chunk[:remaining])
        if len(buffer) > limit_bytes:
            overflow.set()


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    *,
    terminate_descendants: bool = False,
) -> None:
    """Terminate and bounded-reap the owned process tree.

    A live process group is terminated after the direct child exits only when
    a capture reader still proves that a descendant owns the pipe. The default
    keeps repeated cleanup from signalling a reaped process group.
    """
    if process.poll() is None or (terminate_descendants and os.name == "posix"):
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        else:
            try:
                process.kill()
            except OSError:
                pass
    try:
        process.wait(timeout=_PROCESS_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # The caller still has a hard deadline; leave pipe cleanup to the
        # existing close path if a hostile child ignores the bounded reap.
        pass


def _close_capture_pipes(process: subprocess.Popen[bytes]) -> None:
    """Close parent-side capture pipes to unblock any remaining daemon reader."""
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError):
                pass


def _remaining(deadline: float) -> float:
    """Return non-negative seconds remaining before one absolute deadline."""
    return max(0.0, deadline - time.monotonic())


def run_bounded_capture(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` with one hard deadline and bounded output capture.

    Stdout and stderr are drained concurrently so neither pipe can deadlock the
    child. POSIX commands run in a dedicated session so timeout/overflow cleanup
    can terminate descendants that inherited a capture pipe. Process reaping
    and reader joins share the original deadline rather than extending it.
    Machine-readable stdout is decoded strictly as UTF-8. Malformed stdout
    becomes a stable data-error result rather than replacement-decoded content;
    diagnostic stderr alone uses replacement decoding.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_stdout_bytes < 0 or max_stderr_bytes < 0:
        raise ValueError("output limits must be non-negative")
    if not command:
        raise ValueError("command must not be empty")

    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        start_new_session=os.name == "posix",
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RuntimeError("bounded capture requires stdout and stderr pipes")

    stdout = bytearray()
    stderr = bytearray()
    stdout_overflow = threading.Event()
    stderr_overflow = threading.Event()
    readers = [
        threading.Thread(
            target=_drain_bounded,
            kwargs={
                "stream": process.stdout,
                "limit_bytes": max_stdout_bytes,
                "buffer": stdout,
                "overflow": stdout_overflow,
            },
            daemon=True,
        ),
        threading.Thread(
            target=_drain_bounded,
            kwargs={
                "stream": process.stderr,
                "limit_bytes": max_stderr_bytes,
                "buffer": stderr,
                "overflow": stderr_overflow,
            },
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    overflowed = False
    while process.poll() is None:
        if stdout_overflow.is_set() or stderr_overflow.is_set():
            overflowed = True
            _terminate_process_tree(process)
            break
        remaining = _remaining(deadline)
        if remaining <= 0.0:
            timed_out = True
            _terminate_process_tree(process)
            break
        time.sleep(min(0.01, remaining))

    if process.poll() is None:
        try:
            process.wait(timeout=_remaining(deadline))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(process)

    for reader in readers:
        reader.join(timeout=_remaining(deadline))
        if reader.is_alive():
            overflowed = overflowed or stdout_overflow.is_set() or stderr_overflow.is_set()
            if not overflowed:
                timed_out = True
            _terminate_process_tree(process, terminate_descendants=True)
            _close_capture_pipes(process)
            break

    if timed_out:
        _terminate_process_tree(process)
        _close_capture_pipes(process)
        raise subprocess.TimeoutExpired(list(command), timeout_seconds)
    if stdout_overflow.is_set():
        _terminate_process_tree(process)
        _close_capture_pipes(process)
        raise BoundedSubprocessOutputError("stdout", max_stdout_bytes)
    if stderr_overflow.is_set():
        _terminate_process_tree(process)
        _close_capture_pipes(process)
        raise BoundedSubprocessOutputError("stderr", max_stderr_bytes)

    _close_capture_pipes(process)
    stderr_text = stderr.decode("utf-8", errors="replace")
    try:
        stdout_text = stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        decode_error = BoundedSubprocessDecodeError("stdout")
        diagnostic = stderr_text.strip()
        if diagnostic:
            diagnostic = f"{diagnostic}\n{decode_error}"
        else:
            diagnostic = str(decode_error)
        return subprocess.CompletedProcess(
            list(command),
            _DATA_ERROR_RETURN_CODE,
            "",
            diagnostic,
        )
    return subprocess.CompletedProcess(
        list(command),
        process.returncode,
        stdout_text,
        stderr_text,
    )
