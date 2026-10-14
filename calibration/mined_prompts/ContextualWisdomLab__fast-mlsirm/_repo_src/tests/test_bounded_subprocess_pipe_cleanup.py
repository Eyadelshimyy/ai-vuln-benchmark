"""Regression coverage for successful bounded subprocess pipe cleanup."""

from __future__ import annotations

import sys

import pytest

from scripts import _bounded_subprocess as bounded


def test_successful_bounded_capture_closes_parent_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal completion must close the parent's stdout and stderr descriptors."""
    closed_processes: list[object] = []
    real_close = bounded._close_capture_pipes

    def record_close(process: object) -> None:
        closed_processes.append(process)
        real_close(process)  # type: ignore[arg-type]

    monkeypatch.setattr(bounded, "_close_capture_pipes", record_close)

    completed = bounded.run_bounded_capture(
        [sys.executable, "-c", "print('ok')"],
        timeout_seconds=5,
        max_stdout_bytes=1024,
        max_stderr_bytes=1024,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "ok"
    assert len(closed_processes) == 1
    process = closed_processes[0]
    assert process.stdout.closed  # type: ignore[attr-defined]
    assert process.stderr.closed  # type: ignore[attr-defined]
