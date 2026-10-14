"""Regression coverage for bounded GitHub CLI command duration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess


def _load_governance():
    """Load the governance script without requiring package installation."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_pr_queue_governance.py"
    spec = importlib.util.spec_from_file_location("build_pr_queue_governance_timeout", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_gh_json_timeout_fails_closed_without_retrying(monkeypatch) -> None:
    """A hung GitHub CLI call becomes stable error evidence after one attempt."""
    module = _load_governance()
    observed: list[tuple[list[str], float | None]] = []

    def fake_run(
        command,
        *,
        timeout_seconds,
        max_stdout_bytes,
        max_stderr_bytes,
        **kwargs,
    ):
        observed.append((list(command), timeout_seconds))
        raise subprocess.TimeoutExpired(command, timeout_seconds)

    monkeypatch.setattr(module, "run_bounded_capture", fake_run)

    payload, error = module._run_gh_json(
        ["gh", "api", "/rate_limit"],
        max_attempts=3,
        retry_sleep_seconds=0.0,
    )

    assert payload is None
    assert error == {
        "command": ["api", "/rate_limit"],
        "stderr": "command timed out after 60 seconds",
        "returncode": 124,
    }
    assert observed == [(["gh", "api", "/rate_limit"], 60)]
