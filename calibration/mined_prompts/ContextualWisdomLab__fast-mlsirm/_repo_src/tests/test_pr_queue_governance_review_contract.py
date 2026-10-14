"""Review regressions for the bounded GitHub PR governance snapshot."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def _load_governance():
    """Load the governance builder as an isolated module for contract tests."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_pr_queue_governance.py"
    spec = importlib.util.spec_from_file_location("build_pr_queue_governance_review", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pr_list_command_binds_repository_explicitly() -> None:
    """History queries must never fall back to the caller's default repository."""
    module = _load_governance()
    repo = "ContextualWisdomLab/fast-mlsirm"
    command = module._pr_list_command(
        repo,
        state="all",
        limit=module._HISTORY_PR_LIST_LIMIT,
        fields=module._HISTORY_PR_JSON_FIELDS,
    )

    assert "--repo" in command
    assert command[command.index("--repo") + 1] == repo


@pytest.mark.parametrize("status", [502, 503, 504])
def test_run_gh_json_recovers_from_each_approved_transient_status(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    """Every allowlisted transient gateway status gets the same bounded retry."""
    module = _load_governance()
    sleeps: list[float] = []
    responses = iter(
        [
            subprocess.CompletedProcess(
                ["gh", "pr", "list"],
                1,
                "",
                f"HTTP {status}: transient gateway failure",
            ),
            subprocess.CompletedProcess(
                ["gh", "pr", "list"],
                0,
                json.dumps([{"number": status}]),
                "",
            ),
        ]
    )
    monkeypatch.setattr(
        module,
        "run_bounded_capture",
        lambda *args, **kwargs: next(responses),
    )
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))

    payload, error = module._run_gh_json(
        ["gh", "pr", "list"],
        max_attempts=3,
        retry_sleep_seconds=0.01,
    )

    assert error is None
    assert payload == [{"number": status}]
    assert sleeps == [0.01]


@pytest.mark.parametrize("status", [502, 503, 504])
def test_run_gh_json_exhausts_exact_retry_budget_for_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    """Persistent transient failures stop after the package-owned attempt ceiling."""
    module = _load_governance()
    calls = 0
    sleeps: list[float] = []

    def fail_transiently(*args, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            ["gh", "pr", "list"],
            1,
            "",
            f"HTTP {status}: transient gateway failure",
        )

    monkeypatch.setattr(module, "run_bounded_capture", fail_transiently)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))

    payload, error = module._run_gh_json(
        ["gh", "pr", "list"],
        max_attempts=module._GH_JSON_MAX_ATTEMPTS,
        retry_sleep_seconds=0.01,
    )

    assert payload is None
    assert error is not None
    assert error["returncode"] == 1
    assert calls == module._GH_JSON_MAX_ATTEMPTS
    assert sleeps == [0.01] * (module._GH_JSON_MAX_ATTEMPTS - 1)
