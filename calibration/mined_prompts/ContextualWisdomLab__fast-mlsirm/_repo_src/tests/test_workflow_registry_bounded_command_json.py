"""Regression for bounded GitHub CLI JSON in workflow-registry auditing."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_workflow_registry.py"


def _load_script() -> ModuleType:
    """Load the workflow-registry auditor under a test-local module identity."""
    module_name = "bounded_workflow_registry_command_json"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _overdeep_json_object() -> str:
    """Return valid JSON whose nesting exceeds the repository depth contract."""
    value: object = 0
    for _ in range(129):
        value = [value]
    return json.dumps({"payload": value})


def test_successful_gh_output_still_obeys_bounded_json_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-exit gh command cannot bypass command-output structural limits."""
    module = _load_script()
    completed = subprocess.CompletedProcess(
        args=["gh", "api"],
        returncode=0,
        stdout=_overdeep_json_object(),
        stderr="",
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: completed)

    with pytest.raises(module.GitHubApiError) as captured:
        module._run_gh_api(
            "/repos/example/project/actions/workflows",
            max_attempts=1,
            retry_sleep_seconds=0,
        )

    assert captured.value.stderr == "GitHub API returned invalid JSON"
    assert isinstance(captured.value.__cause__, ValueError)
