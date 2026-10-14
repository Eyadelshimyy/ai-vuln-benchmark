"""Fresh-state and retry contracts for the workflow-registry audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime

import pytest

from scripts.audit_workflow_registry import (
    _repo_slug,
    _run_gh_api,
    audit_workflow_registry,
)


REPO = "ContextualWisdomLab/fast-mlsirm"
SHA_A = "a" * 40
SHA_B = "b" * 40
OBSERVED = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)


class FakeApi:
    """Return queued deterministic responses for exact REST endpoints."""

    def __init__(self, responses):
        self.responses = {
            endpoint: list(values) if isinstance(values, list) else [values]
            for endpoint, values in responses.items()
        }
        self.calls: list[str] = []

    def __call__(self, endpoint: str):
        self.calls.append(endpoint)
        values = self.responses.get(endpoint)
        if not values:
            raise AssertionError(f"unexpected endpoint: {endpoint}")
        return values.pop(0)


def _ref_payload(sha: str):
    return {"object": {"sha": sha}}


def _tree_payload(path: str):
    return {"truncated": False, "tree": [{"path": path, "type": "blob"}]}


def _registry_payload(path: str):
    return {
        "total_count": 1,
        "workflows": [{"id": 1, "path": path, "state": "active", "name": "CI"}],
    }


def test_audit_retries_whole_snapshot_after_default_branch_movement():
    path = ".github/workflows/ci.yml"
    endpoint = f"repos/{REPO}/actions/workflows?per_page=100&page=1"
    repo_endpoint = f"repos/{REPO}"
    api = FakeApi(
        {
            repo_endpoint: [
                {"default_branch": "main"},
                {"default_branch": "main"},
                {"default_branch": "main"},
            ],
            f"repos/{REPO}/git/ref/heads/main": [
                _ref_payload(SHA_A),
                _ref_payload(SHA_B),
                _ref_payload(SHA_B),
                _ref_payload(SHA_B),
            ],
            f"repos/{REPO}/git/trees/{SHA_A}?recursive=1": _tree_payload(path),
            f"repos/{REPO}/git/trees/{SHA_B}?recursive=1": _tree_payload(path),
            endpoint: [_registry_payload(path), _registry_payload(path)],
        }
    )

    audit = audit_workflow_registry(
        api,
        REPO,
        observed_at=OBSERVED,
        max_snapshot_attempts=2,
    )

    assert audit["status"] == "ok"
    assert audit["snapshot_stable"] is True
    assert audit["default_branch"] == "main"
    assert audit["end_default_branch"] == "main"
    assert audit["default_branch_sha"] == SHA_B
    assert audit["snapshot_attempts"] == [
        {"attempt": 1, "start_sha": SHA_A, "end_sha": SHA_B, "stable": False},
        {"attempt": 2, "start_sha": SHA_B, "end_sha": SHA_B, "stable": True},
    ]
    assert api.calls.count(repo_endpoint) == 3
    assert api.calls.count(endpoint) == 2


def test_run_gh_api_retries_timeout_before_succeeding(monkeypatch):
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=20)
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps({"ok": True}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _run_gh_api("repos/example/project", max_attempts=2, retry_sleep_seconds=0) == {
        "ok": True
    }
    assert calls == 2


@pytest.mark.parametrize(
    "value",
    [
        "owner/repo/extra",
        "owner/repo?ref=main",
        "owner/repo#fragment",
        "/repo",
        "owner/",
        "owner repo/project",
    ],
)
def test_repo_slug_rejects_ambiguous_or_path_injecting_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="owner/name"):
        _repo_slug(value)


def test_repo_slug_accepts_standard_owner_name():
    assert _repo_slug("ContextualWisdomLab/fast-mlsirm") == REPO
