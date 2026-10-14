"""Regression tests for workflow-registry review findings."""

from __future__ import annotations

import pytest

from scripts.audit_workflow_registry import (
    _run_gh_api,
    audit_workflow_registry,
    collect_workflow_registry,
)

REPO = "ContextualWisdomLab/fast-mlsirm"
SHA = "a" * 40


class FakeApi:
    def __init__(self, responses):
        self.responses = {
            endpoint: list(values) if isinstance(values, list) else [values]
            for endpoint, values in responses.items()
        }

    def __call__(self, endpoint: str):
        values = self.responses.get(endpoint)
        if not values:
            raise AssertionError(f"unexpected endpoint: {endpoint}")
        return values.pop(0)


def _ref_payload(sha: str = SHA):
    return {"object": {"sha": sha}}


def _tree_payload(path: str):
    return {"truncated": False, "tree": [{"path": path, "type": "blob"}]}


def _registry_payload(path: str):
    return {
        "total_count": 1,
        "workflows": [{"id": 1, "path": path, "state": "active", "name": "CI"}],
    }


@pytest.mark.parametrize("value", [None, 1.5, float("inf"), object()])
def test_run_gh_api_rejects_invalid_max_attempts(value):
    with pytest.raises(ValueError, match="max_attempts must be an integer"):
        _run_gh_api("repos/example/project", max_attempts=value, retry_sleep_seconds=0)


@pytest.mark.parametrize("value", [None, 1.5, float("inf"), object()])
def test_collect_registry_rejects_invalid_per_page(value):
    def should_not_fetch(_endpoint: str):
        raise AssertionError("invalid pagination controls must fail before REST access")

    with pytest.raises(ValueError, match="per_page must be an integer"):
        collect_workflow_registry(should_not_fetch, REPO, per_page=value)


@pytest.mark.parametrize("value", [None, 1.5, float("inf"), object()])
def test_audit_rejects_invalid_snapshot_attempts(value):
    def should_not_fetch(_endpoint: str):
        raise AssertionError("invalid snapshot controls must fail before REST access")

    with pytest.raises(ValueError, match="max_snapshot_attempts must be an integer"):
        audit_workflow_registry(should_not_fetch, REPO, max_snapshot_attempts=value)


def test_audit_rejects_default_branch_rename_with_unchanged_old_ref():
    path = ".github/workflows/ci.yml"
    registry_endpoint = f"repos/{REPO}/actions/workflows?per_page=100&page=1"
    api = FakeApi(
        {
            f"repos/{REPO}": [
                {"default_branch": "main"},
                {"default_branch": "release"},
            ],
            f"repos/{REPO}/git/ref/heads/main": [_ref_payload(), _ref_payload()],
            f"repos/{REPO}/git/trees/{SHA}?recursive=1": _tree_payload(path),
            registry_endpoint: _registry_payload(path),
        }
    )

    audit = audit_workflow_registry(api, REPO)

    assert audit["status"] == "failed"
    assert audit["snapshot_stable"] is False
    assert audit["default_branch"] == "main"
    assert audit["end_default_branch"] == "release"
    assert "default branch changed during workflow registry audit" in audit["errors"]
