"""Adversarial tests for read-only GitHub Actions workflow-registry auditing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.audit_workflow_registry import (
    GitHubApiError,
    audit_workflow_registry,
    classify_workflows,
    collect_workflow_registry,
)


REPO = "ContextualWisdomLab/fast-mlsirm"
SHA = "a" * 40
OBSERVED = datetime(2026, 8, 12, 15, 0, tzinfo=UTC)


class FakeApi:
    """Deterministic endpoint fixture for registry-audit tests."""

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
        value = values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _repo_payload():
    return {"default_branch": "main"}


def _ref_payload(sha: str = SHA):
    return {"object": {"sha": sha}}


def _tree_payload(*paths: str):
    return {
        "truncated": False,
        "tree": [{"path": path, "type": "blob"} for path in paths],
    }


def _workflow(workflow_id: int, path: str, *, state: str = "active", name: str = "wf"):
    return {"id": workflow_id, "path": path, "state": state, "name": name}


def test_registry_pagination_collects_every_page_and_receipt():
    first = [_workflow(i, f".github/workflows/w{i}.yml") for i in range(100)]
    second = [
        _workflow(101, ".github/workflows/final.yml"),
        _workflow(102, ".github/workflows/last.yml"),
    ]
    api = FakeApi(
        {
            f"repos/{REPO}/actions/workflows?per_page=100&page=1": {
                "total_count": 102,
                "workflows": first,
            },
            f"repos/{REPO}/actions/workflows?per_page=100&page=2": {
                "total_count": 102,
                "workflows": second,
            },
        }
    )

    workflows, pagination = collect_workflow_registry(api, REPO)

    assert len(workflows) == 102
    assert pagination["complete"] is True
    assert pagination["total_count"] == 102
    assert pagination["pages"] == [
        {"page": 1, "count": 100},
        {"page": 2, "count": 2},
    ]


def test_registry_pagination_fails_closed_on_partial_empty_page():
    first = [_workflow(i, f".github/workflows/w{i}.yml") for i in range(100)]
    api = FakeApi(
        {
            f"repos/{REPO}/actions/workflows?per_page=100&page=1": {
                "total_count": 102,
                "workflows": first,
            },
            f"repos/{REPO}/actions/workflows?per_page=100&page=2": {
                "total_count": 102,
                "workflows": [],
            },
        }
    )

    with pytest.raises(RuntimeError, match="partial workflow registry pagination"):
        collect_workflow_registry(api, REPO)


def test_registry_pagination_rejects_duplicate_ids_even_when_payload_matches():
    """Page drift cannot hide missing identities behind duplicate registry IDs."""
    first = [_workflow(i, f".github/workflows/w{i}.yml") for i in range(100)]
    second = [first[-2], first[-1]]
    api = FakeApi(
        {
            f"repos/{REPO}/actions/workflows?per_page=100&page=1": {
                "total_count": 102,
                "workflows": first,
            },
            f"repos/{REPO}/actions/workflows?per_page=100&page=2": {
                "total_count": 102,
                "workflows": second,
            },
        }
    )

    with pytest.raises(RuntimeError, match="duplicate workflow id"):
        collect_workflow_registry(api, REPO)


def test_classification_uses_exact_paths_not_name_heuristics():
    present_snapshot = ".github/workflows/dev-legitimate-snapshot.yml"
    workflows = [
        _workflow(1, present_snapshot, name="Development Snapshot"),
        _workflow(2, ".github/workflows/Dev-Legitimate-Snapshot.yml"),
        _workflow(3, ".github%2Fworkflows%2Fencoded.yml"),
        _workflow(4, "dynamic/github-code-scanning/codeql", name="CodeQL"),
        _workflow(5, ".github/workflows/old.yml", state="disabled_manually"),
    ]

    records = classify_workflows(
        workflows,
        present_paths={present_snapshot},
        default_branch_sha=SHA,
        observed_at=OBSERVED,
    )
    by_id = {record["id"]: record for record in records}

    assert by_id[1]["classification"] == "present"
    assert by_id[2]["classification"] == "orphan"
    assert by_id[3]["classification"] == "unresolved"
    assert by_id[4]["classification"] == "dynamic"
    assert by_id[5]["classification"] == "disabled"


def test_classification_does_not_treat_unknown_state_as_disabled():
    """Only explicit disabled registry states are accepted as disabled evidence."""
    records = classify_workflows(
        [_workflow(9, ".github/workflows/unknown.yml", state="mystery_state")],
        present_paths=set(),
        default_branch_sha=SHA,
        observed_at=OBSERVED,
    )

    assert records[0]["classification"] == "unresolved"


def test_classification_marks_reused_id_and_duplicate_active_path_unresolved():
    workflows = [
        _workflow(7, ".github/workflows/a.yml"),
        _workflow(7, ".github/workflows/b.yml"),
        _workflow(8, ".github/workflows/a.yml"),
    ]

    records = classify_workflows(
        workflows,
        present_paths={".github/workflows/a.yml", ".github/workflows/b.yml"},
        default_branch_sha=SHA,
        observed_at=OBSERVED,
    )

    assert {record["classification"] for record in records} == {"unresolved"}
    assert all(record["integrity_conflict"] for record in records)


def test_audit_binds_tree_and_registry_to_unchanged_default_branch_sha():
    present = ".github/workflows/ci.yml"
    orphan = ".github/workflows/old-repair.yml"
    api = FakeApi(
        {
            f"repos/{REPO}": [_repo_payload(), _repo_payload()],
            f"repos/{REPO}/git/ref/heads/main": [_ref_payload(), _ref_payload()],
            f"repos/{REPO}/git/trees/{SHA}?recursive=1": _tree_payload(present),
            f"repos/{REPO}/actions/workflows?per_page=100&page=1": {
                "total_count": 2,
                "workflows": [_workflow(1, present), _workflow(2, orphan)],
            },
        }
    )

    audit = audit_workflow_registry(api, REPO, observed_at=OBSERVED)

    assert audit["status"] == "ok"
    assert audit["default_branch"] == "main"
    assert audit["end_default_branch"] == "main"
    assert audit["default_branch_sha"] == SHA
    assert audit["snapshot_stable"] is True
    assert audit["summary"]["active_orphan"] == 1
    assert audit["pagination"]["complete"] is True
    assert {record["default_branch_sha"] for record in audit["workflows"]} == {SHA}
    assert {record["observed_at"] for record in audit["workflows"]} == {
        OBSERVED.isoformat().replace("+00:00", "Z")
    }


def test_audit_fails_closed_when_default_branch_moves_mid_snapshot():
    moved = "b" * 40
    api = FakeApi(
        {
            f"repos/{REPO}": [_repo_payload(), _repo_payload()],
            f"repos/{REPO}/git/ref/heads/main": [_ref_payload(), _ref_payload(moved)],
            f"repos/{REPO}/git/trees/{SHA}?recursive=1": _tree_payload(
                ".github/workflows/ci.yml"
            ),
            f"repos/{REPO}/actions/workflows?per_page=100&page=1": {
                "total_count": 1,
                "workflows": [_workflow(1, ".github/workflows/ci.yml")],
            },
        }
    )

    audit = audit_workflow_registry(api, REPO, observed_at=OBSERVED)

    assert audit["status"] == "failed"
    assert audit["snapshot_stable"] is False
    assert audit["end_default_branch"] == "main"
    assert audit["end_default_branch_sha"] == moved
    assert audit["errors"] == ["default branch moved during workflow registry audit"]


def test_audit_rejects_truncated_git_tree_instead_of_inventing_orphans():
    api = FakeApi(
        {
            f"repos/{REPO}": _repo_payload(),
            f"repos/{REPO}/git/ref/heads/main": _ref_payload(),
            f"repos/{REPO}/git/trees/{SHA}?recursive=1": {
                "truncated": True,
                "tree": [],
            },
        }
    )

    with pytest.raises(RuntimeError, match="default-branch tree is truncated"):
        audit_workflow_registry(api, REPO, observed_at=OBSERVED)


@pytest.mark.parametrize("status", [403, 404, 500])
def test_audit_surfaces_permission_missing_and_server_failures(status):
    error = GitHubApiError(
        endpoint=f"repos/{REPO}",
        returncode=1,
        stderr=f"HTTP {status}: synthetic failure",
    )
    api = FakeApi({f"repos/{REPO}": error})

    with pytest.raises(GitHubApiError, match=f"HTTP {status}"):
        audit_workflow_registry(api, REPO, observed_at=OBSERVED)


def test_hourly_governance_publishes_workflow_registry_audit_read_only():
    """The hourly evidence loop observes registry drift without mutation authority."""
    workflow = (
        Path(__file__).parents[1]
        / ".github"
        / "workflows"
        / "hourly-pr-governance.yml"
    ).read_text(encoding="utf-8")

    assert "actions: read" in workflow
    assert "actions: write" not in workflow
    assert "contents: write" not in workflow
    assert "pull-requests: write" not in workflow
    assert "python scripts/audit_workflow_registry.py" in workflow
    assert "--repo ContextualWisdomLab/fast-mlsirm" in workflow
    assert "workflow_registry_audit.json" in workflow
    assert "scripts/audit_workflow_registry.py" in workflow
    assert "tests/test_workflow_registry_audit.py" in workflow
