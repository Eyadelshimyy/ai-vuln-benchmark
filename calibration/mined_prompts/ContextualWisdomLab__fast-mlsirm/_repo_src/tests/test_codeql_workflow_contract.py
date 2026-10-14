"""Regression contract for the required Actions-language CodeQL PR gate."""

from __future__ import annotations

from pathlib import Path
import json
import re

import pytest


WORKFLOW_PATH = Path(__file__).parents[1] / ".github" / "workflows" / "codeql.yml"


def _job_block(workflow: str, job_id: str, next_job_id: str | None = None) -> str:
    """Return one top-level workflow job block from the repository YAML text."""
    start = workflow.index(f"  {job_id}:\n")
    if next_job_id is None:
        return workflow[start:]
    end = workflow.index(f"  {next_job_id}:\n", start + 1)
    return workflow[start:end]


def test_actions_codeql_runs_on_pull_requests_while_python_stays_manual() -> None:
    """Keep the required Actions context reachable without duplicating Python CodeQL."""
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    trigger_block = workflow.split("\npermissions:\n", 1)[0]

    assert "  pull_request:\n" in trigger_block
    assert "  workflow_dispatch:\n" in trigger_block

    actions_job = _job_block(workflow, "analyze-actions", "analyze-python")
    python_job = _job_block(workflow, "analyze-python")

    assert "name: Analyze (actions)" in actions_job
    assert "languages: actions" in actions_job
    assert "\n    if:" not in actions_job

    assert "name: Analyze (python)" in python_job
    assert "languages: python" in python_job
    assert "if: github.event_name == 'workflow_dispatch'" in python_job


def test_advanced_jobs_do_not_upload_while_default_setup_is_enabled() -> None:
    """Run real CodeQL queries without competing with default setup SARIF ownership."""
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    actions_job = _job_block(workflow, "analyze-actions", "analyze-python")
    python_job = _job_block(workflow, "analyze-python")

    assert "upload: never" in actions_job
    assert "upload: never" in python_job


def test_codeql_workflow_keeps_pinned_actions_and_least_permissions() -> None:
    """The trigger repair must not loosen action pinning or workflow permissions."""
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read\n" in workflow
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in workflow
    assert "github/codeql-action/init@ff2f1c621b7f889edc0d3c761ac2e6a3f8cdb0dd" in workflow
    assert "github/codeql-action/analyze@ff2f1c621b7f889edc0d3c761ac2e6a3f8cdb0dd" in workflow


def test_self_hosted_diagnostics_require_the_protected_main_workflow() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    for job in (_job_block(workflow, "analyze-actions", "analyze-python"),
                _job_block(workflow, "analyze-python")):
        selector = next(line for line in job.splitlines() if line.startswith("    runs-on:"))
        assert "github.event_name == 'workflow_dispatch' &&" in selector
        assert "github.ref == 'refs/heads/main' &&" in selector
        assert "github.workflow_ref == 'ContextualWisdomLab/fast-mlsirm/.github/workflows/codeql.yml@refs/heads/main' &&" in selector
        assert '\"group\":\"CWL central CodeQL\"' in selector
        assert "|| '[\"self-hosted\",\"Linux\",\"X64\",\"cwlab-ci-isolated\"]'" in selector
        assert "if-no-files-found: error" in job
        assert "upload: never" in job


_MAIN_WORKFLOW = (
    "ContextualWisdomLab/fast-mlsirm/.github/workflows/codeql.yml@refs/heads/main"
)
_ISOLATED_LABELS = ["self-hosted", "Linux", "X64", "cwlab-ci-isolated"]
_TRUSTED_ROUTE = {
    "group": "CWL central CodeQL",
    "labels": ["self-hosted", "Linux", "X64"],
}


@pytest.mark.parametrize("job_id", ["analyze-actions", "analyze-python"])
@pytest.mark.parametrize(
    "event,ref,workflow_ref,trusted",
    [
        ("pull_request", "refs/pull/2314/merge", _MAIN_WORKFLOW, False),
        ("pull_request", "refs/heads/main", _MAIN_WORKFLOW, False),
        ("workflow_dispatch", "refs/heads/main", _MAIN_WORKFLOW, True),
        ("workflow_dispatch", "refs/heads/feature", _MAIN_WORKFLOW, False),
        ("workflow_dispatch", "refs/heads/main", "other/workflow@refs/heads/main", False),
        ("workflow_dispatch", "refs/tags/v1", _MAIN_WORKFLOW, False),
        ("push", "refs/heads/main", _MAIN_WORKFLOW, False),
    ],
)
def test_codeql_route_preserves_trusted_dispatch_and_isolates_fallback(
    job_id: str, event: str, ref: str, workflow_ref: str, trusted: bool
) -> None:
    """Model the exact retained selector grammar, not GitHub engine execution.

    A selector can prove requested routing only; registered eligible capacity,
    clean-per-job isolation and scanner execution require operator evidence.
    """
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    next_job = "analyze-python" if job_id == "analyze-actions" else None
    job = _job_block(workflow, job_id, next_job)
    selector = next(line.strip() for line in job.splitlines() if line.startswith("    runs-on:"))
    guard = (
        "github.event_name == 'workflow_dispatch' && "
        "github.ref == 'refs/heads/main' && "
        f"github.workflow_ref == '{_MAIN_WORKFLOW}'"
    )
    match = re.fullmatch(
        r"runs-on: \$\{\{ fromJSON\((.*?) && '([^']+)' \|\| '([^']+)'\) \}\}",
        selector,
    )
    assert match is not None, "unsupported runner-selector grammar"
    assert match.group(1) == guard, "protected dispatch guard changed"
    privileged = json.loads(match.group(2))
    fallback = json.loads(match.group(3))
    assert privileged == _TRUSTED_ROUTE
    allowed = (
        event.casefold() == "workflow_dispatch"
        and ref.casefold() == "refs/heads/main"
        and workflow_ref.casefold() == _MAIN_WORKFLOW.casefold()
    )
    assert allowed is trusted
    actual = privileged if allowed else fallback
    assert actual == (_TRUSTED_ROUTE if trusted else _ISOLATED_LABELS)


def test_codeql_explicitly_admits_ready_without_losing_default_pr_activities() -> None:
    """Admit Ready alongside the three default PR activities, and only those.

    Basis: GitHub (n.d.), Events that trigger workflows, pull_request section,
    activity-types table and default-activity note. The read section says that
    opened, synchronize and reopened are defaults; Ready needs explicit types.
    This source contract does not prove an actual hosted event or job executed.

    References:
        GitHub. (n.d.). Events that trigger workflows (pull_request section).
            GitHub Docs. https://docs.github.com/en/actions/reference/
            workflows-and-actions/events-that-trigger-workflows
    """
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    trigger_block = workflow.split("\npermissions:\n", 1)[0]
    assert trigger_block == (
        "name: CodeQL\n\non:\n  pull_request:\n"
        "    types: [opened, synchronize, reopened, ready_for_review]\n"
        "  workflow_dispatch:\n"
    ), "CodeQL must admit Ready and preserve all default PR activities"
