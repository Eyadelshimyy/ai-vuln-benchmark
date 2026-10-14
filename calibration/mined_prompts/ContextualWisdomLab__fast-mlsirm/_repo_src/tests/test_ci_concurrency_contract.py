"""Contract tests for pull-request CI concurrency."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"
_EXPECTED_GROUP = (
    "${{ github.workflow }}-${{ github.repository }}-"
    "${{ github.event.pull_request.number || github.run_id }}"
)


def _scalar(value: str) -> Any:
    """Parse the strict scalar subset used by the concurrency mapping."""
    normalized = value.strip()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    if not normalized:
        raise AssertionError("concurrency mapping values must not be empty")
    return normalized


def _top_level_mapping(section: str) -> dict[str, Any]:
    """Parse direct scalar children of one top-level workflow mapping."""
    lines = _WORKFLOW.read_text(encoding="utf-8").splitlines()
    marker = f"{section}:"
    try:
        start = next(index for index, line in enumerate(lines) if line == marker) + 1
    except StopIteration as error:
        raise AssertionError(f"missing top-level workflow mapping: {section}") from error

    end = len(lines)
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[index]) - len(lines[index].lstrip())
        if indent == 0:
            end = index
            break

    mapping: dict[str, Any] = {}
    for index in range(start, end):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent != 2:
            raise AssertionError("concurrency must contain direct two-space scalar keys")
        key, separator, value = stripped.partition(":")
        if not separator or key in mapping:
            raise AssertionError("concurrency must contain unique scalar keys")
        mapping[key] = _scalar(value)
    return mapping


def test_ci_cancels_superseded_runs_for_the_same_pull_request():
    """A newer PR head invalidates queued or running evidence for the old head."""
    concurrency = _top_level_mapping("concurrency")
    assert concurrency == {
        "group": _EXPECTED_GROUP,
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }


def test_ci_push_runs_remain_independent():
    """Main/develop push evidence cannot replace pending release evidence."""
    group = _top_level_mapping("concurrency")["group"]
    assert group == _EXPECTED_GROUP
    assert "github.repository" in group
    assert "github.event.pull_request.head.sha" not in group
    assert "github.run_id" in group
    assert _top_level_mapping("concurrency")["cancel-in-progress"] == (
        "${{ github.event_name == 'pull_request' }}"
    )


def test_ci_skips_expensive_jobs_for_inactive_pull_requests():
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    assert (
        "types: [opened, synchronize, reopened, ready_for_review, "
        "converted_to_draft, closed]"
    ) in workflow
    for job_name in (
        "python-matrix",
        "python",
        "rust",
        "gpu-smoke",
        "fuzz",
        "package",
    ):
        match = re.search(
            rf"(?ms)^  {re.escape(job_name)}:\n(.*?)(?=^  \S|\Z)", workflow
        )
        assert match is not None
        job = match.group(1)
        assert "!github.event.pull_request.draft" in job
        assert "github.event.action != 'closed'" in job
