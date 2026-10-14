"""Regression tests for cumulative PR snapshot resource and completeness bounds."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_pr_queue_snapshot.py"


def _module():
    """Load the live capture helper without requiring ``scripts`` as a package."""
    spec = importlib.util.spec_from_file_location("capture_pr_queue_snapshot_budget", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(command: Sequence[str]) -> str:
    """Return the requested state from one ``gh pr list`` command."""
    return str(command[command.index("--state") + 1])


def _detail(number: int) -> dict[str, Any]:
    """Return one complete open-PR detail payload."""
    return {
        "number": number,
        "title": f"PR {number}",
        "body": "",
        "headRefName": f"feat/{number}",
        "headRefOid": f"{number:040x}",
        "baseRefName": "main",
        "isDraft": False,
        "mergeStateStatus": "CLEAN",
        "reviewDecision": "REVIEW_REQUIRED",
        "state": "OPEN",
        "updatedAt": "2026-08-15T00:00:00Z",
        "closedAt": None,
        "mergedAt": None,
        "url": f"https://github.com/o/r/pull/{number}",
        "labels": [],
        "files": [],
    }


def _runner(*, identities: list[dict[str, int]], detail_factory=_detail):
    """Return a deterministic live-capture runner for focused budget tests."""
    def run_json(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "open":
            return identities, None
        if list(command[1:3]) == ["pr", "view"]:
            return detail_factory(int(command[3])), None
        if list(command[1:3]) == ["pr", "list"]:
            return [], None
        if command[1] == "api":
            return {"sha": "a" * 40}, None
        raise AssertionError(command)

    return run_json


def test_capture_fails_closed_when_cumulative_enrichment_budget_expires() -> None:
    """Sequential enrichment stops while a bounded snapshot and error remain publishable."""
    module = _module()
    clock = iter((0.0, 1.0, 421.0))

    snapshot = module.capture_pr_queue_snapshot(
        "owner/repo",
        run_json=_runner(identities=[{"number": 1}, {"number": 2}]),
        monotonic=lambda: next(clock),
        capture_budget_seconds=420,
    )

    assert [item["number"] for item in snapshot["open_prs"]] == [1]
    assert snapshot["base_sha"] == ""
    assert len(snapshot["errors"]) == 1
    assert snapshot["errors"][0]["returncode"] == 124
    assert "cumulative capture budget" in snapshot["errors"][0]["stderr"]


def test_capture_fails_closed_if_budget_expires_before_base_identity() -> None:
    """Exact default-branch identity is never fetched after the live budget expires."""
    module = _module()
    clock = iter((0.0, 421.0))

    snapshot = module.capture_pr_queue_snapshot(
        "owner/repo",
        run_json=_runner(identities=[]),
        monotonic=lambda: next(clock),
        capture_budget_seconds=420,
    )

    assert snapshot["base_sha"] == ""
    assert len(snapshot["errors"]) == 1
    assert snapshot["errors"][0]["returncode"] == 124
    assert snapshot["errors"][0]["command"][0] == "api"


@pytest.mark.parametrize("budget", (0, -1, True))
def test_capture_rejects_nonpositive_or_boolean_budget(budget: object) -> None:
    """Invalid internal budgets fail before any GitHub request can start."""
    module = _module()
    with pytest.raises(ValueError, match="capture_budget_seconds must be positive"):
        module.capture_pr_queue_snapshot(
            "owner/repo",
            run_json=lambda _: pytest.fail("GitHub runner must not execute"),
            capture_budget_seconds=budget,
        )


def test_capture_rejects_incomplete_open_pr_detail_before_promotion() -> None:
    """A detail payload missing one governed classification field fails closed."""
    module = _module()

    def incomplete(number: int) -> dict[str, Any]:
        detail = _detail(number)
        del detail["reviewDecision"]
        return detail

    snapshot = module.capture_pr_queue_snapshot(
        "owner/repo",
        run_json=_runner(identities=[{"number": 1}], detail_factory=incomplete),
        monotonic=lambda: 0.0,
    )

    assert snapshot["open_prs"] == []
    assert snapshot["base_sha"] == "a" * 40
    assert len(snapshot["errors"]) == 1
    assert snapshot["errors"][0]["returncode"] == 65
    assert "reviewDecision" in snapshot["errors"][0]["stderr"]


def test_capture_rejects_nonlist_nested_detail_evidence() -> None:
    """Changed-file and label evidence must retain the list shape expected downstream."""
    module = _module()

    def malformed(number: int) -> dict[str, Any]:
        detail = _detail(number)
        detail["files"] = None
        return detail

    snapshot = module.capture_pr_queue_snapshot(
        "owner/repo",
        run_json=_runner(identities=[{"number": 1}], detail_factory=malformed),
        monotonic=lambda: 0.0,
    )

    assert snapshot["open_prs"] == []
    assert len(snapshot["errors"]) == 1
    assert "labels and files must be lists" in snapshot["errors"][0]["stderr"]
