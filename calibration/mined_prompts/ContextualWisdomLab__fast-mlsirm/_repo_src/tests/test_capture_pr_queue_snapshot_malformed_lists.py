"""Regression coverage for malformed PR queue list evidence."""

from __future__ import annotations

from scripts import capture_pr_queue_snapshot as capture


REPO = "ContextualWisdomLab/fast-mlsirm"
MAIN_SHA = "0" * 40


def _detail(number: int) -> dict[str, object]:
    return {
        "number": number,
        "title": "bounded queue evidence",
        "body": "",
        "headRefName": "feature/bounded-queue",
        "headRefOid": "1" * 40,
        "baseRefName": "main",
        "isDraft": False,
        "mergeStateStatus": "CLEAN",
        "reviewDecision": "",
        "state": "OPEN",
        "updatedAt": "2026-08-24T00:00:00Z",
        "closedAt": None,
        "mergedAt": None,
        "url": f"https://github.com/{REPO}/pull/{number}",
        "labels": [],
        "files": [],
    }


def _runner(*, identities: object, history: object):
    def run(command):
        if command[1:3] == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if command[1:3] == ["pr", "list"]:
            state = command[command.index("--state") + 1]
            return (identities if state == "open" else history), None
        if command[1:3] == ["pr", "view"]:
            return _detail(int(command[3])), None
        if command[1] == "api":
            return {"sha": MAIN_SHA}, None
        raise AssertionError(f"unexpected command: {command}")

    return run


def test_mixed_open_identity_list_fails_closed_and_preserves_raw_count():
    snapshot = capture.capture_pr_queue_snapshot(
        REPO,
        run_json=_runner(
            identities=[{"number": 11}, "malformed"],
            history=[],
        ),
    )

    assert snapshot["open_pr_identity_count"] == 2
    assert [record["number"] for record in snapshot["open_prs"]] == [11]
    assert any(
        "open PR identity payload contained non-object entries" in error["stderr"]
        for error in snapshot["errors"]
    )


def test_mixed_history_list_fails_closed_without_dropping_valid_history():
    valid_history = {"number": 9, "state": "MERGED"}
    snapshot = capture.capture_pr_queue_snapshot(
        REPO,
        run_json=_runner(
            identities=[],
            history=[valid_history, "malformed"],
        ),
    )

    assert snapshot["pr_history"] == [valid_history]
    assert any(
        "PR history payload contained non-object entries" in error["stderr"]
        for error in snapshot["errors"]
    )
