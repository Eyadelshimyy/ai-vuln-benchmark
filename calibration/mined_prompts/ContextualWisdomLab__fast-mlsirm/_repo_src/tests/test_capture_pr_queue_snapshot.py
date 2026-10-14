"""Tests for bounded split-query PR queue snapshot capture."""

from __future__ import annotations

import importlib.util
import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_pr_queue_snapshot.py"


def _module():
    """Load the capture script without requiring ``scripts`` to be a package."""
    spec = importlib.util.spec_from_file_location("capture_pr_queue_snapshot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detail(number: int, *, state: str = "OPEN") -> dict[str, Any]:
    """Return one complete PR detail payload for focused capture tests."""
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
        "state": state,
        "updatedAt": "2026-08-15T00:00:00Z",
        "closedAt": None,
        "mergedAt": None,
        "url": f"https://github.com/o/r/pull/{number}",
        "labels": [{"name": "ops"}],
        "files": [{"path": f"file-{number}.py"}],
    }


def _state(command: Sequence[str]) -> str:
    """Return the state argument from one ``gh pr list`` command."""
    return str(command[command.index("--state") + 1])


def test_capture_splits_identity_listing_from_nested_pr_enrichment():
    """Large nested data is fetched one PR at a time, not in the list query."""
    module = _module()
    commands: list[list[str]] = []

    def run_json(command: Sequence[str]):
        commands.append(list(command))
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "open":
            fields = str(command[command.index("--json") + 1]).split(",")
            assert fields == ["number"]
            assert "files" not in fields
            return [{"number": 11}, {"number": 12}], None
        if list(command[1:3]) == ["pr", "view"]:
            fields = str(command[command.index("--json") + 1]).split(",")
            assert "files" in fields
            assert "labels" in fields
            return _detail(int(command[3])), None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "all":
            return [_detail(11), _detail(12)], None
        if command[1] == "api":
            return {"sha": "A" * 40}, None
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=run_json)

    assert snapshot["errors"] == []
    assert snapshot["base_sha"] == "a" * 40
    assert [item["number"] for item in snapshot["open_prs"]] == [11, 12]
    assert snapshot["open_pr_identity_count"] == 2
    assert sum(command[1:3] == ["pr", "view"] for command in commands) == 2


def test_capture_excludes_pr_that_closes_between_identity_and_detail_queries():
    """A queue race cannot reintroduce a PR that is no longer open."""
    module = _module()

    def run_json(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "open":
            return [{"number": 7}], None
        if list(command[1:3]) == ["pr", "view"]:
            return _detail(7, state="CLOSED"), None
        if list(command[1:3]) == ["pr", "list"]:
            return [], None
        if command[1] == "api":
            return {"sha": "b" * 40}, None
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=run_json)

    assert snapshot["open_prs"] == []
    assert snapshot["errors"] == []
    assert snapshot["open_pr_identity_count"] == 1


def test_capture_fails_closed_above_supported_open_pr_cap_without_enrichment():
    """A truncated queue is rejected before any misleading detail snapshot exists."""
    module = _module()
    detail_calls: list[list[str]] = []

    def run_json(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "open":
            return [{"number": number} for number in range(1, 102)], None
        if list(command[1:3]) == ["pr", "view"]:
            detail_calls.append(list(command))
            return _detail(int(command[3])), None
        if list(command[1:3]) == ["pr", "list"]:
            return [], None
        if command[1] == "api":
            return {"sha": "c" * 40}, None
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=run_json)

    assert snapshot["open_prs"] == []
    assert detail_calls == []
    assert snapshot["open_pr_identity_count"] == 101
    assert any("exceeds supported cap 100" in error["stderr"] for error in snapshot["errors"])


def test_capture_rejects_invalid_duplicate_and_mismatched_pr_identities():
    """Malformed identity evidence cannot be treated as a complete queue."""
    module = _module()

    def run_json(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "open":
            return [
                {"number": True},
                {"number": 3},
                {"number": 3},
                {"number": 4},
                {"number": 5},
            ], None
        if list(command[1:3]) == ["pr", "view"]:
            number = int(command[3])
            if number == 4:
                return ["not-an-object"], None
            if number == 5:
                return _detail(6), None
            return _detail(number), None
        if list(command[1:3]) == ["pr", "list"]:
            return [], None
        if command[1] == "api":
            return {"sha": "d" * 40}, None
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=run_json)

    assert [item["number"] for item in snapshot["open_prs"]] == [3]
    assert len(snapshot["errors"]) == 4
    assert all(error["returncode"] == 65 for error in snapshot["errors"])


def test_capture_preserves_command_errors_and_malformed_top_level_payloads():
    """Partial success never erases transport or payload failures."""
    module = _module()
    failure = {"command": ["pr", "view"], "stderr": "HTTP 401", "returncode": 1}

    def run_json(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return [], None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "open":
            return {"number": 1}, None
        if list(command[1:3]) == ["pr", "list"]:
            return {"history": []}, None
        if command[1] == "api":
            return None, failure
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=run_json)

    assert snapshot["default_branch"] == ""
    assert snapshot["base_sha"] == ""
    assert snapshot["open_prs"] == []
    assert snapshot["pr_history"] == []
    assert len(snapshot["errors"]) == 3
    assert failure not in snapshot["errors"]  # no branch means no base lookup


def test_capture_validates_repo_name_and_default_branch_sha():
    """Repository and SHA identities stay canonical and fail closed on drift."""
    module = _module()
    with pytest.raises(ValueError, match="owner/name"):
        module.capture_pr_queue_snapshot("not a repo", run_json=lambda _: (None, None))

    def run_json(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if list(command[1:3]) == ["pr", "list"]:
            return [], None
        if command[1] == "api":
            return {"sha": "not-a-sha"}, None
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=run_json)
    assert snapshot["base_sha"] == ""
    assert snapshot["errors"][0]["returncode"] == 65


def test_run_gh_json_retries_only_approved_gateway_statuses(monkeypatch):
    """502/503/504 retry, while auth errors and command timeouts do not."""
    module = _module()
    sleeps: list[float] = []
    responses = iter(
        [
            subprocess.CompletedProcess([], 1, "", "HTTP 503: unavailable"),
            subprocess.CompletedProcess([], 0, '{"ok": true}', ""),
        ]
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(module.time, "sleep", lambda value: sleeps.append(value))
    payload, error = module._run_gh_json(["gh", "api", "x"], retry_sleep_seconds=0.01)
    assert payload == {"ok": True}
    assert error is None
    assert sleeps == [0.01]

    calls = {"count": 0}

    def auth_failure(*args, **kwargs):
        calls["count"] += 1
        return subprocess.CompletedProcess([], 1, "", "HTTP 401: bad credentials")

    monkeypatch.setattr(module.subprocess, "run", auth_failure)
    payload, error = module._run_gh_json(["gh", "api", "x"])
    assert payload is None and error is not None
    assert calls["count"] == 1

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd="gh", timeout=30)
        ),
    )
    payload, error = module._run_gh_json(["gh", "api", "x"])
    assert payload is None and error["returncode"] == 124


def test_run_gh_json_fails_closed_on_start_and_json_errors(monkeypatch):
    """Missing executables and malformed successful output produce audit errors."""
    module = _module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing gh")),
    )
    payload, error = module._run_gh_json(["gh", "pr", "list"])
    assert payload is None and error["returncode"] == 127

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "not-json", ""),
    )
    payload, error = module._run_gh_json(["gh", "pr", "list"])
    assert payload is None and error["returncode"] == 65


def test_snapshot_writer_is_atomic_and_rejects_directory_target(tmp_path):
    """Published evidence is complete JSON and never replaces a directory."""
    module = _module()
    out = tmp_path / "nested" / "snapshot.json"
    module._write_snapshot(out, {"open_prs": [], "errors": []})
    assert json.loads(out.read_text(encoding="utf-8")) == {"errors": [], "open_prs": []}
    with pytest.raises(ValueError, match="file path"):
        module._write_snapshot(tmp_path, {})


def test_main_returns_success_incomplete_and_usage_failure(tmp_path, monkeypatch, capsys):
    """The CLI distinguishes complete evidence, incomplete evidence, and bad input."""
    module = _module()
    monkeypatch.setattr(
        module,
        "capture_pr_queue_snapshot",
        lambda repo: {"repo": repo, "open_prs": [{"number": 1}], "errors": []},
    )
    assert module.main(["--repo", "owner/repo", "--out", str(tmp_path / "ok.json")]) == 0
    assert '"open_pr_count": 1' in capsys.readouterr().out

    monkeypatch.setattr(
        module,
        "capture_pr_queue_snapshot",
        lambda repo: {"repo": repo, "open_prs": [], "errors": [{"stderr": "failed"}]},
    )
    assert module.main(["--repo", "owner/repo", "--out", str(tmp_path / "bad.json")]) == 1

    monkeypatch.setattr(
        module,
        "capture_pr_queue_snapshot",
        lambda repo: (_ for _ in ()).throw(ValueError("invalid")),
    )
    assert module.main(["--repo", "owner/repo", "--out", str(tmp_path / "error.json")]) == 2
    assert "capture_pr_queue_snapshot: invalid" in capsys.readouterr().err


def test_run_gh_json_exhausts_transient_retries_without_sleep(monkeypatch):
    """A final 504 is returned after the exact bounded attempt count."""
    module = _module()
    calls = {"count": 0}

    def always_transient(*args, **kwargs):
        calls["count"] += 1
        return subprocess.CompletedProcess([], 1, "", "HTTP 504: gateway timeout")

    monkeypatch.setattr(module.subprocess, "run", always_transient)
    payload, error = module._run_gh_json(
        ["gh", "api", "x"],
        max_attempts=2,
        retry_sleep_seconds=0,
    )
    assert payload is None
    assert error["returncode"] == 1
    assert calls["count"] == 2


def test_capture_records_detail_and_base_command_failures():
    """Detail and exact-base command failures remain visible in snapshot errors."""
    module = _module()
    detail_failure = {"command": ["pr", "view"], "stderr": "HTTP 401", "returncode": 1}
    base_failure = {"command": ["api", "repos/x"], "stderr": "HTTP 403", "returncode": 1}

    def run_json(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "open":
            return [{"number": 1}], None
        if list(command[1:3]) == ["pr", "view"]:
            return None, detail_failure
        if list(command[1:3]) == ["pr", "list"]:
            return [], None
        if command[1] == "api":
            return None, base_failure
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=run_json)
    assert snapshot["open_prs"] == []
    assert snapshot["errors"] == [detail_failure, base_failure]


def test_capture_handles_missing_branch_error_and_nonobject_base_payload():
    """Missing branch metadata and malformed base payload take distinct safe paths."""
    module = _module()
    repo_failure = {"command": ["repo", "view"], "stderr": "HTTP 403", "returncode": 1}

    def missing_repo(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return None, repo_failure
        if list(command[1:3]) == ["pr", "list"]:
            return [], None
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=missing_repo)
    assert snapshot["default_branch"] == ""
    assert snapshot["errors"] == [repo_failure]

    def malformed_base(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": "main"}, None
        if list(command[1:3]) == ["pr", "list"]:
            return [], None
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=malformed_base)
    assert snapshot["default_branch"] == ""
    assert snapshot["errors"] == []

    def nonobject_base(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if list(command[1:3]) == ["pr", "list"]:
            return [], None
        if command[1] == "api":
            return [], None
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=nonobject_base)
    assert snapshot["base_sha"] == ""
    assert "not an object" in snapshot["errors"][0]["stderr"]


def test_entrypoint_guard_executes_for_invalid_repository(tmp_path):
    """Direct script execution reaches the guarded CLI entry point."""
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            "invalid repo",
            "--out",
            str(tmp_path / "snapshot.json"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 2
    assert "canonical owner/name" in completed.stderr


def test_entrypoint_guard_is_covered_in_process(tmp_path, monkeypatch):
    """The ``__main__`` guard maps CLI failure to ``SystemExit``."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--repo",
            "invalid repo",
            "--out",
            str(tmp_path / "snapshot.json"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc.value.code == 2


def test_capture_supports_current_37_pr_queue_without_one_nested_list_query():
    """The diagnosed 37-PR queue is enriched through 37 bounded detail requests."""
    module = _module()
    detail_numbers: list[int] = []

    def run_json(command: Sequence[str]):
        if list(command[1:3]) == ["repo", "view"]:
            return {"defaultBranchRef": {"name": "main"}}, None
        if list(command[1:3]) == ["pr", "list"] and _state(command) == "open":
            fields = str(command[command.index("--json") + 1]).split(",")
            assert fields == ["number"]
            return [{"number": number} for number in range(1, 38)], None
        if list(command[1:3]) == ["pr", "view"]:
            number = int(command[3])
            detail_numbers.append(number)
            return _detail(number), None
        if list(command[1:3]) == ["pr", "list"]:
            return [], None
        if command[1] == "api":
            return {"sha": "e" * 40}, None
        raise AssertionError(command)

    snapshot = module.capture_pr_queue_snapshot("owner/repo", run_json=run_json)
    assert snapshot["errors"] == []
    assert len(snapshot["open_prs"]) == 37
    assert detail_numbers == list(range(1, 38))
