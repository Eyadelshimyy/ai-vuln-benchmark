"""Transport-level retry contract for the read-only workflow registry audit."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import scripts.audit_workflow_registry as audit


def test_parse_args_accepts_repository_and_output_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse the two required CLI inputs used by the read-only audit."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_workflow_registry", "--repo", "owner/repo", "--out", "report.json"],
    )

    args = audit._parse_args()

    assert args.repo == "owner/repo"
    assert str(args.out) == "report.json"


@pytest.mark.parametrize("status", [403, 404, 429, 500, 502, 503, 504])
def test_run_gh_api_retries_transient_http_status_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    """Retry bounded transient HTTP failures before returning parsed JSON."""
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr=f"HTTP {status}: transient synthetic failure",
            )
        return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    assert audit._run_gh_api(
        "repos/ContextualWisdomLab/fast-mlsirm/actions/workflows",
        max_attempts=2,
        retry_sleep_seconds=0,
    ) == {"ok": True}
    assert calls == 2


def test_run_gh_api_exhausts_persistent_retryable_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop after the configured attempt ceiling for persistent transient HTTP errors."""
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="HTTP 403: persistent synthetic permission/rate-limit failure",
        )

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    with pytest.raises(audit.GitHubApiError, match="HTTP 403"):
        audit._run_gh_api(
            "repos/ContextualWisdomLab/fast-mlsirm/actions/workflows",
            max_attempts=3,
            retry_sleep_seconds=0,
        )

    assert calls == 3


def test_run_gh_api_does_not_retry_nontransient_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not retry a non-transient authentication failure."""
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="HTTP 401: bad credentials",
        )

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    with pytest.raises(audit.GitHubApiError, match="HTTP 401"):
        audit._run_gh_api(
            "repos/ContextualWisdomLab/fast-mlsirm/actions/workflows",
            max_attempts=3,
            retry_sleep_seconds=0,
        )

    assert calls == 1


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("SECRET_LOCAL_GH_PATH"),
        PermissionError("SECRET_LOCAL_EXECUTION_DETAIL"),
        OSError("SECRET_LOCAL_OS_DETAIL"),
    ],
)
def test_run_gh_api_normalizes_spawn_failures_without_retry_or_detail_leak(
    monkeypatch: pytest.MonkeyPatch,
    failure: OSError,
) -> None:
    """Local process-spawn failures become stable source-free GitHub API errors."""
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    with pytest.raises(audit.GitHubApiError) as captured:
        audit._run_gh_api(
            "repos/ContextualWisdomLab/fast-mlsirm/actions/workflows",
            max_attempts=3,
            retry_sleep_seconds=0,
        )

    error = captured.value
    assert error.returncode == 127
    assert error.stderr == "GitHub CLI transport unavailable"
    assert "SECRET_LOCAL" not in str(error)
    assert calls == 1


def test_main_writes_failed_json_and_exit_code_for_github_api_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Turn a live GitHub transport failure into the documented JSON evidence."""
    output_path = tmp_path / "workflow-registry.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_workflow_registry",
            "--repo",
            "owner/repo",
            "--out",
            str(output_path),
        ],
    )
    monkeypatch.setattr(
        audit,
        "audit_workflow_registry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            audit.GitHubApiError(
                endpoint="repos/owner/repo",
                returncode=503,
                stderr="temporary upstream failure",
            )
        ),
    )

    assert audit.main() == 2

    evidence = json.loads(output_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "failed"
    assert evidence["repository"] == "owner/repo"
    assert evidence["errors"] == [
        "GitHub API request failed for repos/owner/repo: "
        "exit 503: temporary upstream failure"
    ]
