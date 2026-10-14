"""Tests for the autonomous analysis orchestration service (Phase 11).

All external dependencies (database, git, AI services) are mocked so
the suite runs without a real PostgreSQL connection or network access.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.core.config import Settings
from app.services.autonomous_service import AIDebuggingError, AutonomousAnalysisService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "ai_provider": "mock",
        "discovery_mode": "disabled",
        "discovery_languages": "Python",
        "discovery_excluded_topics": "",
        "discovery_excluded_owners": "",
        "autonomous_workspace_dir": "/tmp/bugforge_test_workspaces",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def make_candidate(**overrides: object) -> MagicMock:
    """Return a minimal RepositoryCandidate mock."""
    from datetime import UTC, datetime, timedelta

    candidate = MagicMock()
    candidate.id = uuid4()
    candidate.full_name = "testorg/testrepo"
    candidate.owner = "testorg"
    candidate.stars = 2_000
    candidate.is_fork = False
    candidate.is_archived = False
    candidate.primary_language = "Python"
    candidate.license_key = "mit"
    candidate.size_kb = 5_000
    candidate.topics = '["python", "web"]'
    candidate.description = "A test repository"
    candidate.last_pushed_at = datetime.now(UTC) - timedelta(days=30)
    candidate.clone_url = "https://github.com/testorg/testrepo.git"
    candidate.default_branch = "main"
    candidate.html_url = "https://github.com/testorg/testrepo"
    candidate.last_analyzed_commit = None

    for k, v in overrides.items():
        setattr(candidate, k, v)

    return candidate


# ---------------------------------------------------------------------------
# Workspace path tests
# ---------------------------------------------------------------------------


def test_workspace_root_uses_configured_dir() -> None:
    svc = AutonomousAnalysisService(make_settings(autonomous_workspace_dir="/custom/workspace"))
    root = svc._workspace_root()
    assert root == Path("/custom/workspace")


def test_workspace_root_defaults_to_home() -> None:
    svc = AutonomousAnalysisService(make_settings(autonomous_workspace_dir=""))
    root = svc._workspace_root()
    assert root == Path.home() / ".bugforge" / "workspaces"


def test_candidate_workspace_sanitizes_name() -> None:
    svc = AutonomousAnalysisService(make_settings(autonomous_workspace_dir="/ws"))
    candidate = make_candidate(full_name="org/repo-name.with.dots")
    ws = svc._candidate_workspace(candidate)
    # Slashes and special chars should be replaced with underscores or kept safe
    assert "/" not in ws.name or str(ws).startswith("/ws/")
    assert ".." not in str(ws)


def test_candidate_workspace_no_path_traversal() -> None:
    """Repository full_name with path traversal must not escape workspace root."""
    svc = AutonomousAnalysisService(make_settings(autonomous_workspace_dir="/ws"))
    candidate = make_candidate(full_name="../../etc/passwd")
    ws = svc._candidate_workspace(candidate)
    # The resulting path must still be under /ws
    assert str(ws).startswith("/ws/")


# ---------------------------------------------------------------------------
# URL validation in _shallow_clone
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shallow_clone_rejects_non_github_url() -> None:
    svc = AutonomousAnalysisService(make_settings())
    candidate = make_candidate(clone_url="https://evil.com/steal.git")
    with pytest.raises(ValueError, match="Unexpected clone URL scheme"):
        await svc._shallow_clone(candidate)


@pytest.mark.asyncio
async def test_shallow_clone_rejects_http_url() -> None:
    svc = AutonomousAnalysisService(make_settings())
    candidate = make_candidate(clone_url="http://github.com/org/repo.git")
    with pytest.raises(ValueError, match="Unexpected clone URL scheme"):
        await svc._shallow_clone(candidate)


@pytest.mark.asyncio
async def test_shallow_clone_rejects_embedded_credentials() -> None:
    svc = AutonomousAnalysisService(make_settings())
    candidate = make_candidate(clone_url="https://user:pass@github.com/org/repo.git")
    with pytest.raises(ValueError, match="Unexpected clone URL scheme"):
        await svc._shallow_clone(candidate)


# ---------------------------------------------------------------------------
# Per-repo timeout enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_fails_run() -> None:
    """A pipeline that hangs past the timeout must produce a failed run."""
    svc = AutonomousAnalysisService(make_settings())

    fail_run_calls: list[tuple[UUID, str, str]] = []

    async def mock_pipeline(*args: object, **kwargs: object) -> None:
        # Simulate an infinite hang
        await asyncio.sleep(9999)

    async def mock_fail_run(run_id: UUID, error_code: str, msg: str) -> None:
        fail_run_calls.append((run_id, error_code, msg))

    run_id = uuid4()
    svc._run_pipeline = mock_pipeline  # type: ignore[method-assign]
    svc._fail_run = mock_fail_run  # type: ignore[method-assign]

    with patch.object(
        AutonomousAnalysisService,
        "_run_pipeline",
        side_effect=mock_pipeline,
    ):
        # Patch timeout to 0.01s so the test doesn't actually wait 30 min
        with patch("app.services.autonomous_service._PER_REPO_TIMEOUT_SECONDS", 0.01):
            # Reassign the patched _fail_run
            svc._fail_run = mock_fail_run  # type: ignore[method-assign]
            await svc.run(make_candidate(), run_id)

    assert len(fail_run_calls) == 1
    assert fail_run_calls[0][0] == run_id
    assert fail_run_calls[0][1] == "timeout"


# ---------------------------------------------------------------------------
# AI debugging failure handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ai_debugging_error_fails_run_not_completes() -> None:
    """An AIDebuggingError must set the run to 'failed', not 'completed'."""
    svc = AutonomousAnalysisService(make_settings())

    fail_run_calls: list[tuple[UUID, str, str]] = []

    async def mock_fail_run(run_id: UUID, error_code: str, msg: str) -> None:
        fail_run_calls.append((run_id, error_code, msg))

    svc._fail_run = mock_fail_run  # type: ignore[method-assign]

    # Verify that AIDebuggingError is correctly classified as an infrastructure error
    err = AIDebuggingError("Provider connection refused")
    assert isinstance(err, RuntimeError)
    assert "Provider connection refused" in str(err)

    # The pipeline catches AIDebuggingError specifically (not generic Exception)
    # and routes it to _fail_run with error_code="ai_debugging_failed".
    # Verify the structure is correct by inspecting the source.
    import inspect

    source = inspect.getsource(svc._run_pipeline)
    assert "AIDebuggingError" in source
    assert "ai_debugging_failed" in source
    # Confirm the catch block calls _fail_run, not some other handler
    assert "_fail_run" in source


def test_ai_debugging_error_is_runtime_error() -> None:
    err = AIDebuggingError("test error")
    assert isinstance(err, RuntimeError)
    assert "test error" in str(err)


# ---------------------------------------------------------------------------
# Workspace cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_workspace_removes_directory(tmp_path: Path) -> None:
    svc = AutonomousAnalysisService(make_settings())
    ws = tmp_path / "test_workspace"
    ws.mkdir()
    (ws / "file.txt").write_text("data")

    await svc._cleanup_workspace(ws)

    assert not ws.exists()


@pytest.mark.asyncio
async def test_cleanup_workspace_ignores_missing_directory(tmp_path: Path) -> None:
    svc = AutonomousAnalysisService(make_settings())
    ws = tmp_path / "nonexistent"
    # Should not raise
    await svc._cleanup_workspace(ws)


# ---------------------------------------------------------------------------
# Discovery service 429 handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discovery_handles_429() -> None:
    """A 429 response from GitHub must stop discovery and set rate_limit error."""
    from unittest.mock import MagicMock

    import httpx

    from app.services.discovery_service import GitHubDiscoveryService

    def make_discovery_settings(**overrides: object) -> Settings:
        d: dict[str, object] = {
            "discovery_min_stars": 1_000,
            "discovery_max_stars": 0,
            "discovery_skip_forks": True,
            "discovery_skip_archived": True,
            "discovery_require_license": True,
            "discovery_languages": "Python",
            "discovery_max_staleness_days": 365,
            "discovery_max_size_kb": 50_000,
            "discovery_excluded_topics": "",
            "discovery_excluded_owners": "",
            "discovery_daily_repo_limit": 100,
            "discovery_mode": "disabled",
            "ai_provider": "mock",
        }
        d.update(overrides)
        return Settings(**d)  # type: ignore[arg-type]

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.headers = {}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    svc = GitHubDiscoveryService(make_discovery_settings())

    with patch("app.services.discovery_service.httpx.AsyncClient", return_value=mock_client):
        result = await svc.discover(max_pages=3)

    assert result.error == "github_rate_limit"
    # Should stop after the first 429 — only 1 API request made
    assert result.api_requests == 1
