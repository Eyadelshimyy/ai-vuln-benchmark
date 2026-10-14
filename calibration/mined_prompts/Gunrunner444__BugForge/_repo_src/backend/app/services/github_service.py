"""GitHub Integration service — Phase 9.

Security invariants enforced here:
 - Unverified/rejected/inconclusive candidates cannot be delivered.
 - The delivered patch MUST exactly match the verified patch (hash check).
 - GitHub tokens are NEVER logged, stored in DB, or sent to AI prompts.
 - Git commands always use argument arrays (never shell interpolation).
 - Branch names are sanitized to prevent injection.
 - Repository URLs are validated to be on github.com.
 - Default branch is NEVER modified directly.
 - PRs are never automatically merged.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

# Maximum allowed branch name length
_MAX_BRANCH_LEN = 100
# Pattern for valid branch name characters (after prefix)
_BRANCH_SAFE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-_./]*$")
# Control characters to strip from commit messages
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

_GITHUB_API = "https://api.github.com"
_GITHUB_HOST = "github.com"


# ── Security helpers ──────────────────────────────────────────────────────────


def _patch_hash(patch_diff: str) -> str:
    """Stable SHA-256 of the patch content."""
    return hashlib.sha256(patch_diff.encode()).hexdigest()


def _sanitize_token(text: str, token: str) -> str:
    """Remove the GitHub token from any string before logging/storing."""
    if token:
        text = text.replace(token, "***")
    return text


def _delivery_branch(candidate_id: UUID) -> str:
    """Deterministic collision-resistant branch name for a candidate."""
    short_id = str(candidate_id).replace("-", "")[:12]
    branch = f"bugforge/fix/{short_id}"
    return branch


def _validate_branch_name(branch: str) -> None:
    """Raise ValueError if the branch name is unsafe."""
    if not branch or len(branch) > _MAX_BRANCH_LEN:
        raise ValueError(f"Branch name is empty or too long: {branch!r}")
    # Allow bugforge/ prefix with alphanumeric + safe chars
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9\-_/]*$", branch):
        raise ValueError(f"Branch name contains unsafe characters: {branch!r}")
    # No consecutive dots, no trailing dot, no @{ sequences
    if ".." in branch or branch.endswith(".") or "@{" in branch:
        raise ValueError(f"Branch name has invalid git sequences: {branch!r}")


def _validate_github_url(url: str) -> None:
    """Raise ValueError if the URL is not a valid github.com HTTPS URL.

    Requires:
    - HTTPS scheme
    - Hostname exactly 'github.com' (no subdomain spoofing, no path-based tricks)
    - No embedded credentials (userinfo component)
    - No non-standard port
    - No query parameters or fragments
    - At least a two-component path (owner/repo)
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"Could not parse URL: {url!r}") from exc

    if parsed.scheme != "https":
        raise ValueError(f"Repository URL must use HTTPS, got scheme {parsed.scheme!r}")

    # Reject any embedded credentials — they could be used for SSRF-style attacks
    if parsed.username or parsed.password:
        raise ValueError(f"URL must not contain embedded credentials: {url!r}")

    # Hostname must be exactly 'github.com' — substring checks are insufficient
    hostname = (parsed.hostname or "").lower()
    if hostname != _GITHUB_HOST:
        raise ValueError(
            f"URL hostname must be exactly '{_GITHUB_HOST}', got {hostname!r}: {url!r}"
        )

    # Reject non-standard ports (443 is implicit for HTTPS)
    if parsed.port is not None and parsed.port != 443:
        raise ValueError(f"URL must not specify a non-standard port: {url!r}")

    # Reject query parameters and fragments
    if parsed.query:
        raise ValueError(f"URL must not contain query parameters: {url!r}")
    if parsed.fragment:
        raise ValueError(f"URL must not contain URL fragments: {url!r}")

    # Path must contain at least owner/repo
    path_parts = [p for p in parsed.path.split("/") if p]
    if len(path_parts) < 2:  # noqa: PLR2004
        raise ValueError(f"URL must contain an owner/repo path: {url!r}")


def _sanitize_commit_message(msg: str) -> str:
    """Remove control characters; truncate to safe length."""
    msg = _CONTROL_CHARS.sub(" ", msg)
    return msg[:500].strip()


@contextmanager
def _git_auth_env(token: str) -> Iterator[dict[str, str]]:
    """Context manager that provisions a temporary GIT_ASKPASS helper.

    Supplies GitHub credentials via GIT_ASKPASS rather than embedding the
    token in the remote URL (which would expose it in git error messages,
    process arguments, and shell history).

    Yields an env dict suitable for passing to _git().  The temp directory
    (including the token file) is removed when the context manager exits.
    """
    tmpdir = tempfile.mkdtemp(prefix="bugforge_cred_")
    try:
        os.chmod(tmpdir, 0o700)
        tok_path = os.path.join(tmpdir, ".token")
        with open(tok_path, "w") as _f:
            _f.write(token)
        os.chmod(tok_path, 0o600)

        # The ASKPASS script reads the token from a separate file so the
        # token never appears in the script text itself.
        tok_escaped = tok_path.replace("'", "'\\''")
        script_path = os.path.join(tmpdir, "askpass")
        script = (
            "#!/bin/sh\n"
            'case "$1" in\n'
            "  *Username*) printf '%s' 'x-access-token' ;;\n"
            f"  *) printf '%s' \"$(cat '{tok_escaped}')\" ;;\n"
            "esac\n"
        )
        with open(script_path, "w") as _f:
            _f.write(script)
        os.chmod(script_path, stat.S_IRWXU)

        env = dict(os.environ)
        env["GIT_ASKPASS"] = script_path
        env["GIT_TERMINAL_PROMPT"] = "0"
        # Prevent system-wide credential helpers from interfering
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        # Strip BugForge secrets so the subprocess never sees them
        for _k in (
            "GITHUB_TOKEN",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "SECRET_KEY",
            "DATABASE_URL",
            "POSTGRES_PASSWORD",
        ):
            env.pop(_k, None)

        yield env
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _plain_clone_url(owner: str, repo: str) -> str:
    """Return the plain (no-credentials) HTTPS clone URL for a GitHub repo."""
    return f"https://github.com/{owner}/{repo}.git"


def _git(
    args: list[str], cwd: str | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a git command with argument array (never shell=True)."""
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _require_git() -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is not installed or not on PATH")


# ── GitHub API client ─────────────────────────────────────────────────────────


class _GitHubClient:
    """Minimal httpx-based GitHub REST API client."""

    def __init__(self, token: str) -> None:
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{_GITHUB_API}/repos/{owner}/{repo}",
                headers=self._headers,
            )
        if resp.status_code == 404:
            raise ValueError(f"Repository {owner}/{repo} not found or not accessible")
        if resp.status_code == 401:
            raise ValueError("GitHub token is invalid or expired")
        resp.raise_for_status()
        return dict(resp.json())

    async def list_prs(
        self, owner: str, repo: str, head: str, state: str = "open"
    ) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{_GITHUB_API}/repos/{owner}/{repo}/pulls",
                headers=self._headers,
                params={"head": f"{owner}:{head}", "state": state},
            )
        resp.raise_for_status()
        return list(resp.json())

    async def create_pr(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{_GITHUB_API}/repos/{owner}/{repo}/pulls",
                headers=self._headers,
                json={"title": title, "body": body, "head": head, "base": base},
            )
        if resp.status_code == 422:
            detail = resp.json().get("message", "Unprocessable entity")
            raise ValueError(f"GitHub PR creation failed: {detail}")
        resp.raise_for_status()
        return dict(resp.json())


# ── Service ───────────────────────────────────────────────────────────────────


class GitHubService:
    """Handles GitHub repository connection and verified patch delivery."""

    async def connect_repository(
        self,
        project_id: UUID,
        owner: str,
        repo: str,
    ) -> UUID:
        """Verify GitHub access and create/update the GitHubRepository record."""
        from app.core.config import settings
        from app.database import async_session_factory
        from app.repositories.github_repo import GitHubRepositoryRepo
        from app.repositories.project_repo import ProjectRepository

        if not settings.github_token:
            raise ValueError(
                "GitHub token is not configured (set GITHUB_TOKEN environment variable)"
            )

        # Validate owner/repo names to prevent injection
        _validate_github_owner_repo(owner, repo)

        client = _GitHubClient(settings.github_token)
        try:
            repo_data = await client.get_repo(owner, repo)
        except httpx.HTTPStatusError as exc:
            raise ValueError(f"GitHub API error: {exc.response.status_code}") from exc

        html_url = repo_data.get("html_url", f"https://github.com/{owner}/{repo}")
        default_branch = repo_data.get("default_branch", "main")
        github_id = repo_data.get("id")

        async with async_session_factory() as db:
            project_repo = ProjectRepository(db)
            project = await project_repo.get_by_id(project_id)
            if project is None:
                raise ValueError(f"Project {project_id} not found")

            github_repo_repo = GitHubRepositoryRepo(db)
            existing = await github_repo_repo.get_by_project(project_id)
            if existing is not None:
                await github_repo_repo.update(
                    existing.id,
                    owner=owner,
                    repo=repo,
                    github_id=github_id,
                    default_branch=default_branch,
                    html_url=html_url,
                    connected=True,
                )
                await db.commit()
                return existing.id

            gr = await github_repo_repo.create(
                project_id=project_id,
                owner=owner,
                repo=repo,
                default_branch=default_branch,
                html_url=html_url,
                github_id=github_id,
            )
            await github_repo_repo.update(gr.id, connected=True)
            await db.commit()
            return gr.id

    async def disconnect_repository(self, project_id: UUID) -> None:
        """Remove the GitHub connection for a project."""
        from app.database import async_session_factory
        from app.repositories.github_repo import GitHubRepositoryRepo

        async with async_session_factory() as db:
            github_repo_repo = GitHubRepositoryRepo(db)
            existing = await github_repo_repo.get_by_project(project_id)
            if existing is not None:
                await github_repo_repo.delete(existing.id)
                await db.commit()

    async def start_delivery(
        self,
        project_id: UUID,
        candidate_id: UUID,
    ) -> UUID:
        """Create a delivery record and enqueue the delivery pipeline.

        SECURITY: Only VERIFIED candidates from the specified project are accepted.
        Ownership chain validated before token or infra checks.
        """
        from app.database import async_session_factory
        from app.repositories.github_repo import GitHubDeliveryRepo, GitHubRepositoryRepo
        from app.repositories.repair_repo import RepairRepository
        from app.repositories.verification_repo import VerificationRepository

        async with async_session_factory() as db:
            # ── Ownership chain validation (security-first) ───────────
            repair_repo = RepairRepository(db)
            candidate = await repair_repo.get_candidate(candidate_id)
            if candidate is None:
                raise ValueError(f"Candidate {candidate_id} not found")

            repair_session = await repair_repo.get_session_by_id(candidate.session_id)
            if repair_session is None or repair_session.project_id != project_id:
                raise ValueError(
                    f"Candidate {candidate_id} does not belong to project {project_id}"
                )

            # ── Must be verified — fail closed ────────────────────────
            ver_repo = VerificationRepository(db)
            verification = await ver_repo.get_by_candidate(candidate_id)
            if verification is None:
                raise ValueError(
                    f"Candidate {candidate_id} has no verification record — run verification first"
                )
            if verification.project_id != project_id:
                raise ValueError("Verification does not belong to the specified project")
            if verification.verification_decision != "verified":
                raise ValueError(
                    f"Candidate {candidate_id} cannot be delivered: "
                    f"verification_decision={verification.verification_decision!r}. "
                    "Only 'verified' candidates may be delivered to GitHub."
                )

            # ── GitHub repository must be connected ───────────────────
            github_repo_repo = GitHubRepositoryRepo(db)
            github_repo = await github_repo_repo.get_by_project(project_id)
            if github_repo is None or not github_repo.connected:
                raise ValueError(
                    f"Project {project_id} has no connected GitHub repository — use /github/connect first"
                )

            # ── Infra check: token must be present ────────────────────
            from app.core.config import settings

            if not settings.github_token:
                raise ValueError(
                    "GitHub token is not configured (set GITHUB_TOKEN environment variable)"
                )

            # ── Patch hash for later comparison ───────────────────────
            patch_diff = candidate.patch_diff or ""
            current_hash = _patch_hash(patch_diff)

            # ── Idempotency — prevent duplicate active/pending deliveries ─
            delivery_repo = GitHubDeliveryRepo(db)
            existing_delivery = await delivery_repo.get_by_candidate(candidate_id)
            if existing_delivery is not None and existing_delivery.status not in (
                "failed",
                "aborted",
            ):
                return existing_delivery.id

            delivery_branch = _delivery_branch(candidate_id)
            _validate_branch_name(delivery_branch)

            delivery = await delivery_repo.create(
                project_id=project_id,
                github_repository_id=github_repo.id,
                candidate_id=candidate_id,
                verification_id=verification.id,
                owner=github_repo.owner,
                repo=github_repo.repo,
                base_branch=github_repo.default_branch,
                delivery_branch=delivery_branch,
                verified_patch_hash=current_hash,
            )
            delivery_id = delivery.id
            await db.commit()

        return delivery_id

    async def run_delivery_pipeline(self, delivery_id: UUID) -> None:
        """Execute the full delivery pipeline: clone → patch → verify → commit → push → PR."""
        from app.core.config import settings
        from app.database import async_session_factory
        from app.repositories.github_repo import GitHubDeliveryRepo
        from app.repositories.repair_repo import RepairRepository
        from app.repositories.verification_repo import VerificationRepository

        token = settings.github_token
        if not token:
            raise RuntimeError("GitHub token not configured")

        _require_git()

        async with async_session_factory() as db:
            delivery_repo = GitHubDeliveryRepo(db)
            delivery = await delivery_repo.get_by_id(delivery_id)
            if delivery is None:
                raise ValueError(f"Delivery {delivery_id} not found")
            await delivery_repo.update(
                delivery_id, status="preparing", started_at=datetime.now(UTC)
            )
            await db.commit()

        tmpdir: str | None = None
        try:
            async with async_session_factory() as db:
                delivery_repo = GitHubDeliveryRepo(db)
                delivery = await delivery_repo.get_by_id(delivery_id)
                if delivery is None:
                    return

                # Re-validate ownership chain
                repair_repo = RepairRepository(db)
                candidate = await repair_repo.get_candidate(delivery.candidate_id)
                if candidate is None or candidate.session_id is None:
                    raise ValueError("Candidate not found during delivery")

                ver_repo = VerificationRepository(db)
                verification = await ver_repo.get_by_candidate(delivery.candidate_id)
                if verification is None or verification.verification_decision != "verified":
                    raise ValueError("Verification is no longer 'verified' — delivery aborted")

                patch_diff = candidate.patch_diff or ""

            # ── Hash verification — delivered == verified ─────────────
            current_hash = _patch_hash(patch_diff)
            async with async_session_factory() as db:
                delivery_repo = GitHubDeliveryRepo(db)
                delivery = await delivery_repo.get_by_id(delivery_id)
                if delivery is None:
                    return
                if delivery.verified_patch_hash != current_hash:
                    await delivery_repo.update(
                        delivery_id,
                        status="aborted",
                        error_message=(
                            "Patch content has changed since verification — "
                            "hash mismatch, delivery aborted"
                        ),
                        completed_at=datetime.now(UTC),
                    )
                    await db.commit()
                    return
                owner = delivery.owner
                repo = delivery.repo
                base_branch = delivery.base_branch
                delivery_branch = delivery.delivery_branch

            # ── Clone repository ──────────────────────────────────────
            tmpdir = tempfile.mkdtemp(prefix="bugforge_gh_")
            clone_dir = str(Path(tmpdir) / "repo")
            clone_url = _plain_clone_url(owner, repo)

            with _git_auth_env(token) as auth_env:
                result = _git(
                    ["clone", "--depth", "1", "--branch", base_branch, clone_url, clone_dir],
                    env=auth_env,
                )
                if result.returncode != 0:
                    sanitized_err = _sanitize_token(result.stderr, token)
                    raise RuntimeError(f"git clone failed: {sanitized_err}")

            # ── Create delivery branch ────────────────────────────────
            result = _git(["checkout", "-b", delivery_branch], cwd=clone_dir)
            if result.returncode != 0:
                sanitized_err = _sanitize_token(result.stderr, token)
                raise RuntimeError(f"git checkout failed: {sanitized_err}")

            # branch_created is set AFTER the branch is confirmed to exist
            async with async_session_factory() as db:
                delivery_repo = GitHubDeliveryRepo(db)
                await delivery_repo.update(delivery_id, status="branch_created")
                await db.commit()

            # ── Apply verified patch ──────────────────────────────────
            from app.testing.repair_workspace import RepairWorkspace

            # Use RepairWorkspace's patch application logic on the clone directory
            ws_temp = RepairWorkspace.__new__(RepairWorkspace)
            object.__setattr__(ws_temp, "_workspace_dir", Path(tmpdir))
            object.__setattr__(ws_temp, "_repo_root", Path(clone_dir))
            ok, apply_err = ws_temp.apply_patch(patch_diff)
            if not ok:
                raise RuntimeError(f"Patch application failed: {apply_err}")

            # ── Final pre-push verification ───────────────────────────
            async with async_session_factory() as db:
                delivery_repo = GitHubDeliveryRepo(db)
                await delivery_repo.update(delivery_id, status="final_verification")
                await db.commit()

            final_ok = await _run_final_tests(clone_dir)
            if not final_ok:
                raise RuntimeError("Final pre-push test run failed — patch not delivered")

            # ── Commit ────────────────────────────────────────────────
            async with async_session_factory() as db:
                delivery_repo = GitHubDeliveryRepo(db)
                await delivery_repo.update(delivery_id, status="committing")
                await db.commit()

            result = _git(["add", "--all"], cwd=clone_dir)
            if result.returncode != 0:
                raise RuntimeError(f"git add failed: {result.stderr}")

            commit_msg = _sanitize_commit_message(
                f"fix: BugForge automated repair (candidate {str(delivery.candidate_id)[:8]})\n\n"
                f"Verified by BugForge patch verification pipeline.\n"
                f"Delivery ID: {delivery_id}\n"
                f"Patch hash: {current_hash[:16]}..."
            )
            # Configure git identity for the commit (no real user needed)
            _git(["config", "user.email", "bugforge-bot@localhost"], cwd=clone_dir)
            _git(["config", "user.name", "BugForge Bot"], cwd=clone_dir)

            result = _git(["commit", "-m", commit_msg], cwd=clone_dir)
            if result.returncode != 0:
                raise RuntimeError(f"git commit failed: {result.stderr}")

            # Extract commit SHA
            sha_result = _git(["rev-parse", "HEAD"], cwd=clone_dir)
            commit_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else None

            # ── Push ──────────────────────────────────────────────────
            async with async_session_factory() as db:
                delivery_repo = GitHubDeliveryRepo(db)
                await delivery_repo.update(
                    delivery_id,
                    status="pushing",
                    commit_sha=commit_sha,
                    delivered_patch_hash=current_hash,
                )
                await db.commit()

            with _git_auth_env(token) as push_env:
                result = _git(["push", "origin", delivery_branch], cwd=clone_dir, env=push_env)
                if result.returncode != 0:
                    sanitized_err = _sanitize_token(result.stderr, token)
                    raise RuntimeError(f"git push failed: {sanitized_err}")

            # ── Create PR via GitHub API ───────────────────────────────
            # pr_created is set only AFTER GitHub confirms the PR exists
            client = _GitHubClient(token)

            # Check for existing PR to prevent duplicates
            existing_prs = await client.list_prs(owner, repo, head=delivery_branch)
            if existing_prs:
                pr = existing_prs[0]
                pr_number = pr["number"]
                pr_url = pr["html_url"]
            else:
                pr_title = _sanitize_commit_message(
                    f"BugForge: automated fix (candidate {str(delivery_id)[:8]})"
                )
                pr_body = _build_pr_body(
                    delivery_id=delivery_id,
                    candidate_id=delivery.candidate_id,
                    verification_id=delivery.verification_id,
                    commit_sha=commit_sha,
                    patch_hash=current_hash,
                )
                pr_data = await client.create_pr(
                    owner=owner,
                    repo=repo,
                    title=pr_title,
                    body=pr_body,
                    head=delivery_branch,
                    base=base_branch,
                )
                pr_number = pr_data["number"]
                pr_url = pr_data["html_url"]

            async with async_session_factory() as db:
                delivery_repo = GitHubDeliveryRepo(db)
                await delivery_repo.update(
                    delivery_id,
                    pull_request_number=pr_number,
                    pull_request_url=pr_url,
                    status="pr_created",
                )
                await db.commit()

            async with async_session_factory() as db:
                delivery_repo = GitHubDeliveryRepo(db)
                await delivery_repo.update(
                    delivery_id,
                    status="completed",
                    completed_at=datetime.now(UTC),
                )
                await db.commit()

        except Exception as exc:
            sanitized_msg = _sanitize_token(str(exc), token)
            logger.exception("Delivery %s failed: %s", delivery_id, sanitized_msg)
            async with async_session_factory() as db:
                delivery_repo = GitHubDeliveryRepo(db)
                await delivery_repo.update(
                    delivery_id,
                    status="failed",
                    error_message=sanitized_msg[:2048],
                    completed_at=datetime.now(UTC),
                )
                await db.commit()
        finally:
            if tmpdir is not None:
                try:
                    import shutil as _shutil

                    _shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass


# ── Pure helpers ──────────────────────────────────────────────────────────────


def _validate_github_owner_repo(owner: str, repo: str) -> None:
    """Reject obviously unsafe owner/repo values."""
    pattern = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-_.]*$")
    if not pattern.match(owner):
        raise ValueError(f"Invalid GitHub owner name: {owner!r}")
    if not pattern.match(repo):
        raise ValueError(f"Invalid GitHub repo name: {repo!r}")
    if len(owner) > 255 or len(repo) > 255:
        raise ValueError("GitHub owner/repo names are too long")


async def _run_final_tests(repo_dir: str) -> bool:
    """Run the test suite against the patched clone via BugForge's execution abstraction.

    Uses ExecutorFactory so production always runs inside the Docker sandbox;
    never executes untrusted repository code directly on the BugForge host.
    Raises RuntimeError in production when Docker is unavailable.
    """
    from app.execution import ExecutionConfig, ExecutorFactory

    executor = ExecutorFactory.create()
    config = ExecutionConfig(
        command=[
            "python3",
            "-m",
            "pytest",
            "--tb=no",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        working_directory=repo_dir,
        timeout_seconds=120,
    )
    result = await executor.execute(config)
    return result.exit_code == 0 and not result.timed_out


def _build_pr_body(
    *,
    delivery_id: UUID,
    candidate_id: UUID,
    verification_id: UUID,
    commit_sha: str | None,
    patch_hash: str,
) -> str:
    """Build an evidence-based PR body.  Never includes secrets."""
    lines = [
        "## BugForge Automated Repair",
        "",
        "This pull request was created by **BugForge** after the patch candidate was ",
        "independently verified by the BugForge patch verification pipeline.",
        "",
        "### Verification Evidence",
        f"- Verification ID: `{verification_id}`",
        f"- Candidate ID: `{candidate_id}`",
        f"- Delivery ID: `{delivery_id}`",
        f"- Patch SHA-256 (first 16): `{patch_hash[:16]}...`",
    ]
    if commit_sha:
        lines.append(f"- Commit: `{commit_sha}`")
    lines += [
        "",
        "### What was verified",
        "- Target bug reproduced in baseline ✓",
        "- Patch applied to isolated workspace ✓",
        "- Target bug no longer reproduces post-patch ✓",
        "- No previously-passing tests newly fail ✓",
        "- Security checks passed ✓",
        "- Static analysis comparison completed ✓",
        "",
        "### Important notes",
        "- This patch was **verified** by BugForge — it is not an unreviewed AI suggestion.",
        "- BugForge does **not** automatically merge pull requests.",
        "- Please review the changes before merging.",
        "- BugForge does not guarantee correctness beyond the verified evidence above.",
    ]
    return "\n".join(lines)
