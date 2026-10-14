#!/usr/bin/env python
"""Audit GitHub Actions workflow-registry drift without mutating repository state."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

try:
    from scripts._bounded_json import parse_json_bounded
except ModuleNotFoundError:
    from _bounded_json import parse_json_bounded


_RETRYABLE_HTTP_RE = re.compile(r"\bHTTP (?:403|404|429|5\d\d)\b", re.IGNORECASE)
_REPO_SLUG_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_MAX_ATTEMPTS = 3
_RETRY_SLEEP_SECONDS = 0.5
_GH_TIMEOUT_SECONDS = 20
_WORKFLOW_PREFIX = ".github/workflows/"
_DYNAMIC_PREFIX = "dynamic/"


@dataclass(frozen=True)
class GitHubApiError(RuntimeError):
    """Fail-closed GitHub CLI API error with bounded diagnostic evidence."""

    endpoint: str
    returncode: int
    stderr: str

    def __str__(self) -> str:
        return (
            f"GitHub API request failed for {self.endpoint}: "
            f"exit {self.returncode}: {self.stderr}"
        )


FetchJson = Callable[[str], Any]


def _repo_slug(value: str) -> str:
    """Validate one unambiguous GitHub ``owner/name`` repository slug."""
    if _REPO_SLUG_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("repo must be in owner/name form")
    return value


def _integer_argument(value: Any, *, name: str) -> int:
    """Return an exact integer or normalize invalid input to ``ValueError``."""
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _utc_timestamp(value: datetime | None = None) -> str:
    """Return one RFC-3339 UTC observation timestamp."""
    observed = value or datetime.now(UTC)
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    return observed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _run_gh_api(
    endpoint: str,
    *,
    max_attempts: int = _MAX_ATTEMPTS,
    retry_sleep_seconds: float = _RETRY_SLEEP_SECONDS,
) -> Any:
    """Fetch one GitHub REST payload, retrying bounded transient failures."""
    attempts = max(1, _integer_argument(max_attempts, name="max_attempts"))
    last_error: GitHubApiError | None = None
    for attempt in range(1, attempts + 1):
        try:
            completed = subprocess.run(
                ["gh", "api", endpoint],
                capture_output=True,
                text=True,
                timeout=_GH_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = GitHubApiError(
                endpoint=endpoint,
                returncode=124,
                stderr="GitHub API request timed out",
            )
            if attempt >= attempts:
                raise last_error from exc
            if retry_sleep_seconds > 0:
                time.sleep(retry_sleep_seconds)
            continue
        except OSError as exc:
            raise GitHubApiError(
                endpoint=endpoint,
                returncode=127,
                stderr="GitHub CLI transport unavailable",
            ) from exc
        if completed.returncode == 0:
            try:
                return parse_json_bounded(completed.stdout)
            except ValueError as exc:
                raise GitHubApiError(
                    endpoint=endpoint,
                    returncode=completed.returncode,
                    stderr="GitHub API returned invalid JSON",
                ) from exc

        stderr = completed.stderr.strip()
        last_error = GitHubApiError(
            endpoint=endpoint,
            returncode=completed.returncode,
            stderr=stderr,
        )
        if attempt >= attempts or _RETRYABLE_HTTP_RE.search(stderr) is None:
            break
        if retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds)

    if last_error is None:
        raise GitHubApiError(endpoint=endpoint, returncode=1, stderr="unknown failure")
    raise last_error


def _require_mapping(payload: Any, *, context: str) -> dict[str, Any]:
    """Return a mapping payload or fail closed."""
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} payload is not an object")
    return payload


def _default_branch(fetch_json: FetchJson, repo: str) -> str:
    """Resolve the repository default branch from live repository metadata."""
    payload = _require_mapping(fetch_json(f"repos/{repo}"), context="repository")
    branch = payload.get("default_branch")
    if not isinstance(branch, str) or not branch.strip():
        raise RuntimeError("repository default_branch is missing")
    return branch.strip()


def _branch_sha(fetch_json: FetchJson, repo: str, branch: str) -> str:
    """Resolve an exact branch SHA through the Git ref API."""
    encoded = quote(branch, safe="")
    payload = _require_mapping(
        fetch_json(f"repos/{repo}/git/ref/heads/{encoded}"),
        context="branch ref",
    )
    obj = payload.get("object")
    if not isinstance(obj, dict):
        raise RuntimeError("branch ref object is missing")
    sha = obj.get("sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise RuntimeError("branch ref SHA is invalid")
    return sha.lower()


def _workflow_paths(fetch_json: FetchJson, repo: str, sha: str) -> set[str]:
    """Return exact workflow source paths from a complete commit tree."""
    payload = _require_mapping(
        fetch_json(f"repos/{repo}/git/trees/{sha}?recursive=1"),
        context="git tree",
    )
    if payload.get("truncated") is True:
        raise RuntimeError("default-branch tree is truncated")
    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise RuntimeError("git tree entries are missing")

    paths: set[str] = set()
    for entry in tree:
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue
        path = entry.get("path")
        if (
            isinstance(path, str)
            and path.startswith(_WORKFLOW_PREFIX)
            and path.endswith((".yml", ".yaml"))
        ):
            paths.add(path)
    return paths


def collect_workflow_registry(
    fetch_json: FetchJson,
    repo: str,
    *,
    per_page: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect every Actions workflow-registry page with pagination receipts."""
    page_size = _integer_argument(per_page, name="per_page")
    if page_size <= 0 or page_size > 100:
        raise ValueError("per_page must be between 1 and 100")

    workflows: list[dict[str, Any]] = []
    seen_workflow_ids: set[int] = set()
    receipts: list[dict[str, int]] = []
    expected_total: int | None = None
    page = 1

    while expected_total is None or len(workflows) < expected_total:
        endpoint = f"repos/{repo}/actions/workflows?per_page={page_size}&page={page}"
        payload = _require_mapping(fetch_json(endpoint), context="workflow registry")
        total = payload.get("total_count")
        batch = payload.get("workflows")
        if type(total) is not int or total < 0:
            raise RuntimeError("workflow registry total_count is invalid")
        if not isinstance(batch, list) or any(
            not isinstance(item, dict) for item in batch
        ):
            raise RuntimeError("workflow registry page is malformed")

        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise RuntimeError("workflow registry total_count changed during pagination")

        receipts.append({"page": page, "count": len(batch)})
        if not batch and len(workflows) < expected_total:
            raise RuntimeError("partial workflow registry pagination")

        for workflow in batch:
            workflow_id = workflow.get("id")
            if type(workflow_id) is int:
                if workflow_id in seen_workflow_ids:
                    raise RuntimeError("workflow registry contains duplicate workflow id")
                seen_workflow_ids.add(workflow_id)
        workflows.extend(batch)

        if len(workflows) > expected_total:
            raise RuntimeError("workflow registry returned more records than total_count")
        page += 1

    if expected_total is None:
        raise RuntimeError("workflow registry pagination produced no pages")
    return workflows, {
        "per_page": page_size,
        "total_count": expected_total,
        "received_count": len(workflows),
        "pages": receipts,
        "complete": len(workflows) == expected_total,
    }


def _identity_conflicts(
    workflows: list[dict[str, Any]],
) -> tuple[set[int], set[str]]:
    """Find reused IDs and duplicate active repository paths."""
    id_paths: dict[int, set[str]] = defaultdict(set)
    active_path_ids: dict[str, set[int]] = defaultdict(set)

    for workflow in workflows:
        workflow_id = workflow.get("id")
        path = workflow.get("path")
        state = workflow.get("state")
        if type(workflow_id) is not int or not isinstance(path, str):
            continue
        id_paths[workflow_id].add(path)
        if state == "active" and path.startswith(_WORKFLOW_PREFIX):
            active_path_ids[path].add(workflow_id)

    conflicting_ids = {
        workflow_id for workflow_id, paths in id_paths.items() if len(paths) > 1
    }
    conflicting_paths = {
        path for path, workflow_ids in active_path_ids.items() if len(workflow_ids) > 1
    }
    return conflicting_ids, conflicting_paths


def classify_workflows(
    workflows: list[dict[str, Any]],
    *,
    present_paths: set[str],
    default_branch_sha: str,
    observed_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Classify registry identities using exact paths and integrity checks."""
    timestamp = _utc_timestamp(observed_at)
    conflicting_ids, conflicting_paths = _identity_conflicts(workflows)
    records: list[dict[str, Any]] = []

    for workflow in workflows:
        workflow_id = workflow.get("id")
        path = workflow.get("path")
        state = workflow.get("state")
        name = workflow.get("name")
        integrity_conflict = (
            type(workflow_id) is int and workflow_id in conflicting_ids
        ) or (isinstance(path, str) and path in conflicting_paths)

        if integrity_conflict:
            classification = "unresolved"
        elif not isinstance(path, str) or not path:
            classification = "unresolved"
        elif state == "active":
            if path.startswith(_DYNAMIC_PREFIX):
                classification = "dynamic"
            elif path.startswith(_WORKFLOW_PREFIX):
                classification = "present" if path in present_paths else "orphan"
            else:
                classification = "unresolved"
        elif isinstance(state, str) and state.startswith("disabled_"):
            classification = "disabled"
        else:
            classification = "unresolved"

        records.append(
            {
                "id": workflow_id,
                "name": name,
                "path": path,
                "state": state,
                "classification": classification,
                "integrity_conflict": integrity_conflict,
                "default_branch_sha": default_branch_sha,
                "observed_at": timestamp,
            }
        )

    return records


def _summary(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count registry classifications with explicit active-orphan naming."""
    counts = Counter(
        str(record.get("classification", "unresolved")) for record in records
    )
    return {
        "total": len(records),
        "present": counts["present"],
        "active_orphan": counts["orphan"],
        "disabled": counts["disabled"],
        "dynamic": counts["dynamic"],
        "unresolved": counts["unresolved"],
    }


def _audit_workflow_registry_snapshot(
    fetch_json: FetchJson,
    repo: str,
    branch: str,
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Build one exact branch-bound registry snapshot and detect movement."""
    timestamp = _utc_timestamp(observed_at)
    start_sha = _branch_sha(fetch_json, repo, branch)
    present_paths = _workflow_paths(fetch_json, repo, start_sha)
    workflows, pagination = collect_workflow_registry(fetch_json, repo)
    records = classify_workflows(
        workflows,
        present_paths=present_paths,
        default_branch_sha=start_sha,
        observed_at=observed_at,
    )
    end_sha = _branch_sha(fetch_json, repo, branch)
    end_branch = _default_branch(fetch_json, repo)
    branch_stable = end_branch == branch
    sha_stable = end_sha == start_sha
    stable = branch_stable and sha_stable

    errors: list[str] = []
    if not branch_stable:
        errors.append("default branch changed during workflow registry audit")
    elif not sha_stable:
        errors.append("default branch moved during workflow registry audit")
    if any(record["classification"] == "unresolved" for record in records):
        errors.append("workflow registry contains unresolved identities")

    return {
        "schema_version": 1,
        "status": "ok" if stable and not errors else "failed",
        "repository": repo,
        "default_branch": branch,
        "default_branch_sha": start_sha,
        "end_default_branch": end_branch,
        "end_default_branch_sha": end_sha,
        "snapshot_stable": stable,
        "observed_at": timestamp,
        "workflow_tree": {
            "count": len(present_paths),
            "paths": sorted(present_paths),
        },
        "pagination": pagination,
        "summary": _summary(records),
        "workflows": records,
        "errors": errors,
    }


def audit_workflow_registry(
    fetch_json: FetchJson,
    repo: str,
    *,
    observed_at: datetime | None = None,
    max_snapshot_attempts: int = 1,
) -> dict[str, Any]:
    """Build a SHA-bound audit, retrying moving or renamed defaults when requested."""
    attempts = max(
        1,
        _integer_argument(max_snapshot_attempts, name="max_snapshot_attempts"),
    )
    branch = _default_branch(fetch_json, repo)
    receipts: list[dict[str, Any]] = []
    audit: dict[str, Any] | None = None

    for attempt in range(1, attempts + 1):
        audit = _audit_workflow_registry_snapshot(
            fetch_json,
            repo,
            branch,
            observed_at=observed_at,
        )
        receipts.append(
            {
                "attempt": attempt,
                "start_sha": audit["default_branch_sha"],
                "end_sha": audit["end_default_branch_sha"],
                "stable": audit["snapshot_stable"],
            }
        )
        audit["snapshot_attempts"] = list(receipts)
        if audit["snapshot_stable"]:
            return audit
        branch = audit["end_default_branch"]

    if audit is None:
        raise RuntimeError("workflow registry audit produced no snapshot")
    return audit


def _parse_args() -> argparse.Namespace:
    """Parse the required repository slug and evidence output path."""
    parser = argparse.ArgumentParser(
        description="Audit GitHub Actions registry drift against protected default branch."
    )
    parser.add_argument(
        "--repo",
        required=True,
        type=_repo_slug,
        help="Repository in owner/name form.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    return parser.parse_args()


def main() -> int:
    """Run the live read-only workflow-registry audit."""
    args = _parse_args()
    try:
        audit = audit_workflow_registry(
            _run_gh_api,
            args.repo,
            max_snapshot_attempts=_MAX_ATTEMPTS,
        )
    except (GitHubApiError, RuntimeError, ValueError) as exc:
        audit = {
            "schema_version": 1,
            "status": "failed",
            "repository": args.repo,
            "observed_at": _utc_timestamp(),
            "errors": [str(exc)],
        }
        exit_code = 2
    else:
        exit_code = 0 if audit["status"] == "ok" else 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
