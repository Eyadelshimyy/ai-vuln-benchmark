#!/usr/bin/env python3
"""Capture a bounded live GitHub PR snapshot for queue governance.

The queue-governance builder needs nested changed-file, label, review, and merge
state for every open pull request. Asking GitHub for all of those nested fields
in one large ``gh pr list`` GraphQL request can exceed GitHub resource limits as
the queue grows. This module first enumerates only PR numbers, then enriches each
identity with a separate bounded request. The resulting JSON object is consumed
by ``build_pr_queue_governance.py --offline-snapshot``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Final, Sequence

try:
    from scripts._bounded_json import parse_json_bounded
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts", "scripts._bounded_json"}:
        raise
    try:
        from _bounded_json import parse_json_bounded
    except ModuleNotFoundError as sibling_exc:
        if sibling_exc.name != "_bounded_json":
            raise
        raise RuntimeError("bounded JSON parser is unavailable") from sibling_exc


OPEN_PR_DETAIL_FIELDS: Final = (
    "number,title,body,headRefName,headRefOid,baseRefName,isDraft,"
    "mergeStateStatus,reviewDecision,state,updatedAt,closedAt,mergedAt,"
    "url,labels,files"
)
OPEN_PR_DETAIL_REQUIRED_FIELDS: Final = frozenset(OPEN_PR_DETAIL_FIELDS.split(","))
HISTORY_PR_FIELDS: Final = (
    "number,title,body,headRefName,headRefOid,state,updatedAt,closedAt,"
    "mergedAt,url"
)
OPEN_PR_IDENTITY_FIELDS: Final = "number"
OPEN_PR_CAP: Final = 100
OPEN_PR_IDENTITY_LIMIT: Final = OPEN_PR_CAP + 1
HISTORY_PR_LIMIT: Final = 100
COMMAND_TIMEOUT_SECONDS: Final = 30
CAPTURE_BUDGET_SECONDS: Final = 420
MAX_ATTEMPTS: Final = 3
RETRY_SLEEP_SECONDS: Final = 0.5
_TRANSIENT_STATUS_RE: Final = re.compile(r"\bHTTP (?:502|503|504)\b", re.IGNORECASE)
_REPOSITORY_RE: Final = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

JsonRunner = Callable[[Sequence[str]], tuple[Any, dict[str, Any] | None]]
Clock = Callable[[], float]


def _command_identity(command: Sequence[str]) -> list[str]:
    """Return a short, non-secret command identity for audit evidence."""
    return [str(part) for part in command[1:3]]


def _command_error(
    command: Sequence[str],
    *,
    stderr: str,
    returncode: int,
) -> dict[str, Any]:
    """Create a normalized, bounded GitHub command failure record."""
    return {
        "command": _command_identity(command),
        "stderr": stderr.strip(),
        "returncode": returncode,
    }


def _run_gh_json(
    command: Sequence[str],
    *,
    max_attempts: int = MAX_ATTEMPTS,
    retry_sleep_seconds: float = RETRY_SLEEP_SECONDS,
    timeout_seconds: int = COMMAND_TIMEOUT_SECONDS,
) -> tuple[Any, dict[str, Any] | None]:
    """Run one GitHub CLI JSON command with bounded, status-specific retries.

    Only explicit HTTP 502, 503, and 504 responses are retried. Authentication,
    schema, rate-limit, parsing, and timeout failures remain fail-closed.
    """
    attempts = max(1, int(max_attempts))
    attempt = 1
    while True:
        try:
            completed = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, _command_error(
                command,
                stderr=f"GitHub command timed out after {timeout_seconds} seconds",
                returncode=124,
            )
        except OSError as exc:
            return None, _command_error(
                command,
                stderr=f"GitHub command could not start: {exc}",
                returncode=127,
            )

        if completed.returncode == 0:
            try:
                return parse_json_bounded(completed.stdout), None
            except (RuntimeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
                return None, _command_error(
                    command,
                    stderr=f"GitHub command returned invalid JSON: {exc}",
                    returncode=65,
                )

        stderr = completed.stderr.strip()
        error = _command_error(
            command,
            stderr=stderr,
            returncode=completed.returncode,
        )
        if attempt >= attempts or _TRANSIENT_STATUS_RE.search(stderr) is None:
            return None, error
        if retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds)
        attempt += 1


def _pr_list_command(
    repo: str,
    *,
    state: str,
    limit: int,
    fields: str,
) -> list[str]:
    """Build a bounded ``gh pr list`` command."""
    return [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        state,
        "--limit",
        str(limit),
        "--json",
        fields,
    ]


def _pr_view_command(repo: str, number: int) -> list[str]:
    """Build a full-detail command for one already-enumerated open PR."""
    return [
        "gh",
        "pr",
        "view",
        str(number),
        "--repo",
        repo,
        "--json",
        OPEN_PR_DETAIL_FIELDS,
    ]


def _positive_pr_number(value: object) -> int | None:
    """Return a positive non-Boolean PR number or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _normalized_list(payload: Any) -> list[dict[str, Any]]:
    """Return only object entries from one decoded JSON list."""
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _malformed_payload_error(command: Sequence[str], detail: str) -> dict[str, Any]:
    """Return a fail-closed error for a successful but malformed payload."""
    return _command_error(command, stderr=detail, returncode=65)


def _capture_budget_error(
    command: Sequence[str],
    capture_budget_seconds: float,
) -> dict[str, Any]:
    """Return a fail-closed error when cumulative live capture is exhausted."""
    return _command_error(
        command,
        stderr=(
            "PR queue capture exceeded the cumulative capture budget of "
            f"{capture_budget_seconds:g} seconds"
        ),
        returncode=124,
    )


def _incomplete_detail_error(
    command: Sequence[str],
    detail: dict[str, Any],
) -> dict[str, Any] | None:
    """Return an error when required queue classification evidence is incomplete."""
    missing = sorted(OPEN_PR_DETAIL_REQUIRED_FIELDS.difference(detail))
    if missing:
        return _malformed_payload_error(
            command,
            f"open PR detail omitted required fields: {', '.join(missing)}",
        )
    if not isinstance(detail.get("labels"), list) or not isinstance(detail.get("files"), list):
        return _malformed_payload_error(
            command,
            "open PR detail labels and files must be lists",
        )
    return None


def capture_pr_queue_snapshot(
    repo: str,
    *,
    run_json: JsonRunner = _run_gh_json,
    monotonic: Clock = time.monotonic,
    capture_budget_seconds: float = CAPTURE_BUDGET_SECONDS,
) -> dict[str, Any]:
    """Capture one bounded PR queue snapshot without a large nested query.

    Args:
        repo: GitHub repository in ``owner/name`` form.
        run_json: Injectable command runner used by focused tests.
        monotonic: Monotonic clock used to enforce the cumulative capture budget.
        capture_budget_seconds: Positive wall-clock budget for live capture.

    Returns:
        Snapshot compatible with ``build_pr_queue_governance.py``.

    Raises:
        ValueError: If ``repo`` or the cumulative capture budget is invalid.
    """
    if _REPOSITORY_RE.fullmatch(repo) is None:
        raise ValueError("repository must use canonical owner/name syntax")
    if isinstance(capture_budget_seconds, bool) or capture_budget_seconds <= 0:
        raise ValueError("capture_budget_seconds must be positive")
    deadline = monotonic() + capture_budget_seconds

    repo_command = [
        "gh",
        "repo",
        "view",
        repo,
        "--json",
        "nameWithOwner,defaultBranchRef,visibility,isArchived,pushedAt,updatedAt,url",
    ]
    identity_command = _pr_list_command(
        repo,
        state="open",
        limit=OPEN_PR_IDENTITY_LIMIT,
        fields=OPEN_PR_IDENTITY_FIELDS,
    )
    history_command = _pr_list_command(
        repo,
        state="all",
        limit=HISTORY_PR_LIMIT,
        fields=HISTORY_PR_FIELDS,
    )

    repo_payload, repo_error = run_json(repo_command)
    identity_payload, identity_error = run_json(identity_command)
    history_payload, history_error = run_json(history_command)
    errors = [
        error
        for error in (repo_error, identity_error, history_error)
        if error is not None
    ]

    default_branch = ""
    if isinstance(repo_payload, dict):
        branch = repo_payload.get("defaultBranchRef")
        if isinstance(branch, dict):
            default_branch = str(branch.get("name") or "").strip()
            if not default_branch:
                errors.append(
                    _malformed_payload_error(repo_command, "default branch name was missing")
                )
    elif repo_error is None:
        errors.append(_malformed_payload_error(repo_command, "repository payload was not an object"))

    identities = _normalized_list(identity_payload)
    identity_count = len(identity_payload) if isinstance(identity_payload, list) else 0
    if identity_error is None and not isinstance(identity_payload, list):
        errors.append(
            _malformed_payload_error(identity_command, "open PR identity payload was not a list")
        )
    elif identity_error is None and len(identities) != identity_count:
        errors.append(
            _malformed_payload_error(
                identity_command,
                "open PR identity payload contained non-object entries",
            )
        )
    if identity_count > OPEN_PR_CAP:
        errors.append(
            _command_error(
                identity_command,
                stderr=f"open PR count exceeds supported cap {OPEN_PR_CAP}",
                returncode=75,
            )
        )
        identities = []

    open_prs: list[dict[str, Any]] = []
    seen_numbers: set[int] = set()
    budget_exhausted = False
    for identity in identities:
        number = _positive_pr_number(identity.get("number"))
        if number is None or number in seen_numbers:
            errors.append(
                _malformed_payload_error(
                    identity_command,
                    "open PR identity contained an invalid or duplicate number",
                )
            )
            continue
        seen_numbers.add(number)
        detail_command = _pr_view_command(repo, number)
        if monotonic() >= deadline:
            errors.append(_capture_budget_error(detail_command, capture_budget_seconds))
            budget_exhausted = True
            break
        detail, detail_error = run_json(detail_command)
        if detail_error is not None:
            errors.append(detail_error)
            continue
        if not isinstance(detail, dict):
            errors.append(
                _malformed_payload_error(detail_command, "open PR detail payload was not an object")
            )
            continue
        if _positive_pr_number(detail.get("number")) != number:
            errors.append(
                _malformed_payload_error(detail_command, "open PR detail identity did not match")
            )
            continue
        if str(detail.get("state") or "").upper() != "OPEN":
            continue
        incomplete_error = _incomplete_detail_error(detail_command, detail)
        if incomplete_error is not None:
            errors.append(incomplete_error)
            continue
        open_prs.append(detail)

    if history_error is None and not isinstance(history_payload, list):
        errors.append(
            _malformed_payload_error(history_command, "PR history payload was not a list")
        )
    pr_history = _normalized_list(history_payload)
    if (
        history_error is None
        and isinstance(history_payload, list)
        and len(pr_history) != len(history_payload)
    ):
        errors.append(
            _malformed_payload_error(
                history_command,
                "PR history payload contained non-object entries",
            )
        )

    base_sha = ""
    if default_branch:
        base_command = [
            "gh",
            "api",
            f"repos/{repo}/commits/{default_branch}",
            "--jq",
            '{"sha": .sha}',
        ]
        if budget_exhausted:
            pass
        elif monotonic() >= deadline:
            errors.append(_capture_budget_error(base_command, capture_budget_seconds))
        else:
            base_payload, base_error = run_json(base_command)
            if base_error is not None:
                errors.append(base_error)
            elif isinstance(base_payload, dict):
                candidate = str(base_payload.get("sha") or "")
                if re.fullmatch(r"[0-9a-fA-F]{40}", candidate):
                    base_sha = candidate.lower()
                else:
                    errors.append(
                        _malformed_payload_error(base_command, "default-branch SHA was invalid")
                    )
            else:
                errors.append(
                    _malformed_payload_error(base_command, "default-branch payload was not an object")
                )

    return {
        "mode": "live-split-enrichment",
        "capture_version": 1,
        "repo": repo,
        "default_branch": default_branch,
        "base_sha": base_sha,
        "repo_snapshot": repo_payload,
        "open_pr_identity_count": identity_count,
        "open_prs": open_prs,
        "pr_history": pr_history,
        "errors": errors,
    }


def _write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    """Atomically publish one UTF-8 JSON snapshot in its destination directory."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_dir():
        raise ValueError("snapshot output must be a file path")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(snapshot, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Capture and publish a snapshot, returning failure when evidence is incomplete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub repository in owner/name form")
    parser.add_argument("--out", required=True, help="Destination JSON path")
    args = parser.parse_args(argv)
    try:
        snapshot = capture_pr_queue_snapshot(args.repo)
        _write_snapshot(Path(args.out), snapshot)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"capture_pr_queue_snapshot: {exc}", file=sys.stderr)
        return 2
    summary = {
        "errors": len(snapshot["errors"]),
        "open_pr_count": len(snapshot["open_prs"]),
        "out": str(Path(args.out).resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not snapshot["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())