#!/usr/bin/env python
"""Build deterministic PR queue governance evidence for fast-mlsirm."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from html import escape
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from scripts._bounded_json import MAX_JSON_BYTES, parse_json_bounded, read_json_object
    from scripts._bounded_subprocess import BoundedSubprocessOutputError, run_bounded_capture
except ModuleNotFoundError:
    from _bounded_json import MAX_JSON_BYTES, parse_json_bounded, read_json_object
    from _bounded_subprocess import BoundedSubprocessOutputError, run_bounded_capture


RISK_COUNT_KEYS = [
    "changes_requested",
    "stale",
    "duplicate_candidate",
    "release_scope_conflict",
    "review_or_check_delay",
]

RELEASE_SCOPE_TERMS = {
    "backend",
    "cuda",
    "diagnostic",
    "diagnostics",
    "estimation",
    "estimator",
    "formula",
    "gpu",
    "gradient",
    "likelihood",
    "mlx",
    "model",
    "opencl",
    "pyo3",
    "rust",
}

DUPLICATE_TERMS = {
    "accessible table",
    "cli",
    "csp",
    "html report",
    "performance",
    "report",
    "softplus",
    "stack trace",
}

_CLOSING_REFERENCE_RE = re.compile(
    r"(?im)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+#(?P<issue>\d+)\b"
)
_CANONICAL_REFERENCE_RE = re.compile(
    r"(?im)^\s*canonical-for\s*:\s*#(?P<issue>\d+)\s*$"
)
_MIN_OVERLAP_FILES = 2
_HIGH_OVERLAP_THRESHOLD = 0.80

# Open-queue classification needs merge/review/file evidence.
_OPEN_PR_JSON_FIELDS = (
    "number,title,body,headRefName,headRefOid,baseRefName,isDraft,"
    "mergeStateStatus,reviewDecision,state,updatedAt,closedAt,mergedAt,"
    "url,labels,files"
)
# Audit-history only needs closing-reference and identity fields. Including
# nested ``files``/``labels`` on ``--state all --limit 100`` routinely trips
# GitHub GraphQL HTTP 502 for this repository (see Actions run 31374029017).
_HISTORY_PR_JSON_FIELDS = (
    "number,title,body,headRefName,headRefOid,state,updatedAt,closedAt,"
    "mergedAt,url"
)
_OPEN_PR_LIST_LIMIT = 100
_HISTORY_PR_LIST_LIMIT = 100
_GH_TRANSIENT_STATUS_RE = re.compile(r"\bHTTP (?:502|503|504)\b", re.IGNORECASE)
_GH_JSON_MAX_ATTEMPTS = 3
_GH_JSON_RETRY_SLEEP_SECONDS = 0.5
_GH_COMMAND_TIMEOUT_SECONDS = 60
_GH_STDOUT_MAX_BYTES = MAX_JSON_BYTES
_GH_STDERR_MAX_BYTES = 1024 * 1024
GIT_METADATA_TIMEOUT_SECONDS = 5
_FULL_OBJECT_ID_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one descriptor-safe bounded JSON object from disk."""
    return read_json_object(path)


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    """Resolve a path relative to a repository root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _source_commit(repo_root: Path) -> str:
    """Return the canonical full commit SHA for the checked-out HEAD.

    The value must reconstruct the exact governance-evidence source, so every
    failure mode fails closed: timeouts, missing executables, non-zero exits,
    empty output, and any identity that is not a full lowercase SHA-1 or
    SHA-256 object name raise :class:`RuntimeError` instead of degrading to an
    unreconstructable placeholder.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_METADATA_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("source commit lookup timed out") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("source commit lookup failed") from exc
    candidate = completed.stdout.strip()
    if not _FULL_OBJECT_ID_PATTERN.fullmatch(candidate):
        raise RuntimeError("source commit is not a full lowercase SHA-1 or SHA-256 object id")
    return candidate


def _check(
    name: str,
    category: str,
    ok: bool,
    detail: str,
    **metadata: Any,
) -> dict[str, Any]:
    """Create one normalized governance check record."""
    payload: dict[str, Any] = {
        "name": name,
        "category": category,
        "ok": ok,
        "detail": detail,
    }
    payload.update(metadata)
    return payload


def _json_from_completed(completed: subprocess.CompletedProcess[str]) -> Any:
    """Decode bounded command stdout when the command succeeded and emitted JSON."""
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return parse_json_bounded(completed.stdout, max_bytes=_GH_STDOUT_MAX_BYTES)


def _is_transient_gh_stderr(stderr: str) -> bool:
    """Return True when stderr reports an approved transient GitHub gateway status."""
    return _GH_TRANSIENT_STATUS_RE.search(stderr) is not None


def _run_gh_json(
    command: list[str],
    *,
    max_attempts: int = _GH_JSON_MAX_ATTEMPTS,
    retry_sleep_seconds: float = _GH_JSON_RETRY_SLEEP_SECONDS,
) -> tuple[Any, dict[str, Any] | None]:
    """Execute a GitHub CLI JSON command and return payload plus redacted error.

    Retries only on HTTP 502/503/504. Non-transient, bounded-output, and JSON
    decoding failures fail closed on the first response so real defects are not
    masked and untrusted command output cannot grow without bound in memory.
    """
    attempts = max(1, int(max_attempts))
    last_error: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        try:
            completed = run_bounded_capture(
                command,
                timeout_seconds=_GH_COMMAND_TIMEOUT_SECONDS,
                max_stdout_bytes=_GH_STDOUT_MAX_BYTES,
                max_stderr_bytes=_GH_STDERR_MAX_BYTES,
            )
        except subprocess.TimeoutExpired:
            last_error = {
                "command": command[1:3],
                "stderr": f"command timed out after {_GH_COMMAND_TIMEOUT_SECONDS} seconds",
                "returncode": 124,
            }
            break
        except BoundedSubprocessOutputError as exc:
            last_error = {
                "command": command[1:3],
                "stderr": str(exc),
                "returncode": 75,
            }
            break
        try:
            payload = _json_from_completed(completed)
        except ValueError as exc:
            last_error = {
                "command": command[1:3],
                "stderr": str(exc),
                "returncode": 65,
            }
            break
        if completed.returncode == 0:
            return payload, None
        stderr = completed.stderr.strip()
        last_error = {
            "command": command[1:3],
            "stderr": stderr,
            "returncode": completed.returncode,
        }
        if attempt >= attempts or not _is_transient_gh_stderr(stderr):
            break
        if retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds)
    return None, last_error


def _normalize_pr_list(payload: Any) -> list[dict[str, Any]]:
    """Normalize a GitHub CLI PR list payload."""
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _pr_list_command(
    repo: str,
    *,
    state: str,
    limit: int,
    fields: str,
) -> list[str]:
    """Build a ``gh pr list`` JSON command for one bounded state snapshot."""
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


def _run_gh_snapshot(repo: str) -> dict[str, Any]:
    """Capture open PR state, bounded audit history, and the default-base SHA."""
    repo_payload, repo_error = _run_gh_json(
        [
            "gh",
            "repo",
            "view",
            repo,
            "--json",
            "nameWithOwner,defaultBranchRef,visibility,isArchived,pushedAt,updatedAt,url",
        ]
    )
    open_payload, open_error = _run_gh_json(
        _pr_list_command(
            repo,
            state="open",
            limit=_OPEN_PR_LIST_LIMIT,
            fields=_OPEN_PR_JSON_FIELDS,
        )
    )
    history_payload, history_error = _run_gh_json(
        _pr_list_command(
            repo,
            state="all",
            limit=_HISTORY_PR_LIST_LIMIT,
            fields=_HISTORY_PR_JSON_FIELDS,
        )
    )

    default_branch = ""
    if isinstance(repo_payload, dict):
        branch_ref = repo_payload.get("defaultBranchRef")
        if isinstance(branch_ref, dict):
            default_branch = str(branch_ref.get("name", ""))

    base_sha = ""
    base_error = None
    if default_branch:
        base_payload, base_error = _run_gh_json(
            [
                "gh",
                "api",
                f"repos/{repo}/commits/{default_branch}",
                "--jq",
                '{"sha": .sha}',
            ]
        )
        if isinstance(base_payload, dict):
            base_sha = str(base_payload.get("sha", ""))

    errors = [
        error
        for error in (repo_error, open_error, history_error, base_error)
        if error is not None
    ]
    return {
        "mode": "live",
        "repo": repo,
        "default_branch": default_branch,
        "base_sha": base_sha,
        "repo_snapshot": repo_payload,
        "open_prs": _normalize_pr_list(open_payload),
        "pr_history": _normalize_pr_list(history_payload),
        "errors": errors,
    }


def _snapshot_from_args(
    args: argparse.Namespace,
    repo_root: Path,
) -> dict[str, Any]:
    """Load an offline fixture or capture live GitHub state."""
    snapshot_path = getattr(args, "offline_snapshot", None)
    if snapshot_path:
        path = _resolve_path(snapshot_path, base=repo_root).resolve()
        snapshot = _read_json(path)
        snapshot.setdefault("mode", "offline")
        snapshot.setdefault("repo", args.repo)
        snapshot.setdefault("base_sha", "")
        snapshot.setdefault("errors", [])
        snapshot.setdefault("pr_history", snapshot.get("open_prs", []))
        snapshot["snapshot_file"] = str(path)
        return snapshot
    if getattr(args, "offline_github", False):
        return {
            "mode": "offline",
            "repo": args.repo,
            "default_branch": "",
            "base_sha": "",
            "open_prs": [],
            "pr_history": [],
            "errors": [
                {
                    "command": "snapshot",
                    "stderr": "offline mode requires --offline-snapshot",
                    "returncode": 2,
                }
            ],
        }
    return _run_gh_snapshot(args.repo)


def _extract_prs(snapshot: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Extract normalized PR objects from a snapshot field."""
    payload = snapshot.get(key)
    if isinstance(payload, list):
        return [pr for pr in payload if isinstance(pr, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [pr for pr in data if isinstance(pr, dict)]
    return []


def _extract_open_prs(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract active open PR objects from a snapshot."""
    return _extract_prs(snapshot, "open_prs")


def _extract_pr_history(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract bounded open/closed/merged PR audit history."""
    history = _extract_prs(snapshot, "pr_history")
    return history if history else _extract_open_prs(snapshot)


def _issue_references(body: object) -> list[int]:
    """Return sorted issue numbers referenced by closing keywords."""
    if not isinstance(body, str):
        return []
    return sorted(
        {int(match.group("issue")) for match in _CLOSING_REFERENCE_RE.finditer(body)}
    )


def _canonical_issue_references(body: object) -> list[int]:
    """Return sorted issue numbers explicitly designated by ``Canonical-For``."""
    if not isinstance(body, str):
        return []
    return sorted(
        {int(match.group("issue")) for match in _CANONICAL_REFERENCE_RE.finditer(body)}
    )


def _label_names(pr: dict[str, Any]) -> list[str]:
    """Return normalized label names from a GitHub PR payload."""
    labels = pr.get("labels")
    if not isinstance(labels, list):
        return []
    names: set[str] = set()
    for label in labels:
        if isinstance(label, str):
            candidate = label
        elif isinstance(label, dict):
            candidate = str(label.get("name", ""))
        else:
            continue
        candidate = candidate.strip()
        if candidate:
            names.add(candidate)
    return sorted(names)


def _changed_files(pr: dict[str, Any]) -> list[str]:
    """Return sorted changed paths from GitHub CLI file records."""
    files = pr.get("files")
    if not isinstance(files, list):
        return []
    paths: set[str] = set()
    for item in files:
        if isinstance(item, str):
            candidate = item
        elif isinstance(item, dict):
            candidate = str(item.get("path", ""))
        else:
            continue
        candidate = candidate.strip()
        if candidate:
            paths.add(candidate)
    return sorted(paths)


def classify_pr(
    pr: dict[str, Any],
    *,
    now: datetime,
    max_stale_days: int,
) -> dict[str, Any]:
    """Classify one open PR for queue, release-boundary, and audit risks."""
    title = str(pr.get("title", ""))
    head = str(pr.get("headRefName", ""))
    text = f"{title} {head}".lower()
    review_decision = str(pr.get("reviewDecision") or "").upper()
    merge_state = str(pr.get("mergeStateStatus") or "").upper()
    updated_at = str(pr.get("updatedAt") or "")
    updated_dt = None
    age_days = None
    if updated_at:
        try:
            updated_dt = _parse_datetime(updated_at)
            age_days = max(0, (now - updated_dt).days)
        except ValueError:
            updated_dt = None

    release_scope_terms = sorted(term for term in RELEASE_SCOPE_TERMS if term in text)
    duplicate_terms = sorted(term for term in DUPLICATE_TERMS if term in text)
    changes_requested = review_decision == "CHANGES_REQUESTED"
    stale = age_days is not None and age_days > max_stale_days
    review_or_check_delay = (
        review_decision == "REVIEW_REQUIRED" or merge_state == "QUEUED"
    )
    release_scope_conflict = bool(release_scope_terms)
    duplicate_candidate = bool(duplicate_terms) and not release_scope_conflict

    risk_reasons: list[str] = []
    if changes_requested:
        risk_reasons.append("changes_requested_review")
    if stale:
        risk_reasons.append("stale_update")
    if duplicate_candidate:
        risk_reasons.append("duplicate_or_already_productized_scope")
    if release_scope_conflict:
        risk_reasons.append("model_or_backend_scope")
    if review_or_check_delay:
        risk_reasons.append("review_or_check_delay")
    if merge_state == "BLOCKED":
        risk_reasons.append("merge_blocked")

    classified = dict(pr)
    classified.update(
        {
            "age_days": age_days,
            "changes_requested": changes_requested,
            "stale": stale,
            "duplicate_candidate": duplicate_candidate,
            "release_scope_conflict": release_scope_conflict,
            "review_or_check_delay": review_or_check_delay,
            "release_scope_terms": release_scope_terms,
            "duplicate_terms": duplicate_terms,
            "risk_reasons": risk_reasons,
            "closing_issue_references": _issue_references(pr.get("body")),
            "canonical_issue_references": _canonical_issue_references(pr.get("body")),
            "label_names": _label_names(pr),
            "changed_files": _changed_files(pr),
        }
    )
    if updated_dt is not None:
        classified["updated_at_utc"] = updated_dt.isoformat(timespec="seconds")
    return classified


def _risk_counts(classified_prs: list[dict[str, Any]]) -> dict[str, int]:
    """Count PR-level risk flags."""
    return {
        key: sum(1 for pr in classified_prs if pr.get(key) is True)
        for key in RISK_COUNT_KEYS
    }


def _claimant_record(pr: dict[str, Any], issue_number: int) -> dict[str, Any]:
    """Create a compact deterministic issue-claimant record."""
    canonical_refs = pr.get("canonical_issue_references", [])
    return {
        "pr_number": int(pr.get("number", 0)),
        "title": str(pr.get("title", "")),
        "url": str(pr.get("url", "")),
        "head_ref": str(pr.get("headRefName", "")),
        "head_sha": str(pr.get("headRefOid", "")),
        "updated_at": str(pr.get("updatedAt", "")),
        "canonical_for_issue": issue_number in canonical_refs,
    }


def _duplicate_issue_claims(
    classified_prs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map multiply claimed issues to canonical or unresolved conflict state."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for pr in classified_prs:
        for issue_number in pr.get("closing_issue_references", []):
            grouped.setdefault(int(issue_number), []).append(pr)

    conflicts: dict[str, dict[str, Any]] = {}
    for issue_number, claimants in sorted(grouped.items()):
        if len(claimants) < 2:
            continue
        records = sorted(
            (_claimant_record(pr, issue_number) for pr in claimants),
            key=lambda record: record["pr_number"],
        )
        designated = [
            record["pr_number"] for record in records if record["canonical_for_issue"]
        ]
        canonical_pr = designated[0] if len(designated) == 1 else None
        conflicts[str(issue_number)] = {
            "issue_number": issue_number,
            "claimant_prs": records,
            "canonical_pr": canonical_pr,
            "status": "resolved" if canonical_pr is not None else "conflict",
            "reason": (
                "one claimant is explicitly designated with Canonical-For"
                if canonical_pr is not None
                else "multiple open PRs claim the issue without exactly one canonical designation"
            ),
        }
    return conflicts


def _duplicate_head_warnings(
    classified_prs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect exact duplicate head branches among active PRs."""
    grouped: dict[str, list[int]] = {}
    for pr in classified_prs:
        head = str(pr.get("headRefName", "")).strip()
        if head:
            grouped.setdefault(head, []).append(int(pr.get("number", 0)))
    return [
        {"head_ref": head, "pr_numbers": sorted(numbers)}
        for head, numbers in sorted(grouped.items())
        if len(numbers) > 1
    ]


def _changed_file_overlap_warnings(
    classified_prs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect identical or high-overlap changed-file sets without one-file noise."""
    warnings: list[dict[str, Any]] = []
    ordered = sorted(classified_prs, key=lambda pr: int(pr.get("number", 0)))
    for left, right in combinations(ordered, 2):
        left_files = set(left.get("changed_files", []))
        right_files = set(right.get("changed_files", []))
        if min(len(left_files), len(right_files)) < _MIN_OVERLAP_FILES:
            continue
        intersection = sorted(left_files & right_files)
        union = sorted(left_files | right_files)
        score = len(intersection) / len(union)
        identical = left_files == right_files
        if not identical and score < _HIGH_OVERLAP_THRESHOLD:
            continue
        warnings.append(
            {
                "pr_numbers": [
                    int(left.get("number", 0)),
                    int(right.get("number", 0)),
                ],
                "identical": identical,
                "jaccard": round(score, 6),
                "intersection": intersection,
                "union_size": len(union),
            }
        )
    return warnings


def _issue_claim_history(pr_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain closing-reference history while excluding inactive PRs from conflicts."""
    records: list[dict[str, Any]] = []
    for pr in pr_history:
        if not isinstance(pr, dict):
            continue
        references = _issue_references(pr.get("body"))
        if not references:
            continue
        records.append(
            {
                "pr_number": int(pr.get("number", 0)),
                "state": str(pr.get("state", "")),
                "merged_at": str(pr.get("mergedAt") or ""),
                "closed_at": str(pr.get("closedAt") or ""),
                "updated_at": str(pr.get("updatedAt") or ""),
                "head_ref": str(pr.get("headRefName", "")),
                "head_sha": str(pr.get("headRefOid", "")),
                "issue_references": references,
                "url": str(pr.get("url", "")),
            }
        )
    return sorted(records, key=lambda record: record["pr_number"])


def _safe_url(url: object) -> str:
    """Return a safe report URL or ``#`` for disallowed schemes."""
    if not isinstance(url, str):
        return "#"
    candidate = url.strip()
    if not candidate:
        return "#"
    parsed = urlparse(candidate)
    if parsed.scheme and parsed.scheme.lower() not in {"http", "https", "mailto"}:
        return "#"
    return candidate


def _content_security_policy() -> str:
    """Return the standalone report content-security policy."""
    return (
        "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    )


def _report_css() -> str:
    """Return accessible standalone report CSS."""
    return """
:root { color: #172026; background: #f5f7f8; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }
.hero { background: #12343b; color: #fff; border-radius: 8px; padding: 28px; }
.hero p, .hero h1 { margin: 0; }
.hero p { color: #b7d7d0; font-size: 0.86rem; font-weight: 700; text-transform: uppercase; }
.hero h1 { margin-top: 8px; font-size: 2rem; }
.hero span { display: inline-block; margin-top: 14px; color: #dce8e5; overflow-wrap: anywhere; }
.report-section { margin-top: 22px; background: #fff; border: 1px solid #d8e1e3; border-radius: 8px; padding: 22px; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.metric-card { border: 1px solid #d8e1e3; border-radius: 8px; padding: 14px; }
.metric-card span { display: block; color: #5e6f76; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; }
.metric-card strong { display: block; margin-top: 8px; overflow-wrap: anywhere; }
.table-wrap { overflow-x: auto; border: 1px solid #d8e1e3; border-radius: 8px; }
.table-wrap:focus-visible { outline: 3px solid #0f766e; outline-offset: 3px; }
table { width: 100%; min-width: 920px; border-collapse: collapse; }
caption {
  position: absolute; width: 1px; height: 1px; margin: -1px;
  overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap;
}
th, td {
  padding: 10px 12px; border-bottom: 1px solid #e8edef;
  text-align: left; vertical-align: top; font-variant-numeric: tabular-nums;
}
code { overflow-wrap: anywhere; }
.note { color: #5e6f76; margin-bottom: 0; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition-duration: 0.01ms !important; } }
@media print {
  * { print-color-adjust: exact !important; }
  .report-section, .metric-card, .table-wrap { break-inside: avoid; }
}
"""


def _table(
    *,
    label: str,
    caption: str,
    headings: list[str],
    rows: list[str],
) -> list[str]:
    """Return accessible table markup lines."""
    return [
        f'<div class="table-wrap" role="region" aria-label="{escape(label, quote=True)}" tabindex="0">',
        f"<table><caption>{escape(caption)}</caption>",
        "<thead><tr>"
        + "".join(f'<th scope="col">{escape(heading)}</th>' for heading in headings)
        + "</tr></thead><tbody>",
        *rows,
        "</tbody></table></div>",
    ]


def _render_report(manifest: dict[str, Any]) -> str:
    """Render deterministic standalone HTML governance evidence."""
    risk_counts = manifest.get("risk_counts", {})
    duplicate_claims = manifest.get("duplicate_issue_claims", {})
    unresolved_count = int(manifest.get("unresolved_duplicate_issue_claim_count", 0))
    cards = [
        ("Status", manifest.get("status", "")),
        ("Open PRs", manifest.get("open_pr_count", "")),
        ("Base SHA", manifest.get("base_sha", "")),
        ("Duplicate conflicts", unresolved_count),
        ("Duplicate heads", len(manifest.get("duplicate_head_warnings", []))),
        (
            "File-overlap warnings",
            len(manifest.get("changed_file_overlap_warnings", [])),
        ),
        (
            "Changes requested",
            risk_counts.get("changes_requested", "")
            if isinstance(risk_counts, dict)
            else "",
        ),
        (
            "Review delay",
            risk_counts.get("review_or_check_delay", "")
            if isinstance(risk_counts, dict)
            else "",
        ),
    ]
    card_markup = [
        '<article class="metric-card">'
        f"<span>{escape(label)}</span><strong>{escape(str(value))}</strong></article>"
        for label, value in cards
    ]

    pr_rows: list[str] = []
    for pr in manifest.get("pull_requests", []):
        if not isinstance(pr, dict):
            continue
        pr_rows.append(
            "<tr>"
            f'<th scope="row"><a href="{escape(_safe_url(pr.get("url")), quote=True)}">'
            f"#{escape(str(pr.get('number', '')))}</a></th>"
            f"<td>{escape(str(pr.get('title', '')))}</td>"
            f"<td><code>{escape(str(pr.get('headRefOid', '')))}</code></td>"
            f"<td>{escape(', '.join(f'#{item}' for item in pr.get('closing_issue_references', [])))}</td>"
            f"<td>{escape(str(pr.get('reviewDecision', '')))}</td>"
            f"<td>{escape(str(pr.get('mergeStateStatus', '')))}</td>"
            f"<td>{escape(', '.join(str(reason) for reason in pr.get('risk_reasons', [])))}</td>"
            "</tr>"
        )

    claim_rows: list[str] = []
    if isinstance(duplicate_claims, dict):
        for issue_key in sorted(duplicate_claims, key=lambda key: int(key)):
            claim = duplicate_claims[issue_key]
            if not isinstance(claim, dict):
                continue
            claimants = claim.get("claimant_prs", [])
            claimant_text = ", ".join(
                f"#{record.get('pr_number')} ({record.get('head_sha', '')})"
                for record in claimants
                if isinstance(record, dict)
            )
            claim_rows.append(
                "<tr>"
                f'<th scope="row">#{escape(str(claim.get("issue_number", "")))}</th>'
                f"<td>{escape(str(claim.get('status', '')))}</td>"
                f"<td>{escape(str(claim.get('canonical_pr') or ''))}</td>"
                f"<td>{escape(claimant_text)}</td>"
                f"<td>{escape(str(claim.get('reason', '')))}</td>"
                "</tr>"
            )

    warning_rows: list[str] = []
    for warning in manifest.get("changed_file_overlap_warnings", []):
        if not isinstance(warning, dict):
            continue
        warning_rows.append(
            "<tr>"
            f'<th scope="row">{escape(", ".join(f"#{number}" for number in warning.get("pr_numbers", [])))}</th>'
            f"<td>{escape(str(warning.get('identical', '')))}</td>"
            f"<td>{escape(str(warning.get('jaccard', '')))}</td>"
            f"<td>{escape(', '.join(str(path) for path in warning.get('intersection', [])))}</td>"
            "</tr>"
        )

    history_rows: list[str] = []
    for record in manifest.get("issue_claim_history", []):
        if not isinstance(record, dict):
            continue
        history_rows.append(
            "<tr>"
            f'<th scope="row">#{escape(str(record.get("pr_number", "")))}</th>'
            f"<td>{escape(str(record.get('state', '')))}</td>"
            f"<td><code>{escape(str(record.get('head_sha', '')))}</code></td>"
            f"<td>{escape(', '.join(f'#{item}' for item in record.get('issue_references', [])))}</td>"
            f"<td>{escape(str(record.get('updated_at', '')))}</td>"
            f"<td>{escape(str(record.get('closed_at', '')))}</td>"
            f"<td>{escape(str(record.get('merged_at', '')))}</td>"
            "</tr>"
        )

    check_rows: list[str] = []
    for check in manifest.get("checks", []):
        if not isinstance(check, dict):
            continue
        check_rows.append(
            "<tr>"
            f'<th scope="row">{escape(str(check.get("name", "")))}</th>'
            f"<td>{escape(str(check.get('category', '')))}</td>"
            f"<td>{escape('go' if check.get('ok') else 'failed')}</td>"
            f"<td>{escape(str(check.get('detail', '')))}</td>"
            "</tr>"
        )

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f'<meta http-equiv="Content-Security-Policy" content="{escape(_content_security_policy(), quote=True)}">',
            "<title>fast-mlsirm PR Queue Governance</title>",
            "<style>",
            _report_css(),
            "</style>",
            "</head>",
            "<body><main>",
            '<section class="hero"><p>fast-mlsirm buyer governance</p><h1>PR Queue Governance</h1>',
            f"<span>Repository: {escape(str(manifest.get('repo', '')))} · "
            f"Base: {escape(str(manifest.get('base_sha', '')))} · "
            f"Generated: {escape(str(manifest.get('generated_at', '')))}</span></section>",
            '<section class="report-section"><h2>Queue Summary</h2><div class="metrics-grid">',
            *card_markup,
            "</div></section>",
            '<section class="report-section"><h2>Open PR Risk Classification</h2>',
            *_table(
                label="PR queue governance table",
                caption="PR queue governance table",
                headings=[
                    "PR",
                    "Title",
                    "Head SHA",
                    "Issue references",
                    "Review",
                    "Merge",
                    "Risk reasons",
                ],
                rows=pr_rows,
            ),
            "</section>",
            '<section class="report-section"><h2>Duplicate Issue Claims</h2>',
            *_table(
                label="Duplicate issue claim table",
                caption="Duplicate issue claim table",
                headings=[
                    "Issue",
                    "Status",
                    "Canonical PR",
                    "Active claimants",
                    "Reason",
                ],
                rows=claim_rows,
            ),
            '<p class="note">A duplicate issue claim is resolved only when exactly one active claimant includes '
            "<code>Canonical-For: #issue</code> in its body.</p></section>",
            '<section class="report-section"><h2>Changed-file Overlap Warnings</h2>',
            *_table(
                label="Changed-file overlap warning table",
                caption="Changed-file overlap warning table",
                headings=["PRs", "Identical", "Jaccard", "Shared files"],
                rows=warning_rows,
            ),
            '<p class="note">One-file intersections are excluded to avoid treating '
            "shared central files as duplicates.</p>",
            "</section>",
            '<section class="report-section"><h2>Issue Claim Audit History</h2>',
            *_table(
                label="Issue claim audit history table",
                caption="Issue claim audit history table",
                headings=[
                    "PR",
                    "State",
                    "Head SHA",
                    "Issue references",
                    "Updated",
                    "Closed",
                    "Merged",
                ],
                rows=history_rows,
            ),
            "</section>",
            '<section class="report-section"><h2>Evidence Checks</h2>',
            *_table(
                label="PR queue governance check table",
                caption="PR queue governance check table",
                headings=["Check", "Category", "Status", "Detail"],
                rows=check_rows,
            ),
            "</section></main></body></html>",
        ]
    )


def build_pr_queue_governance(args: argparse.Namespace) -> dict[str, Any]:
    """Build machine-readable and accessible PR queue governance evidence."""
    repo_root = Path(args.repo_root).resolve()
    out_dir = _resolve_path(args.out, base=repo_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = getattr(args, "generated_at", None) or datetime.now(UTC).isoformat(
        timespec="seconds"
    )
    now = _parse_datetime(generated_at)
    snapshot = _snapshot_from_args(args, repo_root)
    open_prs = _extract_open_prs(snapshot)
    history = _extract_pr_history(snapshot)
    classified = [
        classify_pr(
            pr,
            now=now,
            max_stale_days=int(getattr(args, "max_stale_days", 14)),
        )
        for pr in open_prs
    ]
    risk_counts = _risk_counts(classified)
    duplicate_claims = _duplicate_issue_claims(classified)
    unresolved_claims = [
        claim
        for claim in duplicate_claims.values()
        if claim.get("status") == "conflict"
    ]
    duplicate_heads = _duplicate_head_warnings(classified)
    overlap_warnings = _changed_file_overlap_warnings(classified)
    claim_history = _issue_claim_history(history)

    snapshot_errors = snapshot.get("errors")
    snapshot_ok = isinstance(open_prs, list) and snapshot_errors == []
    base_sha = str(snapshot.get("base_sha", ""))
    base_sha_ok = bool(base_sha) or snapshot.get("mode") == "offline"

    checks = [
        _check(
            "github:snapshot",
            "github",
            snapshot_ok,
            "GitHub open PR and audit-history snapshots were recorded",
            mode=snapshot.get("mode"),
            errors=snapshot_errors,
        ),
        _check(
            "github:base_sha",
            "github",
            base_sha_ok,
            "default-branch base SHA is recorded for live evidence",
            base_sha=base_sha,
        ),
        _check(
            "queue:classified",
            "queue_state",
            len(classified) == len(open_prs),
            "every open PR has governance classification fields",
            open_pr_count=len(open_prs),
        ),
        _check(
            "risk:coverage",
            "risk_classification",
            set(RISK_COUNT_KEYS).issubset(risk_counts),
            "risk count coverage includes all established queue categories",
            risk_count_keys=sorted(risk_counts),
        ),
        _check(
            "queue:duplicate_issue_claims",
            "queue_state",
            not unresolved_claims,
            (
                "no unresolved duplicate issue claims exist"
                if not unresolved_claims
                else (
                    f"{len(unresolved_claims)} duplicate issue claim conflict(s) "
                    "require canonical designation or closure"
                )
            ),
            conflicting_issue_numbers=sorted(
                int(claim["issue_number"]) for claim in unresolved_claims
            ),
        ),
        _check(
            "queue:audit_history",
            "audit",
            isinstance(claim_history, list),
            "closed and merged issue claims are retained in bounded audit history",
            claim_history_count=len(claim_history),
        ),
        _check(
            "release:boundary",
            "release_boundary",
            True,
            "review delays are evidence, while unresolved duplicate issue claims fail the commercial release gate",
        ),
    ]
    failed = [check for check in checks if not check["ok"]]
    manifest: dict[str, Any] = {
        "command": "build_pr_queue_governance",
        "status": "ok" if not failed else "failed",
        "contract_value_krw": args.contract_value_krw,
        "generated_at": generated_at,
        "source_commit": _source_commit(repo_root),
        "repo_root": str(repo_root),
        "repo": args.repo,
        "default_branch": snapshot.get("default_branch", ""),
        "base_sha": base_sha,
        "max_stale_days": int(getattr(args, "max_stale_days", 14)),
        "open_pr_count": len(classified),
        "risk_counts": risk_counts,
        "duplicate_issue_claims": duplicate_claims,
        "unresolved_duplicate_issue_claim_count": len(unresolved_claims),
        "duplicate_head_warnings": duplicate_heads,
        "changed_file_overlap_warnings": overlap_warnings,
        "issue_claim_history": claim_history,
        "github": snapshot,
        "pull_requests": classified,
        "checks": checks,
        "failed_checks": failed,
    }
    html_path = out_dir / "pr_queue_governance_report.html"
    manifest_path = out_dir / "pr_queue_governance_manifest.json"
    html_path.write_text(_render_report(manifest), encoding="utf-8")
    manifest["html_report_file"] = str(html_path)
    manifest["html_report_sha256"] = _sha256(html_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build PR queue governance evidence for fast-mlsirm."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--out",
        default="pr-queue-governance",
        help="Output directory.",
    )
    parser.add_argument(
        "--repo",
        default="ContextualWisdomLab/fast-mlsirm",
        help="GitHub repository name.",
    )
    parser.add_argument(
        "--contract-value-krw",
        type=int,
        default=2_000_000_000,
        help="Target contract value.",
    )
    parser.add_argument(
        "--offline-snapshot",
        help="JSON snapshot with open_prs and optional pr_history.",
    )
    parser.add_argument(
        "--offline-github",
        action="store_true",
        help="Fail fast unless --offline-snapshot is supplied.",
    )
    parser.add_argument(
        "--max-stale-days",
        type=int,
        default=14,
        help="Age in days after which an open PR is stale.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the governance builder CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_pr_queue_governance(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "out": str(Path(args.out).resolve()),
                "manifest": str(
                    Path(args.out).resolve() / "pr_queue_governance_manifest.json"
                ),
                "html": manifest["html_report_file"],
                "open_pr_count": manifest["open_pr_count"],
                "failed_checks": len(manifest["failed_checks"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
