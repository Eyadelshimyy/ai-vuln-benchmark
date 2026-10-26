"""Bash lane from a stored failure → issue → branch → PR → retry (issue #46).

The exercise is bash: this module builds the script. ``--apply`` may run the safe
openers (fetch + checkout -b, and optionally ``ops failure issue``) after HITL; it
never commits, pushes, merges, or writes the code fix.
"""

from __future__ import annotations

import re
import shlex

from jobbot.ops.failures import FailureRecord


def branch_name(record: FailureRecord) -> str:
    """Stable branch: fix/f0001-<8 hex of fingerprint>."""
    fid = record.id.strip().casefold()
    fp = re.sub(r"[^a-f0-9]", "", record.fingerprint.casefold())[:8] or "unknown"
    return f"fix/{fid}-{fp}"


def original_command(record: FailureRecord) -> str:
    """Command line to re-run after the fix (best effort from the record)."""
    cmd = (record.command or "").strip()
    if not cmd:
        return "uv run jobbot <original-command>"
    if cmd.startswith("uv "):
        return cmd
    if cmd.startswith("jobbot "):
        return f"uv run {cmd}"
    if cmd.startswith("python"):
        return cmd
    return f"uv run {cmd}"


def work_script(record: FailureRecord, *, issue_url: str | None = None) -> str:
    """Full bash script for the maintainer loop (printable dry-run)."""
    name = branch_name(record)
    linked = issue_url or record.issue_url
    rerun = original_command(record)
    if linked:
        issue_lines = f"# Issue already linked:\n# {linked}\n"
    else:
        issue_lines = (
            "# 1) Open the GitHub issue (HITL — review the proposed body)\n"
            f"uv run jobbot ops failure issue {record.id}\n"
        )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            f"# jobbot ops failure work {record.id} — maintainer lane (bash)",
            f"# fingerprint={record.fingerprint}  component={record.component}",
            "set -euo pipefail",
            "",
            issue_lines.rstrip(),
            "",
            "git fetch --all",
            "git checkout develop 2>/dev/null || git checkout main",
            "git pull --ff-only",
            f"git checkout -b {shlex.quote(name)}",
            "",
            "# --- you / the agent: fix here ---",
            "# 1) regression unit test that fails first",
            "# 2) implement the fix",
            "uv run ruff check .",
            "uv run mypy src",
            "uv run pytest",
            "git add -A",
            "git status",
            f'# git commit -m "fix({record.component}): {record.id} {record.error_class}"',
            "git push -u origin HEAD",
            "gh pr create --fill",
            f"uv run jobbot ops failure triage {record.id} --status fixed",
            "",
            "git fetch --all",
            "# goto 0 — reproduce:",
            rerun,
            "",
        ]
    )


def open_branch_commands(record: FailureRecord) -> list[list[str]]:
    """Argv lists for --apply: fetch + checkout -b only (no commit/push/PR)."""
    name = branch_name(record)
    return [
        ["git", "fetch", "--all"],
        ["git", "checkout", "develop"],
        ["git", "pull", "--ff-only"],
        ["git", "checkout", "-b", name],
    ]
