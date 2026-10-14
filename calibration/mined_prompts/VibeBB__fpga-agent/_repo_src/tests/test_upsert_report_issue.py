"""Drive scripts/upsert_report_issue.sh through a gh stub.

The report workflows share this helper: a clean run must never create a
new issue (that is the create+close churn this removes), edits and closes
the canonical open issue, and collapses duplicates.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPOSITORY = "VibeBB/fpga-agent"
SCRIPT = Path(__file__).parents[1] / "scripts/upsert_report_issue.sh"


def _run(
    tmp_path: Path, *, clean: str, issues: str = ""
) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$GH_STUB_CALLS"
case "$1 $2" in
  "label create") ;;
  "issue list") printf '%s\\n' "$GH_STUB_ISSUES" ;;
  "issue create") printf 'https://github.com/%s/issues/99\\n' "$GITHUB_REPOSITORY" ;;
  "issue edit") ;;
  "issue close") ;;
  *) echo "unexpected gh call: $*" >&2; exit 1 ;;
esac
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    calls = tmp_path / "calls.log"
    report = tmp_path / "report.md"
    report.write_text("# report\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "GH_STUB_CALLS": str(calls),
            "GH_STUB_ISSUES": issues,
            "GITHUB_REPOSITORY": REPOSITORY,
            "LABEL": "dependency-updates",
            "TITLE": "Dependency update check report",
            "REPORT_MD": str(report),
            "CLEAN": clean,
            "CLOSE_COMMENT": "All checked dependencies are current.",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT)], check=False, capture_output=True, text=True, env=env
    )
    return result, calls.read_text(encoding="utf-8") if calls.exists() else ""


def test_clean_run_with_no_issue_never_creates(tmp_path: Path) -> None:
    result, log = _run(tmp_path, clean="1")
    assert result.returncode == 0, result.stderr
    assert "issue create" not in log
    assert "issue edit" not in log
    assert "issue close" not in log


def test_unclean_run_with_no_issue_creates(tmp_path: Path) -> None:
    result, log = _run(tmp_path, clean="0")
    assert result.returncode == 0, result.stderr
    assert "issue create" in log
    assert "--label dependency-updates" in log
    assert "issue close" not in log


def test_clean_run_closes_open_issue_after_edit(tmp_path: Path) -> None:
    result, log = _run(tmp_path, clean="1", issues="7\n")
    assert result.returncode == 0, result.stderr
    assert "issue edit 7" in log
    assert "issue close 7" in log
    assert "issue create" not in log


def test_unclean_run_edits_open_issue(tmp_path: Path) -> None:
    result, log = _run(tmp_path, clean="0", issues="7\n")
    assert result.returncode == 0, result.stderr
    assert "issue edit 7" in log
    assert "issue close" not in log


def test_duplicate_open_issues_collapsed(tmp_path: Path) -> None:
    result, log = _run(tmp_path, clean="0", issues="7\n11\n")
    assert result.returncode == 0, result.stderr
    assert "issue edit 7" in log
    assert "issue close 11" in log
    assert "Duplicate of the canonical report issue #7" in log
