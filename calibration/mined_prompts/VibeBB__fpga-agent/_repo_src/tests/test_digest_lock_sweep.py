"""Drive scripts/digest_lock_sweep.sh through a gh stub.

The sweep merges stalled bot digest-lock PRs and backfills post-merge
dispatch when a bot merge landed outside a live publisher (bot merges do
not fire push events). These tests cover the mergeable-state decisions
and the backfill dispatch condition.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPOSITORY = "VibeBB/fpga-agent"
SCRIPT = Path(__file__).parents[1] / "scripts/digest_lock_sweep.sh"


def _run(
    tmp_path: Path,
    *,
    open_prs: list[int] | None = None,
    merged_prs: list[str] | None = None,
    mergeable: str = "clean",
    merge_fails: bool = False,
    run_count: int = 0,
) -> tuple[subprocess.CompletedProcess[str], str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$GH_STUB_CALLS"
case "$1 $2" in
  "pr list")
    if [[ "$*" == *"--state merged"* ]]; then
      printf '%s\\n' "$GH_STUB_MERGED_PRS"
    else
      printf '%s\\n' "$GH_STUB_OPEN_PRS"
    fi
    ;;
  "pr merge")
    if [[ "$*" != *"--auto"* ]] && [[ "$GH_STUB_MERGE_FAIL" == "1" ]]; then
      printf 'merge blocked\\n' >&2
      exit 1
    fi
    ;;
  api\\ *) printf '%s\\n' "$GH_STUB_MERGEABLE" ;;
  "run list") printf '%s\\n' "$GH_STUB_RUN_COUNT" ;;
  "workflow run") ;;
  *) echo "unexpected gh call: $*" >&2; exit 1 ;;
esac
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    calls = tmp_path / "calls.log"
    summary = tmp_path / "summary.md"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "GH_STUB_CALLS": str(calls),
            # The script passes --jq filters to gh; the stub emits the
            # already-filtered payloads (PR numbers / mergedAt lines).
            "GH_STUB_OPEN_PRS": "\n".join(str(n) for n in open_prs or []),
            "GH_STUB_MERGED_PRS": "\n".join(merged_prs or []),
            "GH_STUB_MERGEABLE": mergeable,
            "GH_STUB_MERGE_FAIL": "1" if merge_fails else "0",
            "GH_STUB_RUN_COUNT": str(run_count),
            "GITHUB_REPOSITORY": REPOSITORY,
            "GITHUB_STEP_SUMMARY": str(summary),
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT)], check=False, capture_output=True, text=True, env=env
    )
    return (
        result,
        calls.read_text(encoding="utf-8") if calls.exists() else "",
        summary.read_text(encoding="utf-8") if summary.exists() else "",
    )


def _merged_pr(hours_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_no_prs_no_backfill_is_noop(tmp_path: Path) -> None:
    result, log, summary = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "no open digest-lock PRs" in summary
    assert "pr merge" not in log
    assert "workflow run" not in log


def test_clean_pr_is_merged_and_post_merge_dispatched(tmp_path: Path) -> None:
    result, log, summary = _run(tmp_path, open_prs=[7])
    assert result.returncode == 0, result.stderr
    assert f"pr merge --repo {REPOSITORY} --squash --delete-branch 7" in log
    assert "--auto" not in log
    assert "workflow run ci.yml --repo" in log and "--ref main" in log
    assert "merged https://github.com/VibeBB/fpga-agent/pull/7" in summary


def test_blocked_pr_falls_back_to_auto_merge(tmp_path: Path) -> None:
    result, log, summary = _run(tmp_path, open_prs=[7], mergeable="blocked", merge_fails=True)
    assert result.returncode == 0, result.stderr
    assert f"pr merge --repo {REPOSITORY} --auto --squash --delete-branch 7" in log
    assert "auto-merge armed" in summary
    # Nothing merged this run and there is no uncovered recent merge, so
    # no post-merge dispatch happens; a later sweep's backfill covers the
    # armed auto-merge landing.
    assert "workflow run" not in log


def test_dirty_pr_is_skipped(tmp_path: Path) -> None:
    result, log, summary = _run(tmp_path, open_prs=[7], mergeable="dirty")
    assert result.returncode == 0, result.stderr
    assert "skipped (mergeable_state=dirty)" in summary
    assert "pr merge" not in log


def test_recent_uncovered_merge_triggers_dispatch(tmp_path: Path) -> None:
    result, log, summary = _run(tmp_path, merged_prs=[_merged_pr(1)], run_count=0)
    assert result.returncode == 0, result.stderr
    assert "workflow run ci.yml" in log
    assert "dispatched post-merge verification on main" in summary


def test_recent_covered_merge_is_not_redispatched(tmp_path: Path) -> None:
    result, log, _summary = _run(tmp_path, merged_prs=[_merged_pr(1)], run_count=1)
    assert result.returncode == 0, result.stderr
    assert "workflow run" not in log


def test_stale_merge_outside_window_is_not_redispatched(tmp_path: Path) -> None:
    result, log, _summary = _run(tmp_path, merged_prs=[_merged_pr(30)], run_count=0)
    assert result.returncode == 0, result.stderr
    assert "workflow run" not in log
