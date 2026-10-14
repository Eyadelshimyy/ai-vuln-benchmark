from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = f"VibeBB/{REPO_ROOT.name}"
SCRIPT = REPO_ROOT / "scripts/release_bump.sh"
PR_URL = f"https://github.com/{REPOSITORY}/pull/77"

SKILLS = (
    "fpga-contract",
    "fpga-device-pins",
    "fpga-external-libraries",
    "fpga-sibling-cooperation",
    "fpga-verification",
    "fpga-workflow",
)

GH_STUB = """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$GH_STUB_CALLS"
case "$1" in
  pr)
    case "$2" in
      create)
        printf '%s\\n' "$GH_STUB_PR_URL"
        ;;
      merge)
        # Simulate the server-side squash merge: the bot branch was already
        # pushed to the bare origin, so copy it onto main there (bypasses
        # the test pre-receive hook, which only runs on receive-pack).
        if [ -n "${GH_STUB_MERGE_FETCH:-}" ]; then
          git -C "$GH_STUB_BARE" fetch -q . \
            "+refs/heads/${GH_STUB_BRANCH}:refs/heads/main"
        fi
        ;;
    esac
    ;;
  run)
    case "$2" in
      list)
        case "$*" in
          *action_required*) ;;
          *) printf '%s\\n' "${GH_STUB_RUN_ID:-9001}" ;;
        esac
        ;;
      view)
        printf '%s\\n' "${GH_STUB_CONCLUSION:-success}"
        ;;
      watch) ;;
    esac
    ;;
  workflow) ;;
  api)
    case "$*" in
      *pulls/*) printf '%s\\n' "${GH_STUB_MERGED_AT:-2026-10-04T00:00:00Z}" ;;
      *) printf '77\\n' ;;
    esac
    ;;
esac
"""

REJECT_MAIN_HOOK = """#!/usr/bin/env bash
# Test stand-in for the branch ruleset: reject direct pushes to main,
# accept everything else (bot/* branches included).
while read -r _old _new ref; do
  if [ "$ref" = "refs/heads/main" ]; then
    echo "push to main rejected by test hook" >&2
    exit 1
  fi
done
"""


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return proc.stdout.strip()


def _write_version_files(work: Path, version: str) -> None:
    for rel in (
        "plugins/fpga/.plugin",
        "scripts",
        *(f"plugins/fpga/skills/{skill}" for skill in SKILLS),
    ):
        (work / rel).mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / "scripts/bump_version.py", work / "scripts/bump_version.py")
    (work / "plugins/fpga/.plugin/plugin.json").write_text(
        f'{{\n  "version": "{version}"\n}}\n', encoding="utf-8"
    )
    (work / "pyproject.toml").write_text(
        f'[project]\nname = "fpga-agent"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    for skill in SKILLS:
        (work / f"plugins/fpga/skills/{skill}/SKILL.md").write_text(
            f"---\nversion: {version}\n---\n", encoding="utf-8"
        )
    (work / "uv.lock").write_text(
        f'[[package]]\nname = "fpga-agent"\nversion = "{version}"\n',
        encoding="utf-8",
    )


@pytest.fixture
def release_repo(tmp_path: Path) -> tuple[Path, Path, dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(GH_STUB, encoding="utf-8")
    gh.chmod(0o755)

    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-b", "main")
    _git(work, "config", "user.name", "Test")
    _git(work, "config", "user.email", "test@example.com")
    _write_version_files(work, "1.2.3")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "initial")

    bare = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", str(bare)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    _git(work, "remote", "add", "origin", str(bare))
    _git(work, "push", "-u", "origin", "main")

    calls = tmp_path / "calls.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "BUMP": "patch",
            "SET_VERSION": "",
            "DRY_RUN": "false",
            "GH_TOKEN": "stub",
            "GITHUB_REPOSITORY": REPOSITORY,
            "GITHUB_OUTPUT": str(tmp_path / "github_output"),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
            "GITHUB_RUN_ID": "4242",
            "GITHUB_SERVER_URL": "https://github.com",
            "GH_STUB_CALLS": str(calls),
            "GH_STUB_PR_URL": PR_URL,
            "RELEASE_BUMP_RETRY_ATTEMPTS": "1",
            "RELEASE_BUMP_RETRY_DELAY_SECONDS": "0",
            "RELEASE_BUMP_APPROVE_POLL_ATTEMPTS": "3",
            "RELEASE_BUMP_APPROVE_IDLE_SECONDS": "0",
            "RELEASE_BUMP_APPROVE_SEEN_SECONDS": "0",
            "RELEASE_BUMP_RUN_POLL_ATTEMPTS": "2",
            "RELEASE_BUMP_RUN_POLL_SECONDS": "0",
            "RELEASE_BUMP_CONCLUSION_POLL_ATTEMPTS": "2",
            "RELEASE_BUMP_CONCLUSION_POLL_SECONDS": "0",
            "RELEASE_BUMP_MERGE_WAIT_ATTEMPTS": "2",
            "RELEASE_BUMP_MERGE_WAIT_SECONDS": "0",
        }
    )
    return work, bare, env, calls


def run_bump(work: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=work,
    )


def outputs(env: dict[str, str]) -> dict[str, str]:
    path = Path(env["GITHUB_OUTPUT"])
    if not path.is_file():
        return {}
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())


def summary(env: dict[str, str]) -> str:
    path = Path(env["GITHUB_STEP_SUMMARY"])
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def install_reject_main_hook(bare: Path) -> None:
    hook = bare / "hooks" / "pre-receive"
    hook.write_text(REJECT_MAIN_HOOK, encoding="utf-8")
    hook.chmod(0o755)


def test_dry_run_resolves_bump_without_writes(
    release_repo: tuple[Path, Path, dict[str, str], Path],
) -> None:
    work, _bare, env, calls = release_repo
    env["DRY_RUN"] = "true"
    head = _git(work, "rev-parse", "HEAD")

    result = run_bump(work, env)

    assert result.returncode == 0, result.stderr
    out = outputs(env)
    assert out == {"sha": head, "version": "1.2.4", "tag": "v1.2.4"}
    assert "dry-run: v1.2.4 would release" in summary(env)
    # A rehearsal writes nothing: no gh calls, no commit, no file edits.
    assert not calls.exists() or calls.read_text(encoding="utf-8") == ""
    assert _git(work, "rev-parse", "HEAD") == head
    assert _git(work, "status", "--porcelain") == ""
    assert 'version = "1.2.3"' in (work / "pyproject.toml").read_text(encoding="utf-8")


def test_dry_run_set_equal_current_rehearses_consistency_release(
    release_repo: tuple[Path, Path, dict[str, str], Path],
) -> None:
    work, _bare, env, _calls = release_repo
    env["DRY_RUN"] = "true"
    env["SET_VERSION"] = "1.2.3"

    result = run_bump(work, env)

    assert result.returncode == 0, result.stderr
    assert outputs(env)["version"] == "1.2.3"


def test_dry_run_existing_tag_fails(
    release_repo: tuple[Path, Path, dict[str, str], Path],
) -> None:
    work, bare, env, _calls = release_repo
    env["DRY_RUN"] = "true"
    _git(bare, "tag", "v1.2.4", _git(bare, "rev-parse", "main"))

    result = run_bump(work, env)

    assert result.returncode == 1
    assert "::error::tag v1.2.4 already exists" in result.stderr


def test_real_bump_commits_and_pushes_directly(
    release_repo: tuple[Path, Path, dict[str, str], Path],
) -> None:
    work, bare, env, calls = release_repo

    result = run_bump(work, env)

    assert result.returncode == 0, result.stderr
    head = _git(work, "rev-parse", "HEAD")
    assert outputs(env) == {"sha": head, "version": "1.2.4", "tag": "v1.2.4"}
    assert _git(work, "log", "-1", "--format=%s") == ("Release v1.2.4: update version files")
    assert _git(bare, "rev-parse", "main") == head
    assert 'version = "1.2.4"' in (work / "pyproject.toml").read_text(encoding="utf-8")
    assert not calls.exists() or calls.read_text(encoding="utf-8") == ""


def test_set_version_equal_current_skips_commit(
    release_repo: tuple[Path, Path, dict[str, str], Path],
) -> None:
    work, _bare, env, calls = release_repo
    env["SET_VERSION"] = "1.2.3"
    head = _git(work, "rev-parse", "HEAD")

    result = run_bump(work, env)

    assert result.returncode == 0, result.stderr
    assert outputs(env) == {"sha": head, "version": "1.2.3", "tag": "v1.2.3"}
    assert _git(work, "rev-parse", "HEAD") == head
    assert not calls.exists() or calls.read_text(encoding="utf-8") == ""


def test_set_version_greater_commits(
    release_repo: tuple[Path, Path, dict[str, str], Path],
) -> None:
    work, _bare, env, _calls = release_repo
    env["SET_VERSION"] = "2.0.0"

    result = run_bump(work, env)

    assert result.returncode == 0, result.stderr
    assert outputs(env)["version"] == "2.0.0"
    assert _git(work, "log", "-1", "--format=%s") == ("Release v2.0.0: update version files")


def test_rejected_push_falls_back_to_auto_merged_pr(
    release_repo: tuple[Path, Path, dict[str, str], Path],
) -> None:
    work, bare, env, calls = release_repo
    install_reject_main_hook(bare)
    branch = "bot/release-bump-v1.2.4-4242"
    env.update(
        {
            "GH_STUB_MERGE_FETCH": "1",
            "GH_STUB_BARE": str(bare),
            "GH_STUB_BRANCH": branch,
        }
    )

    result = run_bump(work, env)

    assert result.returncode == 0, result.stderr
    head = _git(work, "rev-parse", "HEAD")
    out = outputs(env)
    assert out["version"] == "1.2.4"
    assert out["tag"] == "v1.2.4"
    assert out["sha"] == head
    call_log = calls.read_text(encoding="utf-8")
    assert f"pr create --repo {REPOSITORY} --base main --head {branch}" in call_log
    assert f"workflow run ci.yml --repo {REPOSITORY} --ref {branch}" in call_log
    assert f"workflow run workflow-lint.yml --repo {REPOSITORY} --ref {branch}" in call_log
    assert "--auto --squash --delete-branch" in call_log
    # The stub merge landed the bump commit on origin/main.
    assert _git(bare, "rev-parse", "main") == head
    assert f"version-bump PR: {PR_URL}" in summary(env)


def test_pr_fallback_fails_when_dispatched_checks_fail(
    release_repo: tuple[Path, Path, dict[str, str], Path],
) -> None:
    work, bare, env, calls = release_repo
    install_reject_main_hook(bare)
    env["GH_STUB_CONCLUSION"] = "failure"

    result = run_bump(work, env)

    assert result.returncode == 1
    assert "version-bump PR checks failed" in summary(env)
    call_log = calls.read_text(encoding="utf-8")
    assert "pr create" in call_log
    assert "--auto --squash --delete-branch" not in call_log


def test_release_workflow_delegates_bump_to_script() -> None:
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    assert "bash scripts/release_bump.sh" in workflow
    # The state machine lives in the script; the step stays a thin wrapper.
    assert "gh pr create" not in workflow
    assert "git ls-remote" not in workflow
    # The repo-specific paths the workflow guard used to assert on the
    # inline block now live in the script.
    assert "plugins/fpga/.plugin/plugin.json" in script
    assert "plugins/fpga/skills/*/SKILL.md" in script
    assert "workflow-lint.yml" in script
