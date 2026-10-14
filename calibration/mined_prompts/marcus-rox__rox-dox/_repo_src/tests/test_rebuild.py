from __future__ import annotations

import json
import stat
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
import rox_dox.cli as cli
import rox_dox.rebuild as rebuild
from rox_dox.rebuild import (
    GitHubPublisher,
    PullRequest,
    TOKEN_ENV,
    resolve_commit,
    run_rebuild,
)

COMMIT = "a" * 40


class QueuedRunner:
    def __init__(self, results: list[subprocess.CompletedProcess[str]]) -> None:
        self.results = results
        self.commands: list[list[str]] = []

    def __call__(
        self,
        command: list[str],
        **_options: object,
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return self.results.pop(0)


class RootFailureRunner:
    def __init__(self, commit: str) -> None:
        self.commit = commit
        self.commands: list[list[str]] = []

    def __call__(
        self,
        command: list[str],
        **_options: object,
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, f"{self.commit}\n", "")
        if any("rox_core.py" in argument for argument in command):
            return subprocess.CompletedProcess(command, 1, "", "needle not found")
        if any("feature_map.py" in argument for argument in command):
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command}")


class PublisherPushRunner:
    def __init__(self, token: str) -> None:
        self.token = token
        self.command: list[str] = []
        self.environment: dict[str, str] = {}
        self.askpass_path: Path | None = None
        self.askpass_content = ""
        self.askpass_mode: int | None = None

    def __call__(
        self,
        command: list[str],
        **options: object,
    ) -> subprocess.CompletedProcess[str]:
        self.command = command
        environment = options.get("env")
        assert isinstance(environment, Mapping)
        self.environment = dict(environment)

        askpass_path = Path(self.environment["GIT_ASKPASS"])
        assert askpass_path.is_file()
        self.askpass_path = askpass_path
        self.askpass_content = askpass_path.read_text(encoding="utf-8")
        self.askpass_mode = stat.S_IMODE(askpass_path.stat().st_mode)
        assert self.token not in self.askpass_content
        return subprocess.CompletedProcess(command, 0, "", "")


def _fixed_clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def _project_root(path: Path) -> Path:
    (path / "pages" / "authoring").mkdir(parents=True)
    (path / "features").mkdir()
    return path


def test_resolve_commit_uses_explicit_commit_without_fetch(tmp_path: Path) -> None:
    runner = QueuedRunner([subprocess.CompletedProcess([], 0, f"{COMMIT}\n", "")])

    resolved = resolve_commit(
        tmp_path / "rox-core",
        "beef123",
        command_runner=runner,
    )

    assert resolved == COMMIT
    assert len(runner.commands) == 1
    assert runner.commands[0][3:5] == ["rev-parse", "--verify"]
    assert runner.commands[0][-1] == "beef123^{commit}"


def test_resolve_commit_fetches_origin_main_when_commit_is_omitted(
    tmp_path: Path,
) -> None:
    runner = QueuedRunner(
        [
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, f"{COMMIT}\n", ""),
        ]
    )

    resolved = resolve_commit(
        tmp_path / "rox-core",
        None,
        command_runner=runner,
    )

    assert resolved == COMMIT
    assert runner.commands == [
        ["git", "-C", str(tmp_path / "rox-core"), "fetch", "origin", "main"],
        ["git", "-C", str(tmp_path / "rox-core"), "rev-parse", "origin/main"],
    ]


def test_cli_registers_rebuild_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_rebuild_mock = Mock(return_value=0)
    monkeypatch.setattr(cli, "run_rebuild", run_rebuild_mock)
    repo = tmp_path / "rox-core"
    output_dir = tmp_path / "site"

    exit_code = cli.main(
        [
            "rebuild",
            "--repo",
            str(repo),
            "--commit",
            COMMIT,
            "--out",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    arguments = run_rebuild_mock.call_args.kwargs
    assert arguments["repo"] == repo
    assert arguments["requested_commit"] == COMMIT
    assert arguments["output_dir"] == output_dir
    assert arguments["open_pr"] is False
    assert arguments["merge"] is False


def test_cli_accepts_merge_only_with_open_pr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_rebuild_mock = Mock(return_value=0)
    monkeypatch.setattr(cli, "run_rebuild", run_rebuild_mock)

    exit_code = cli.main(
        [
            "rebuild",
            "--repo",
            str(tmp_path / "rox-core"),
            "--open-pr",
            "--merge",
        ]
    )

    assert exit_code == 0
    arguments = run_rebuild_mock.call_args.kwargs
    assert arguments["open_pr"] is True
    assert arguments["merge"] is True


def test_cli_rejects_merge_without_open_pr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "rebuild",
                "--repo",
                str(tmp_path / "rox-core"),
                "--merge",
            ]
        )

    assert error.value.code == 2
    assert "--merge requires --open-pr" in capsys.readouterr().err


def test_rebuild_skips_steps_depending_on_failed_root_page(
    tmp_path: Path,
) -> None:
    project_root = _project_root(tmp_path / "docs")
    output_dir = tmp_path / "site"
    runner = RootFailureRunner(COMMIT)

    exit_code = run_rebuild(
        repo=tmp_path / "rox-core",
        requested_commit=COMMIT,
        output_dir=output_dir,
        open_pr=False,
        merge=False,
        project_root=project_root,
        command_runner=runner,
        clock=_fixed_clock,
        publisher=None,
    )

    report = json.loads((output_dir / "rebuild-report.json").read_text())
    statuses = {step["name"]: step["status"] for step in report["steps"]}
    assert exit_code == 1
    assert statuses["root page (pages/authoring/rox_core.py)"] == "failure"
    assert statuses["feature map (pages/authoring/feature_map.py)"] == "success"
    assert statuses["feature pages (pages/authoring/feature_pages.py)"] == "skipped"
    assert statuses["site build"] == "skipped"
    assert statuses["page commit validation"] == "skipped"
    assert statuses["diagram lint"] == "skipped"
    assert "needle not found" in next(
        step["diagnostic"]
        for step in report["steps"]
        if step["name"] == "root page (pages/authoring/rox_core.py)"
    )
    assert not any(
        "feature_pages.py" in argument
        for command in runner.commands
        for argument in command
    )
    assert not any(
        "rox-dox" in argument for command in runner.commands for argument in command
    )


def test_publisher_push_uses_temporary_askpass_and_isolated_git_config(
    tmp_path: Path,
) -> None:
    token = "test-token-not-in-askpass"
    branch = "devin/123-push-probe"
    runner = PublisherPushRunner(token)
    publisher = GitHubPublisher(
        tmp_path,
        token=token,
        unix_timestamp=123,
        command_runner=runner,
    )

    publisher._push_branch(branch)

    assert runner.command[1:4] == ["-c", "credential.helper=", "push"]
    assert runner.command[-1] == branch
    assert runner.environment["GIT_TERMINAL_PROMPT"] == "0"
    assert runner.environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert runner.environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert runner.environment[TOKEN_ENV] == token
    assert "Username*) echo x-access-token" in runner.askpass_content
    assert "printf '%s\\n' \"$MARCUS_ROX_DOX_GITHUB_TOKEN\"" in (runner.askpass_content)
    assert runner.askpass_mode == 0o700
    assert token not in runner.askpass_content
    assert runner.askpass_path is not None
    assert not runner.askpass_path.exists()


def test_github_publisher_merges_pull_request_through_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token = "merge-test-token"
    request_calls: list[tuple[rebuild.Request, int]] = []

    def fake_urlopen(request: rebuild.Request, *, timeout: int) -> BytesIO:
        request_calls.append((request, timeout))
        return BytesIO(json.dumps({"merged": True, "sha": "b" * 40}).encode("utf-8"))

    monkeypatch.setattr(rebuild, "urlopen", fake_urlopen)
    publisher = GitHubPublisher(
        tmp_path,
        token=token,
        unix_timestamp=123,
        command_runner=Mock(),
    )

    merge_sha = publisher.merge(
        PullRequest(
            number=123,
            url="https://github.com/marcus-rox/rox-dox/pull/123",
        )
    )

    request, timeout = request_calls[0]
    assert merge_sha == "b" * 40
    assert request.full_url == f"{rebuild.PUBLISH_API}/123/merge"
    assert request.method == "PUT"
    assert json.loads(request.data or b"{}") == {"merge_method": "merge"}
    assert request.get_header("Authorization") == f"Bearer {token}"
    assert request.get_header("Accept") == "application/vnd.github+json"
    assert timeout == 30
