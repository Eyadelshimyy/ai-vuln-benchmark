from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import rox_dox.cli as cli
from rox_dox.rebuild import TOKEN_ENV
from rox_dox.setup import (
    ROX_CORE_URL,
    ROX_DOX_URL,
    STEP_NAMES,
    run_setup,
)

TOKEN = "tok-secret-123"


class FakeRunner:
    """Records commands and simulates git/uv effects without network or real git."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.askpass_paths: list[Path] = []
        self.askpass_contents: list[str] = []

    def __call__(
        self,
        command: list[str],
        **options: object,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(command), dict(options)))
        environment = options.get("env")
        if isinstance(environment, Mapping) and "GIT_ASKPASS" in environment:
            askpass_path = Path(str(environment["GIT_ASKPASS"]))
            self.askpass_paths.append(askpass_path)
            self.askpass_contents.append(askpass_path.read_text(encoding="utf-8"))
        joined = " ".join(command)
        if self.fail_on is not None and self.fail_on in joined:
            return subprocess.CompletedProcess(
                command, 1, "", f"fatal: {TOKEN} authentication failed"
            )
        if "get-url" in command:
            path = Path(command[command.index("-C") + 1])
            if (path / ".git").is_dir():
                return subprocess.CompletedProcess(command, 0, "origin\n", "")
            return subprocess.CompletedProcess(
                command, 2, "", "error: No such remote 'origin'"
            )
        if "clone" in command:
            destination = Path(command[-1])
            (destination / ".git").mkdir(parents=True)
            return subprocess.CompletedProcess(command, 0, "", "")
        if "fetch" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ["uv", "sync"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[0] == "bash":
            cwd = Path(str(options["cwd"]))
            jar = cwd / "tools" / "plantuml.jar"
            jar.parent.mkdir(parents=True, exist_ok=True)
            jar.write_text("jar", encoding="utf-8")
            return subprocess.CompletedProcess(
                command, 0, "Downloaded PlantUML to tools/plantuml.jar\n", ""
            )
        raise AssertionError(f"unexpected command: {command}")

    def commands(self) -> list[list[str]]:
        return [command for command, _options in self.calls]

    def options_for(self, needle: str) -> dict[str, object]:
        for command, options in self.calls:
            if needle in " ".join(command):
                return options
        raise AssertionError(f"no recorded command containing {needle!r}")


def test_fresh_workdir_runs_every_step(tmp_path: Path) -> None:
    workdir = tmp_path / "w"
    runner = FakeRunner()

    result = run_setup(workdir, rox_core=None, token=TOKEN, command_runner=runner)

    assert result.error is None
    assert [step.step for step in result.steps] == list(STEP_NAMES)
    assert all(step.action == "done" for step in result.steps)
    commands = runner.commands()
    assert any("clone" in command and ROX_DOX_URL in command for command in commands)
    assert any("clone" in command and ROX_CORE_URL in command for command in commands)
    rox_dox_env = runner.options_for(ROX_DOX_URL)["env"]
    assert isinstance(rox_dox_env, Mapping)
    assert rox_dox_env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert rox_dox_env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert rox_dox_env["GIT_TERMINAL_PROMPT"] == "0"
    assert rox_dox_env[TOKEN_ENV] == TOKEN
    assert "GIT_ASKPASS" in rox_dox_env
    for command, options in runner.calls:
        if ROX_CORE_URL in command or str(workdir / "rox-core") in command:
            env = options.get("env")
            assert env is None or TOKEN_ENV not in env
    assert runner.askpass_paths
    assert all(not path.exists() for path in runner.askpass_paths)


def test_second_run_fetches_and_skips_plantuml(tmp_path: Path) -> None:
    workdir = tmp_path / "w"
    runner = FakeRunner()
    run_setup(workdir, rox_core=None, token=TOKEN, command_runner=runner)
    first_call_count = len(runner.calls)

    result = run_setup(workdir, rox_core=None, token=TOKEN, command_runner=runner)

    assert result.error is None
    second_commands = runner.commands()[first_call_count:]
    assert not any("clone" in command for command in second_commands)
    assert any(
        command[:4]
        == [
            "git",
            "-c",
            "credential.helper=",
            "-C",
        ]
        and str(workdir / "rox-dox") in command
        and command[-2:] == ["fetch", "origin"]
        for command in second_commands
    )
    assert any(
        command[-3:] == ["fetch", "origin", "main"]
        and str(workdir / "rox-core") in command
        for command in second_commands
    )
    assert result.steps[-1].step == "plantuml"
    assert result.steps[-1].action == "skipped"


def test_rox_core_option_reuses_existing_checkout(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    (existing / ".git").mkdir(parents=True)
    workdir = tmp_path / "w"
    runner = FakeRunner()

    result = run_setup(workdir, rox_core=existing, token=TOKEN, command_runner=runner)

    assert result.error is None
    assert result.rox_core == existing
    commands = runner.commands()
    assert not any(
        "clone" in command and ROX_CORE_URL in command for command in commands
    )
    assert any(
        command[-3:] == ["fetch", "origin", "main"] and str(existing) in command
        for command in commands
    )
    assert not (workdir / "rox-core").exists()


def test_token_never_appears_in_commands_or_output(tmp_path: Path) -> None:
    workdir = tmp_path / "w"
    runner = FakeRunner()

    result = run_setup(workdir, rox_core=None, token=TOKEN, command_runner=runner)

    assert result.error is None
    assert TOKEN not in json.dumps(result.to_json())
    for command in runner.commands():
        assert TOKEN not in " ".join(command)
    for content in runner.askpass_contents:
        assert TOKEN not in content

    failing = FakeRunner(fail_on="fetch origin")
    failed = run_setup(workdir, rox_core=None, token=TOKEN, command_runner=failing)
    assert failed.error is not None
    assert failed.error["step"] == "rox-dox"
    assert "[redacted]" in failed.error["message"]
    assert TOKEN not in json.dumps(failed.to_json())
    for command in failing.commands():
        assert TOKEN not in " ".join(command)


def test_cli_setup_requires_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    exit_code = cli.main(["setup", "--workdir", str(tmp_path / "w"), "--json"])

    assert exit_code == 2
    document = json.loads(capsys.readouterr().out)
    assert document["ok"] is False
    assert document["error"]["step"] == "input"
    assert TOKEN_ENV in document["error"]["message"]
