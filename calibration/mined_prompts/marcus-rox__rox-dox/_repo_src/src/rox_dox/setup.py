from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from rox_dox.rebuild import ASKPASS_SCRIPT, TOKEN_ENV, CommandRunner

ROX_DOX_URL = "https://github.com/marcus-rox/rox-dox.git"
ROX_CORE_URL = "https://github.com/Rox-AI/rox-core.git"
PLANTUML_JAR = Path("tools/plantuml.jar")
FETCH_PLANTUML = Path("scripts/fetch_plantuml.sh")

STEP_ROX_DOX = "rox-dox"
STEP_UV_SYNC = "uv sync"
STEP_ROX_CORE = "rox-core"
STEP_PLANTUML = "plantuml"
STEP_NAMES = (STEP_ROX_DOX, STEP_UV_SYNC, STEP_ROX_CORE, STEP_PLANTUML)

REDACTED = "[redacted]"


@dataclass(frozen=True)
class SetupStep:
    step: str
    action: Literal["done", "skipped"]
    detail: str


@dataclass(frozen=True)
class SetupResult:
    workdir: Path
    rox_dox: Path
    rox_core: Path | None
    steps: tuple[SetupStep, ...]
    error: dict[str, str] | None

    def to_json(self) -> dict[str, object]:
        return {
            "ok": self.error is None,
            "workdir": str(self.workdir),
            "rox_dox": str(self.rox_dox),
            "rox_core": str(self.rox_core) if self.rox_core is not None else None,
            "steps": [
                {"step": step.step, "action": step.action, "detail": step.detail}
                for step in self.steps
            ],
            "error": self.error,
        }


def _command_output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        output.strip()
        for output in (result.stdout or "", result.stderr or "")
        if output and output.strip()
    )


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    command_runner: CommandRunner,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    options: dict[str, object] = {
        "cwd": cwd,
        "capture_output": True,
        "text": True,
        "check": False,
    }
    if env is not None:
        options["env"] = dict(env)
    return command_runner(list(command), **options)


def _failure(result: subprocess.CompletedProcess[str], token: str) -> str:
    message = (
        _command_output(result) or f"command exited with status {result.returncode}"
    )
    return message.replace(token, REDACTED) if token else message


@contextmanager
def _authenticated_env(token: str) -> Iterator[dict[str, str]]:
    """Yield a git environment that authenticates via ASKPASS with an isolated config."""
    askpass_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="rox-dox-askpass-",
            delete=False,
        ) as askpass:
            askpass.write(ASKPASS_SCRIPT)
            askpass_path = Path(askpass.name)
        askpass_path.chmod(0o700)
        environment = os.environ.copy()
        environment[TOKEN_ENV] = token
        environment.update(
            {
                "GIT_ASKPASS": str(askpass_path),
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
            }
        )
        yield environment
    finally:
        if askpass_path is not None:
            askpass_path.unlink(missing_ok=True)


def _has_origin(path: Path, *, command_runner: CommandRunner) -> bool:
    if not path.is_dir():
        return False
    result = _run(
        ["git", "-C", str(path), "remote", "get-url", "origin"],
        cwd=path,
        command_runner=command_runner,
    )
    return result.returncode == 0


def _fetch_rox_dox(
    rox_dox_dir: Path,
    *,
    token: str,
    command_runner: CommandRunner,
) -> tuple[SetupStep | None, str | None]:
    with _authenticated_env(token) as environment:
        if _has_origin(rox_dox_dir, command_runner=command_runner):
            result = _run(
                [
                    "git",
                    "-c",
                    "credential.helper=",
                    "-C",
                    str(rox_dox_dir),
                    "fetch",
                    "origin",
                ],
                cwd=rox_dox_dir,
                command_runner=command_runner,
                env=environment,
            )
            detail = f"fetched origin in {rox_dox_dir}"
        else:
            result = _run(
                [
                    "git",
                    "-c",
                    "credential.helper=",
                    "clone",
                    ROX_DOX_URL,
                    str(rox_dox_dir),
                ],
                cwd=rox_dox_dir.parent,
                command_runner=command_runner,
                env=environment,
            )
            detail = f"cloned {ROX_DOX_URL} into {rox_dox_dir}"
    if result.returncode:
        return None, _failure(result, token)
    return SetupStep(step=STEP_ROX_DOX, action="done", detail=detail), None


def _uv_sync(
    rox_dox_dir: Path,
    *,
    command_runner: CommandRunner,
) -> tuple[SetupStep | None, str | None]:
    result = _run(
        ["uv", "sync"],
        cwd=rox_dox_dir,
        command_runner=command_runner,
    )
    if result.returncode:
        return None, _failure(result, "")
    return (
        SetupStep(step=STEP_UV_SYNC, action="done", detail=f"uv sync in {rox_dox_dir}"),
        None,
    )


def _fetch_rox_core(
    workdir: Path,
    rox_core: Path | None,
    *,
    command_runner: CommandRunner,
) -> tuple[Path | None, SetupStep | None, str | None]:
    prefix = ""
    checkout: Path | None = None
    detail = ""
    if rox_core is not None:
        if _has_origin(rox_core, command_runner=command_runner):
            checkout = rox_core
        else:
            prefix = f"{rox_core} has no origin remote; "
    default_dir = workdir / "rox-core"
    if checkout is None and _has_origin(default_dir, command_runner=command_runner):
        checkout = default_dir
    if checkout is None:
        result = _run(
            ["git", "clone", ROX_CORE_URL, str(default_dir)],
            cwd=workdir,
            command_runner=command_runner,
        )
        if result.returncode:
            return None, None, _failure(result, "")
        checkout = default_dir
        detail = f"cloned {ROX_CORE_URL} into {checkout}"
    else:
        detail = f"reused {checkout}"
    fetched = _run(
        ["git", "-C", str(checkout), "fetch", "origin", "main"],
        cwd=checkout,
        command_runner=command_runner,
    )
    if fetched.returncode:
        return checkout, None, _failure(fetched, "")
    step = SetupStep(
        step=STEP_ROX_CORE,
        action="done",
        detail=f"{prefix}{detail}; fetched origin main",
    )
    return checkout, step, None


def _plantuml(
    rox_dox_dir: Path,
    *,
    command_runner: CommandRunner,
) -> tuple[SetupStep | None, str | None]:
    if (rox_dox_dir / PLANTUML_JAR).is_file():
        return (
            SetupStep(
                step=STEP_PLANTUML,
                action="skipped",
                detail=f"{PLANTUML_JAR} present",
            ),
            None,
        )
    result = _run(
        ["bash", str(FETCH_PLANTUML)],
        cwd=rox_dox_dir,
        command_runner=command_runner,
    )
    if result.returncode:
        return None, _failure(result, "")
    lines = (result.stdout or "").strip().splitlines()
    detail = lines[-1] if lines else f"bash {FETCH_PLANTUML}"
    return SetupStep(step=STEP_PLANTUML, action="done", detail=detail), None


def _report_step(
    progress: tqdm,
    steps: list[SetupStep],
    name: str,
    step: SetupStep | None,
    failure: str | None,
) -> None:
    if step is not None:
        steps.append(step)
    progress.set_description(name)
    progress.set_postfix_str(step.action if step is not None else "failure")
    progress.update(1)


def _result(
    workdir: Path,
    rox_dox_dir: Path,
    rox_core_dir: Path | None,
    steps: list[SetupStep],
    failed_step: str,
    failure: str | None,
) -> SetupResult:
    error = {"step": failed_step, "message": failure} if failure is not None else None
    return SetupResult(
        workdir=workdir,
        rox_dox=rox_dox_dir,
        rox_core=rox_core_dir,
        steps=tuple(steps),
        error=error,
    )


def run_setup(
    workdir: Path,
    *,
    rox_core: Path | None,
    token: str,
    command_runner: CommandRunner,
) -> SetupResult:
    """Clone or fetch rox-dox and rox-core under workdir, sync deps, fetch PlantUML."""
    workdir.mkdir(parents=True, exist_ok=True)
    rox_dox_dir = workdir / "rox-dox"
    rox_core_dir: Path | None = None
    steps: list[SetupStep] = []
    with tqdm(total=len(STEP_NAMES), desc="setup", unit="step") as progress:
        step, failure = _fetch_rox_dox(
            rox_dox_dir, token=token, command_runner=command_runner
        )
        _report_step(progress, steps, STEP_ROX_DOX, step, failure)
        if failure is not None:
            return _result(workdir, rox_dox_dir, None, steps, STEP_ROX_DOX, failure)

        step, failure = _uv_sync(rox_dox_dir, command_runner=command_runner)
        _report_step(progress, steps, STEP_UV_SYNC, step, failure)
        if failure is not None:
            return _result(workdir, rox_dox_dir, None, steps, STEP_UV_SYNC, failure)

        rox_core_dir, step, failure = _fetch_rox_core(
            workdir, rox_core, command_runner=command_runner
        )
        _report_step(progress, steps, STEP_ROX_CORE, step, failure)
        if failure is not None:
            return _result(
                workdir, rox_dox_dir, rox_core_dir, steps, STEP_ROX_CORE, failure
            )

        step, failure = _plantuml(rox_dox_dir, command_runner=command_runner)
        _report_step(progress, steps, STEP_PLANTUML, step, failure)
        return _result(
            workdir, rox_dox_dir, rox_core_dir, steps, STEP_PLANTUML, failure
        )
