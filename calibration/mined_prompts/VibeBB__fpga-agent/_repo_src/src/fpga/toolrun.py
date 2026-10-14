"""Run one external tool as a subprocess and keep its transcript."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolRun:
    ok: bool
    argv: list[str]
    detail: str = ""
    returncode: int | None = None
    output: str = ""
    seconds: float = 0.0
    evidence: list[str] = field(default_factory=list[str])


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", "replace")


def run_tool(argv: list[str], cwd: Path, log: Path | None, timeout_s: float) -> ToolRun:
    if shutil.which(argv[0]) is None:
        return ToolRun(False, argv, f"{argv[0]} not found on PATH")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        output, code, detail = proc.stdout, proc.returncode, ""
    except subprocess.TimeoutExpired as exc:
        output, code, detail = _text(exc.output), None, f"timed out after {timeout_s:g}s"
    except OSError as exc:
        return ToolRun(False, argv, f"{argv[0]}: {exc}")
    seconds = time.monotonic() - started
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(argv)}\n{output}\n")
    if code is not None and code != 0:
        detail = f"{Path(argv[0]).name} exited {code}: {_last_error(output)}"
    return ToolRun(code == 0, argv, detail, code, output, seconds)


def _last_error(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        if "error" in lowered or "fatal" in lowered or "failure" in lowered:
            return line[:300]
    return lines[-1][:300] if lines else "no output"
