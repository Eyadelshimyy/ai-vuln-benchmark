#!/usr/bin/env python3
"""Deny terminal commands that program FPGA hardware."""

from __future__ import annotations

import json
import shlex
import sys
from typing import Any, cast

COMMAND_SEPARATORS = {"|", "||", "&&", "&", ";", "(", ")"}
WRAPPER_COMMANDS = {
    "sudo",
    "doas",
    "env",
    "nice",
    "time",
    "ionice",
    "taskset",
    "stdbuf",
}
PROGRAMMERS = {
    "openFPGALoader",
    "openfpgaloader",
    "iceprog",
    "ecpprog",
    "ecpdap",
    "fujprog",
    "dfu-util",
    "programmer_cli",
    "vivado_lab",
    "quartus_pgm",
}
FPGA_ENTRY_POINTS = {"fpga", "fpga_launcher.py"}


def _command_name(token: str) -> str:
    return token.rsplit("/", 1)[-1]


def _segments(tokens: list[str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in COMMAND_SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def _fpga_program(name: str, args: list[str]) -> bool:
    if name in FPGA_ENTRY_POINTS:
        return bool(args) and args[0] == "program"
    if name.startswith("python") and len(args) >= 3 and args[0] == "-m":
        return args[1] in ("fpga", "fpga.cli") and args[2] == "program"
    if name.startswith("python") and len(args) >= 2:
        return _command_name(args[0]) == "fpga_launcher.py" and args[1] == "program"
    return False


def _denied_segment(segment: list[str]) -> str | None:
    i = 0
    while i < len(segment) and _command_name(segment[i]) in WRAPPER_COMMANDS:
        i += 1
        while i < len(segment) and "=" in segment[i] and not segment[i].startswith("-"):
            i += 1
    if i >= len(segment):
        return None
    name = _command_name(segment[i])
    args = segment[i + 1 :]
    if name in PROGRAMMERS:
        return f"{name} programs hardware; a human runs `fpga program` on the host"
    if _fpga_program(name, args) and "--dry-run" not in args:
        return "fpga program writes to hardware; only a human may run it (use --dry-run)"
    return None


def evaluate(command: str) -> str | None:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    for segment in _segments(tokens):
        denied = _denied_segment(segment)
        if denied is not None:
            return denied
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"invalid hook input: {exc}", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        return 0
    payload = cast(dict[str, Any], payload)
    if payload.get("tool_name") != "terminal":
        return 0
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0
    command = cast(dict[str, Any], tool_input).get("command")
    if not isinstance(command, str):
        return 0
    denied = evaluate(command)
    if denied is not None:
        print(f"programming denied: {denied}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
