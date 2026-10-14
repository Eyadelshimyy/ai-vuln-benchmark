#!/usr/bin/env python3
"""Reject writes to generated FPGA projections and reports.

Generated files include the constraint files ``*.fpga.pcf``,
``*.fpga.lpf`` and ``*.fpga.cst``, ``*.fpga-pinmap.json``,
``*.fpga-pinmap.md``, ``*.fpga-regmap.json``, the register constants
``*_regs_pkg.vhd`` / ``*_regs.vh``, ``*.fpga-production.json``,
``*.fpga-report.json/.md``, ``sim-*.log`` and
``formal-*.log`` transcripts, ``observations/fpga/*.jsonl``, and
``intake/attachments/manifest.jsonl``. Editing projections by hand breaks
the contract-is-truth invariant; generated evidence records are not edited
directly either.

Only path-bearing arguments decide the verdict: file bodies such as
file_text/new_str may legitimately mention artifact names, so payload
content is never scanned. For the terminal, writes are detected from
shell-level operators (redirects, tee, cp/mv destinations, dd, sed -i, rm,
mkdir, chmod, ...) instead of any mention of a protected path, so
read-only commands like `cat` or `grep` on the artifacts are allowed.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from typing import Any, cast

ARTIFACT_SUFFIXES = (
    ".fpga.pcf",
    ".fpga.lpf",
    ".fpga.cst",
    ".fpga-pinmap.json",
    ".fpga-pinmap.md",
    ".fpga-regmap.json",
    "_regs_pkg.vhd",
    "_regs.vh",
    ".fpga-production.json",
    ".fpga-report.json",
    ".fpga-report.md",
    ".bard-song.json",
)
ARTIFACT_NAMES: tuple[str, ...] = (
    "decisions.jsonl",
    "impressions.jsonl",
    "vision-reviews.jsonl",
    "vision-tool-events.jsonl",
    "image-observations.jsonl",
    "records-status.json",
)
ARTIFACT_PREFIXED = (("sim-", ".log"), ("formal-", ".log"))
WRITE_TOOLS = {"file_editor", "apply_patch"}
VIEW_ACTIONS = {"view", "read", "undo_edit"}
WRITE_ACTIONS = {"create", "str_replace", "insert", "edit", "write"}
PATH_KEYS = ("path", "file_path", "paths", "target_file", "old_path", "new_path")
PATCH_PATH_PREFIXES = (
    "*** Update File:",
    "*** Add File:",
    "*** Delete File:",
    "*** Move to:",
    "+++ b/",
    "--- a/",
)
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
LAST_OPERAND_WRITES = {"cp", "mv", "install", "rsync", "ln", "scp", "cpio"}
ANY_OPERAND_WRITES = {
    "tee",
    "rm",
    "rmdir",
    "touch",
    "mkdir",
    "chmod",
    "chown",
    "chgrp",
    "truncate",
    "shred",
}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in cast(dict[str, Any], value).values() for item in _strings(child)]
    if isinstance(value, list):
        return [item for child in cast(list[Any], value) for item in _strings(child)]
    return []


def _path_values(tool_input: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in PATH_KEYS:
        if key in tool_input:
            values.extend(_strings(tool_input[key]))
    return values


def _patch_paths(tool_input: dict[str, Any]) -> list[str]:
    patch = tool_input.get("patch")
    if not isinstance(patch, str):
        return []
    paths: list[str] = []
    for line in patch.splitlines():
        line = line.strip()
        for prefix in PATCH_PATH_PREFIXES:
            if line.startswith(prefix):
                path = line[len(prefix) :].split("\t", 1)[0].strip().strip("\"'")
                if path and path != "/dev/null":
                    paths.append(path)
                break
    return paths


def _is_protected(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    base = normalized.rsplit("/", 1)[-1]
    if base in ARTIFACT_NAMES:
        return True
    if any(base.startswith(head) and base.endswith(tail) for head, tail in ARTIFACT_PREFIXED):
        return True
    if base.endswith(ARTIFACT_SUFFIXES):
        return True
    parts = normalized.strip("/").split("/")
    if (
        len(parts) >= 3
        and parts[-3:-1] == ["observations", "fpga"]
        and parts[-1].endswith(".jsonl")
    ) or parts[-3:] == ["intake", "attachments", "manifest.jsonl"]:
        return True
    if len(parts) >= 2 and parts[-2] == "liaison" and parts[-1].endswith(".ux-response.json"):
        return True
    return len(parts) >= 2 and parts[-2] == "fpga-reports" and parts[-1].endswith(".png")


def _is_artifact_write(payload: dict[str, Any]) -> bool:
    tool_name = payload.get("tool_name")
    if tool_name not in WRITE_TOOLS:
        return False
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return False
    tool_input = cast(dict[str, Any], tool_input)
    if tool_name == "apply_patch":
        paths = _patch_paths(tool_input) + _path_values(tool_input)
        return any(_is_protected(value) for value in paths)
    if not any(_is_protected(value) for value in _path_values(tool_input)):
        return False
    action = tool_input.get("command") or tool_input.get("action")
    if isinstance(action, str):
        if action in VIEW_ACTIONS:
            return False
        if action in WRITE_ACTIONS:
            return True
    return any(key in tool_input for key in ("file_text", "new_str", "content", "insert_text"))


def _operands(tokens: list[str]) -> list[str]:
    return [token for token in tokens if not token.startswith("-")]


def _command_name(token: str) -> str:
    return token.rsplit("/", 1)[-1]


def _terminal_write_target(command: str) -> str | None:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in COMMAND_SEPARATORS:
            i += 1
            continue
        if token in {">", ">>", "1>", "1>>", "2>", "2>>", "&>", "&>>"}:
            if i + 1 < len(tokens) and _is_protected(tokens[i + 1]):
                return tokens[i + 1]
            i += 2
            continue
        redirect = re.match(r"^(\d*|&)>(>?)([^&].*)?$", token)
        if redirect is not None:
            target = redirect.group(3) or (tokens[i + 1] if i + 1 < len(tokens) else "")
            if _is_protected(target):
                return target
            i += 2 if not redirect.group(3) else 1
            continue
        name = _command_name(token)
        if name in WRAPPER_COMMANDS:
            i += 1
            while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-"):
                i += 1
            continue
        if name == "dd":
            j = i + 1
            while j < len(tokens) and tokens[j] not in COMMAND_SEPARATORS:
                if tokens[j].startswith("of=") and _is_protected(tokens[j][3:]):
                    return tokens[j][3:]
                j += 1
            i = j
            continue
        if name == "sed":
            j = i + 1
            args: list[str] = []
            in_place = False
            while j < len(tokens) and tokens[j] not in COMMAND_SEPARATORS:
                if tokens[j] == "-i" or tokens[j].startswith("-i"):
                    in_place = True
                else:
                    args.append(tokens[j])
                j += 1
            if in_place:
                for operand in _operands(args):
                    if _is_protected(operand):
                        return operand
            i = j
            continue
        if name in LAST_OPERAND_WRITES or name in ANY_OPERAND_WRITES:
            j = i + 1
            args = []
            while j < len(tokens) and tokens[j] not in COMMAND_SEPARATORS:
                args.append(tokens[j])
                j += 1
            operands = _operands(args)
            if name in LAST_OPERAND_WRITES:
                operands = operands[-1:] if operands else []
            for operand in operands:
                if _is_protected(operand):
                    return operand
            i = j
            continue
        i += 1
    return None


def _is_terminal_write(payload: dict[str, Any]) -> str | None:
    if payload.get("tool_name") != "terminal":
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    tool_input = cast(dict[str, Any], tool_input)
    command = tool_input.get("command")
    if not isinstance(command, str):
        return None
    return _terminal_write_target(command)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"invalid hook input: {exc}", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        print("invalid hook input: not an object", file=sys.stderr)
        return 2
    payload = cast(dict[str, Any], payload)
    if _is_artifact_write(payload):
        print(
            "generated FPGA artifacts (*.fpga.pcf/.lpf/.cst, *.fpga-pinmap.*, *.fpga-regmap.json,"
            " *_regs_pkg.vhd, *_regs.vh, *.fpga-production.json, *.fpga-report.*,"
            " sim-*.log, formal-*.log, liaison/*.bard-song.json, observations/fpga/*.jsonl,"
            " intake/attachments/manifest.jsonl) are generated records or projections"
            " and must not be edited directly; regenerate deterministic projections"
            " with `fpga constraints`, `fpga pinmap`, `fpga regmap`,"
            " `fpga production` or `fpga gates`",
            file=sys.stderr,
        )
        return 2
    target = _is_terminal_write(payload)
    if target is not None:
        print(
            f"generated artifacts may not be written through the terminal: {target}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
