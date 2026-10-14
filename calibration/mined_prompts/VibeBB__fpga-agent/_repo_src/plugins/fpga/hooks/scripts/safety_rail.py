#!/usr/bin/env python3
"""Deny terminal commands that are catastrophic or banned by the repo contract.

A deterministic rail, not a security analyzer: it pattern-matches the shell
command and denies a small, unambiguous denylist — destructive filesystem
operations against the root or home tree, device writes, power commands, and
the git operations this repository's working agreement forbids (force push
without --force-with-lease, pushes to main/master, reset --hard, clean -f,
checkout/restore path restores, stash drop/clear, `git add .`, commit
--amend/--no-verify). Everything else passes, including unparseable commands:
the rail only denies what it positively recognizes, it never blocks on
ambiguity.
"""

from __future__ import annotations

import json
import re
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
DEVICE_TARGET = re.compile(r"/dev/(sd|hd|vd|xvd|nvme|mmcblk|disk|loop|mapper)")
ROOT_OR_HOME = {
    "/",
    "/*",
    "~",
    "~/",
    "~/*",
    "$HOME",
    "$HOME/",
    "$HOME/*",
    "${HOME}",
    "${HOME}/",
    "${HOME}/*",
}

BLOCKED_COMMANDS = {
    "fdisk",
    "parted",
    "wipefs",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init",
    "kexec",
}
POWER_TARGETS = {"poweroff", "reboot", "halt", "suspend", "hibernate", "kexec"}


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


def _is_root_or_home(operand: str) -> bool:
    normalized = operand.strip().strip("'\"")
    return normalized in ROOT_OR_HOME


def _denied_rm(args: list[str]) -> bool:
    recursive = force = False
    operands: list[str] = []
    for arg in args:
        if arg.startswith("--"):
            recursive = recursive or arg == "--recursive"
            force = force or arg == "--force"
        elif arg.startswith("-") and len(arg) > 1:
            flags = arg[1:]
            recursive = recursive or "r" in flags or "R" in flags
            force = force or "f" in flags
        else:
            operands.append(arg)
    return recursive and force and any(_is_root_or_home(operand) for operand in operands)


def _git_ref_to_main(token: str) -> bool:
    cleaned = token.lstrip("+")
    if cleaned in ("main", "master", "HEAD:main", "HEAD:master"):
        return True
    return cleaned.startswith("refs/heads/") and cleaned.rsplit("/", 1)[-1] in (
        "main",
        "master",
    )


def _denied_git(args: list[str]) -> str | None:
    if not args:
        return None
    subcommand = args[0]
    rest = args[1:]
    flags = [arg for arg in rest if arg.startswith("-")]
    operands = [arg for arg in rest if not arg.startswith("-")]
    if subcommand == "push":
        if any(
            flag in ("-f", "--force", "--force-all") or flag.startswith("--force=")
            for flag in flags
        ):
            return "git push --force (use --force-with-lease only on your own branch)"
        if any(_git_ref_to_main(operand) for operand in operands):
            return "pushing to main/master is banned by the working agreement"
    if subcommand == "reset" and "--hard" in flags:
        return "git reset --hard is banned by the working agreement"
    if subcommand == "clean" and any(
        "f" in flag.lstrip("-") or flag == "--force" for flag in flags
    ):
        return "git clean -f* is banned by the working agreement"
    if subcommand in ("checkout", "restore") and "--" in rest:
        return "restoring paths with git checkout/restore -- is banned"
    if subcommand == "stash" and rest and rest[0] in ("drop", "clear"):
        return "git stash drop/clear is banned by the working agreement"
    if subcommand == "add" and ("." in operands or any(f in ("-A", "--all") for f in flags)):
        return "git add . / -A/--all is banned; add files individually"
    if subcommand == "commit" and any(flag in ("--amend", "--no-verify", "-n") for flag in flags):
        return "git commit --amend/--no-verify is banned by the working agreement"
    return None


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
    if name in BLOCKED_COMMANDS or name.startswith("mkfs"):
        return f"{name} is on the denylist"
    if name == "systemctl" and args and args[0] in POWER_TARGETS:
        return f"systemctl {args[0]} is on the denylist"
    if name == "rm" and _denied_rm(args):
        return "rm -rf against the root or home tree is denied"
    if name == "dd" and any(arg.startswith("of=") and DEVICE_TARGET.match(arg[3:]) for arg in args):
        return "dd writes to block devices are denied"
    if name in ("tee", "install", "cp", "mv", "shred") and any(
        DEVICE_TARGET.match(operand) for operand in args if not operand.startswith("-")
    ):
        return f"{name} against a block device is denied"
    if name == "kill" and any(operand == "-1" for operand in args):
        return "kill -1 targets every process on the box"
    if name == "git":
        return _denied_git(args)
    return None


def _denied_redirect(tokens: list[str]) -> str | None:
    for i, token in enumerate(tokens):
        if token in (">", ">>", "1>", "1>>", "2>", "2>>", "&>", "&>>") and i + 1 < len(tokens):
            if DEVICE_TARGET.match(tokens[i + 1]):
                return f"writing to a block device ({tokens[i + 1]}) is denied"
        else:
            match = re.match(r"^(\d*|&)>(>?)([^&].*)$", token)
            if match and DEVICE_TARGET.match(match.group(3)):
                return f"writing to a block device ({match.group(3)}) is denied"
    return None


def evaluate(command: str) -> str | None:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    denied = _denied_redirect(tokens)
    if denied is not None:
        return denied
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
        print(f"safety rail: {denied}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
