"""Statement-local Solidity expressions.

Classification is per occurrence inside one statement, not a whole function.
Unknown aliasing stays unknown. This is not a compiler frontend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ASSIGN = re.compile(r"(?<![=!<>])(\+=|-=|\*=|/=|%=|<<=|>>=|\|=|&=|\^=|=)(?!=)")


@dataclass(frozen=True)
class Occurrence:
    path: str
    base: str
    index_text: str
    member_path: str
    kind: str
    start: int
    end: int


def occurrences(statement: str, names: set[str]) -> list[Occurrence]:
    """Read, write, and read-modify-write occurrences in one statement."""
    text = statement.strip()
    if not text or not names:
        return []
    declared = re.match(
        r"(?:u?int\d*|address|bool|string|bytes\d*|mapping\s*\([^;]+?\))\s+([A-Za-z_]\w*)",
        text,
    )
    if declared:
        names = set(names) - {declared.group(1)}
        if not names:
            return []
    delete = re.match(r"delete\s+(.+);?$", text)
    if delete:
        return [_mark(item, "write") for item in _paths(delete.group(1), names, 0)]
    prefix = re.match(r"(\+\+|--)\s*(.+);?$", text)
    suffix = re.match(r"(.+?)\s*(\+\+|--)\s*;?$", text)
    if prefix is not None:
        target = prefix.group(2)
        return [_mark(item, "read-modify-write") for item in _paths(target, names, 0)]
    if suffix is not None and text.rstrip(";").endswith(("++", "--")):
        target = suffix.group(1)
        return [_mark(item, "read-modify-write") for item in _paths(target, names, 0)]
    assign = _top_assign(text)
    if assign is None:
        return [_mark(item, "read") for item in _paths(text, names, 0)]
    op, lhs, rhs, lhs_at, rhs_at = assign
    writes = [_mark(item, "write") for item in _paths(lhs, names, lhs_at)]
    reads = [_mark(item, "read") for item in _paths(rhs, names, rhs_at)]
    if op != "=":
        writes = [_mark(item, "read-modify-write") for item in writes]
        reads = [*reads, *[_mark(item, "read") for item in _paths(lhs, names, lhs_at)]]
    return [*reads, *writes]


def _mark(item: Occurrence, kind: str) -> Occurrence:
    return Occurrence(
        item.path, item.base, item.index_text, item.member_path, kind, item.start, item.end
    )


def _top_assign(text: str) -> tuple[str, str, str, int, int] | None:
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0:
            match = _ASSIGN.match(text, index)
            if match:
                op = match.group(1)
                return (
                    op,
                    text[: match.start()].strip(),
                    text[match.end() :].strip(),
                    0,
                    match.end(),
                )
        index += 1
    return None


def _paths(text: str, names: set[str], offset: int) -> list[Occurrence]:
    found: list[Occurrence] = []
    for match in re.finditer(r"\b([A-Za-z_]\w*)\b", text):
        base = match.group(1)
        if base not in names:
            continue
        cursor = match.end()
        indexes: list[str] = []
        members: list[str] = []
        while cursor < len(text):
            if text[cursor].isspace():
                cursor += 1
                continue
            if text[cursor] == "[":
                end = _close(text, cursor, "[", "]")
                if end < 0:
                    break
                indexes.append(text[cursor:end])
                cursor = end
                continue
            if text[cursor] == ".":
                member = re.match(r"\.\s*([A-Za-z_]\w*)", text[cursor:])
                if member is None:
                    break
                members.append(member.group(1))
                cursor += member.end()
                continue
            break
        index_text = "".join(indexes)
        member_path = ".".join(members)
        path = base + index_text + (("." + member_path) if member_path else "")
        found.append(
            Occurrence(
                path,
                base,
                index_text,
                member_path,
                "read",
                offset + match.start(),
                offset + cursor,
            )
        )
    return found


def _close(text: str, start: int, open_char: str, close_char: str) -> int:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == open_char:
            depth += 1
        elif text[index] == close_char:
            depth -= 1
            if depth == 0:
                return index + 1
    return -1
