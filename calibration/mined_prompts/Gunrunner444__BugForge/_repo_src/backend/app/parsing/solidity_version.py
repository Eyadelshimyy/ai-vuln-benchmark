"""Canonical Solidity version semantics.

A property is known only when every version allowed by the pragma has that
property. A range that includes both 0.7 and 0.8 does not claim checked
arithmetic. Comments and strings are not pragmas. Parser context, the
compiler model, and security rules all use this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PRAGMA = re.compile(r"(?m)^[ \t]*pragma\s+solidity\s+([^;]+);")
_ATOM = re.compile(r"(?P<op>\^|~|>=|<=|>|<)?\s*(?P<maj>\d+)\.(?P<min>\d+)(?:\.(?P<pat>\d+))?")
_CHECKED = (0, 8, 0)
_SELFDESTRUCT = (0, 8, 18)


@dataclass(frozen=True)
class _Interval:
    low: tuple[int, int, int]
    high: tuple[int, int, int] | None


@dataclass(frozen=True)
class SolidityLanguageFacts:
    """Known language facts. ``unknown`` stays unknown."""

    arithmetic: str = "unknown"
    selfdestruct: str = "unknown"
    source: str = ""
    floor: str = ""
    constraints: str = ""


def solidity_language_facts(source: str = "", compiler_version: str = "") -> SolidityLanguageFacts:
    """Facts for a source pragma, a bare constraint, or an exact compiler version.

    An exact compiler version is used only when the source has no pragma.
    When both exist and disagree, the property stays unknown.
    """
    clauses = _constraint_clauses(source)
    point = _compiler_point(compiler_version)
    if clauses is None:
        return SolidityLanguageFacts()
    intervals = _intervals(clauses)
    if clauses and intervals is None:
        return SolidityLanguageFacts(constraints=" || ".join(clauses)[:300])
    if not intervals:
        if point is None:
            return SolidityLanguageFacts()
        return _facts([_Interval(point, _next_patch(point))], "compiler", "")
    facts = _facts(intervals, "pragma", " || ".join(clauses)[:300])
    if point is None:
        return facts
    compiled = _facts([_Interval(point, _next_patch(point))], "compiler", facts.constraints)
    if compiled.arithmetic != facts.arithmetic or compiled.selfdestruct != facts.selfdestruct:
        return SolidityLanguageFacts(constraints=facts.constraints)
    return facts


def pragma_allows(source: str, compiler_version: str) -> bool | None:
    """Whether every ``pragma solidity`` in ``source`` admits ``compiler_version``.

    ``None`` means it cannot be decided (no pragma, an unparsable range, or an
    unparsable version); callers must treat that as unknown, never as allowed.
    """
    point = _compiler_point(compiler_version)
    clauses = _constraint_clauses(source)
    if point is None or not clauses:
        return None
    intervals = _intervals(clauses)
    if intervals is None:
        return None
    return any(item.low <= point and (item.high is None or point < item.high) for item in intervals)


def checked_arithmetic(context: tuple[str, ...]) -> bool | None:
    """Read the canonical arithmetic flag. Unknown stays ``None``."""
    for item in context:
        if item == "checked_arithmetic=true":
            return True
        if item == "checked_arithmetic=false":
            return False
    return None


def _facts(intervals: list[_Interval], source: str, constraints: str) -> SolidityLanguageFacts:
    arithmetic = _uniform(intervals, _CHECKED)
    selfdestruct = _uniform(intervals, _SELFDESTRUCT)
    floor = ""
    if len(intervals) == 1:
        low = intervals[0].low
        floor = f"{low[0]}.{low[1]}.{low[2]}"
    arithmetic_label = "unknown"
    if arithmetic is True:
        arithmetic_label = "checked"
    elif arithmetic is False:
        arithmetic_label = "wrapping"
    selfdestruct_label = "unknown"
    if selfdestruct is True:
        selfdestruct_label = "deprecated"
    elif selfdestruct is False:
        selfdestruct_label = "present"
    return SolidityLanguageFacts(
        arithmetic_label,
        selfdestruct_label,
        source,
        floor,
        constraints,
    )


def _uniform(intervals: list[_Interval], boundary: tuple[int, int, int]) -> bool | None:
    """True when every version is >= boundary, False when every version is below it."""
    if not intervals:
        return None
    if all(item.low >= boundary for item in intervals):
        return True
    if all(item.high is not None and item.high <= boundary for item in intervals):
        return False
    return None


def _intervals(clauses: list[str]) -> list[_Interval] | None:
    allowed: list[_Interval] | None = None
    for clause in clauses:
        alternatives: list[_Interval] = []
        for part in clause.split("||"):
            interval = _parse_alternative(part)
            if interval is None:
                return None
            alternatives.append(interval)
        if allowed is None:
            allowed = alternatives
            continue
        merged: list[_Interval] = []
        for left in allowed:
            for right in alternatives:
                hit = _intersect(left, right)
                if hit is not None:
                    merged.append(hit)
        allowed = merged
    return allowed or []


def _parse_alternative(text: str) -> _Interval | None:
    interval = _Interval((0, 0, 0), None)
    found = False
    cursor = 0
    for match in _ATOM.finditer(text):
        if text[cursor : match.start()].strip():
            return None
        piece = _atom_interval(match)
        if piece is None:
            return None
        narrowed = _intersect(interval, piece)
        if narrowed is None:
            return None
        interval = narrowed
        found = True
        cursor = match.end()
    if text[cursor:].strip() or not found:
        return None
    return interval


def _atom_interval(match: re.Match[str]) -> _Interval | None:
    op = match.group("op") or ""
    version = (int(match.group("maj")), int(match.group("min")), int(match.group("pat") or 0))
    if op == "~":
        return None
    if op == "^":
        if version[0] == 0 and version[1] == 0:
            high = _next_patch(version)
        elif version[0] == 0:
            high = (0, version[1] + 1, 0)
        else:
            high = (version[0] + 1, 0, 0)
        return _Interval(version, high)
    if op == ">=":
        return _Interval(version, None)
    if op == ">":
        return _Interval(_next_patch(version), None)
    if op == "<":
        return _Interval((0, 0, 0), version)
    if op == "<=":
        return _Interval((0, 0, 0), _next_patch(version))
    return _Interval(version, _next_patch(version))


def _intersect(left: _Interval, right: _Interval) -> _Interval | None:
    low = max(left.low, right.low)
    if left.high is None:
        high = right.high
    elif right.high is None:
        high = left.high
    else:
        high = min(left.high, right.high)
    if high is not None and low >= high:
        return None
    return _Interval(low, high)


def _next_patch(version: tuple[int, int, int]) -> tuple[int, int, int]:
    return (version[0], version[1], version[2] + 1)


def _compiler_point(text: str) -> tuple[int, int, int] | None:
    if not text or re.search(r"[\^<>=~|]", text):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _constraint_clauses(text: str) -> list[str] | None:
    stripped = _strip(text)
    pragmas = [item.strip() for item in _PRAGMA.findall(stripped)]
    if pragmas:
        return pragmas
    bare = stripped.strip().rstrip(";")
    if not bare:
        return []
    if "pragma" in bare:
        return None
    if re.search(r"\d+\.\d+", bare):
        return [bare]
    return []


def _strip(source: str) -> str:
    stripped = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    stripped = re.sub(r"//.*?$", "", stripped, flags=re.MULTILINE)
    stripped = re.sub(r'"(?:\\.|[^"\\])*"', " ", stripped)
    stripped = re.sub(r"'(?:\\.|[^'\\])*'", " ", stripped)
    return stripped
