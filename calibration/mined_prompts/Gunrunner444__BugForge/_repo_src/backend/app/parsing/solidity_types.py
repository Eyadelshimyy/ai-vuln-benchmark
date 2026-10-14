"""Canonical Solidity ABI types.

Selectors use the canonical type string. Data locations are not part of a
selector. A type BugForge cannot canonicalize does not get a guessed selector.
"""

from __future__ import annotations

import re

_ELEMENTARY = {
    "address": "address",
    "bool": "bool",
    "string": "string",
    "bytes": "bytes",
    "uint": "uint256",
    "int": "int256",
}
_DYNAMIC = frozenset({"bytes", "string"})


def canonical_solidity_type(annotation: str) -> str:
    """Return the canonical ABI type, or '' when it cannot be established."""
    text = _strip_locations(annotation)
    if not text:
        return ""
    return _piece(text)


def canonical_with_aliases(annotation: str, aliases: dict[str, str]) -> str:
    """Canonicalize ``annotation``, using aliases only for names that are known.

    ``aliases`` values must already be canonical ABI types. An unknown name
    stays unresolved instead of being guessed.
    """
    text = _strip_locations(annotation).replace(" ", "")
    if not text:
        return ""
    return _piece_aliased(text, aliases)


def _piece_aliased(compact: str, aliases: dict[str, str]) -> str:
    array = re.fullmatch(r"(.+)\[(\d*)\]", compact)
    if array and array.group(1):
        base = _piece_aliased(array.group(1), aliases)
        if not base:
            return ""
        return f"{base}[{array.group(2)}]"
    if compact.startswith("("):
        if not compact.endswith(")"):
            return ""
        inner = compact[1:-1]
        parts = _split_top(inner)
        if not parts or any(part == "" for part in parts):
            return "" if inner else "()"
        canon = [_piece_aliased(part, aliases) for part in parts]
        if not all(canon):
            return ""
        return "(" + ",".join(canon) + ")"
    if compact in aliases and aliases[compact]:
        return aliases[compact]
    return _piece(compact)


def type_kind(annotation: str) -> str:
    """elementary, array, fixed_array, mapping, tuple, dynamic, or unresolved."""
    text = _strip_locations(annotation).replace(" ", "")
    if not text:
        return "unresolved"
    if text.startswith("mapping("):
        return "mapping"
    array = re.fullmatch(r"(.+)\[(\d*)\]", text)
    if array and array.group(1):
        return "fixed_array" if array.group(2) else "array"
    if text.startswith("(") and text.endswith(")"):
        return "tuple"
    canon = _piece(text)
    if not canon:
        return "unresolved"
    if canon in _DYNAMIC:
        return "dynamic"
    if canon.endswith("]"):
        return type_kind(canon)
    return "elementary"


def is_dynamic_type(annotation: str) -> bool:
    """ABI dynamic/static classification.

    A tuple or fixed array is dynamic only when a component is dynamic.
    Unresolved names are not treated as dynamic.
    """
    text = _strip_locations(annotation).replace(" ", "")
    if not text:
        return False
    array = re.fullmatch(r"(.+)\[(\d*)\]", text)
    if array and array.group(1):
        if array.group(2) == "":
            return True
        return is_dynamic_type(array.group(1))
    if text.startswith("(") and text.endswith(")"):
        parts = _split_top(text[1:-1])
        if not parts and text != "()":
            return False
        return any(is_dynamic_type(part) for part in parts)
    return type_kind(text) == "dynamic"


def _strip_locations(annotation: str) -> str:
    text = annotation.strip()
    text = re.sub(r"\b(memory|calldata|storage|indexed)\b", "", text)
    text = re.sub(r"\bpayable\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _piece(text: str) -> str:
    compact = text.replace(" ", "")
    if not compact:
        return ""
    array = re.fullmatch(r"(.+)\[(\d*)\]", compact)
    if array and array.group(1):
        base = _piece(array.group(1))
        if not base:
            return ""
        return f"{base}[{array.group(2)}]"
    if compact.startswith("("):
        if not compact.endswith(")"):
            return ""
        inner = compact[1:-1]
        parts = _split_top(inner)
        if not parts or any(part == "" for part in parts):
            return "" if inner else "()"
        canon = [_piece(part) for part in parts]
        if not all(canon):
            return ""
        return "(" + ",".join(canon) + ")"
    if compact.startswith("mapping("):
        return ""
    if compact in _ELEMENTARY:
        return _ELEMENTARY[compact]
    if re.fullmatch(r"u?int\d+", compact) or re.fullmatch(r"bytes\d+", compact):
        return compact
    return ""


def _split_top(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part for part in parts if part != ""]
