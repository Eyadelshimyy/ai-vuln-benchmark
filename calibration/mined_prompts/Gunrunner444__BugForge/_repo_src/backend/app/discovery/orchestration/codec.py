"""Deterministic serialization for orchestration state.

Sets are written sorted, mappings are written with sorted keys, and audit-only
timestamps are excluded from every hash. Loading is strict: an unknown shape
raises instead of being guessed.
"""

from __future__ import annotations

import hashlib
import json
import types
import typing
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

AUDIT_KEYS = frozenset({"created_at", "updated_at", "timestamp"})


class CodecError(ValueError):
    """The stored value does not match the expected shape."""


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): to_jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        return sorted((to_jsonable(item) for item in value), key=_sort_key)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise CodecError(f"cannot serialize {type(value).__name__}")


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def strip_audit(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_audit(item) for key, item in value.items() if key not in AUDIT_KEYS}
    if isinstance(value, list):
        return [strip_audit(item) for item in value]
    return value


def canonical_json(value: Any, *, include_audit: bool = True) -> str:
    data = to_jsonable(value)
    if not include_audit:
        data = strip_audit(data)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any, *, length: int = 16) -> str:
    """Hash of the canonical form without audit fields."""
    blob = canonical_json(value, include_audit=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def from_jsonable(kind: Any, data: Any) -> Any:
    """Rebuild a typed value. The type comes from annotations, never from the data."""
    origin = typing.get_origin(kind)
    args = typing.get_args(kind)
    if kind is Any:
        return data
    if origin in (typing.Union, types.UnionType):
        options = [item for item in args if item is not type(None)]
        if data is None:
            if len(options) != len(args):
                return None
            raise CodecError("null is not allowed here")
        return from_jsonable(options[0], data)
    if isinstance(kind, type) and issubclass(kind, Enum):
        try:
            return kind(data)
        except ValueError as exc:
            raise CodecError(f"unknown {kind.__name__}: {data!r}") from exc
    if isinstance(kind, type) and is_dataclass(kind):
        if not isinstance(data, dict):
            raise CodecError(f"{kind.__name__} needs an object")
        hints = typing.get_type_hints(kind)
        known = {item.name for item in fields(kind)}
        extra = set(data) - known
        if extra:
            raise CodecError(f"{kind.__name__} has unknown fields {sorted(extra)}")
        built = {name: from_jsonable(hints[name], value) for name, value in data.items()}
        return kind(**built)
    if origin is tuple:
        if not isinstance(data, list):
            raise CodecError("a tuple needs a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_jsonable(args[0], item) for item in data)
        if len(args) != len(data):
            raise CodecError("tuple length differs")
        return tuple(from_jsonable(item, value) for item, value in zip(args, data, strict=True))
    if origin in (frozenset, set):
        if not isinstance(data, list):
            raise CodecError("a set needs a list")
        return frozenset(from_jsonable(args[0], item) for item in data)
    if origin is list:
        if not isinstance(data, list):
            raise CodecError("a list needs a list")
        return [from_jsonable(args[0], item) for item in data]
    if origin is dict:
        if not isinstance(data, dict):
            raise CodecError("a mapping needs an object")
        return {str(key): from_jsonable(args[1], value) for key, value in data.items()}
    if kind is bool:
        if not isinstance(data, bool):
            raise CodecError("expected a boolean")
        return data
    if kind is int:
        if isinstance(data, bool) or not isinstance(data, int):
            raise CodecError("expected an integer")
        return data
    if kind is float:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise CodecError("expected a number")
        return float(data)
    if kind is str:
        if not isinstance(data, str):
            raise CodecError("expected a string")
        return data
    raise CodecError(f"unsupported type {kind!r}")
