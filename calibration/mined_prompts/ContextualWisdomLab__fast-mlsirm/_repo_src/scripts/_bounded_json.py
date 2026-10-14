"""Descriptor-safe bounded JSON input for repository automation.

The module opens the requested leaf without following symbolic links where the
platform supports that flag, validates the opened descriptor as a stable regular
file, performs one bounded read, validates structural depth without recursion,
then delegates syntax and value construction to :mod:`json`.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Final

MAX_JSON_BYTES: Final = 32 * 1024 * 1024
MAX_JSON_DEPTH: Final = 128
_READ_CHUNK_BYTES: Final = 64 * 1024
_SAFE_OPEN_ERROR = "JSON input could not be opened as a stable regular file"
_UNSTABLE_PATH_ERROR = "JSON input path changed during the bounded read"
_DUPLICATE_MEMBER_ERROR = "JSON input contains a duplicate JSON object member"
_NONFINITE_NUMBER_ERROR = "JSON input contains a non-finite JSON numeric value"


def _positive_limit(value: object, field_name: str) -> int:
    """Return ``value`` as a positive exact built-in integer."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _open_flags() -> int:
    """Return portable descriptor flags for non-following, nonblocking reads."""
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _descriptor_identity(file_status: os.stat_result) -> tuple[int, int]:
    """Return the device and inode identity for one stat result."""
    return file_status.st_dev, file_status.st_ino


def _validate_path_identity(path: Path, descriptor_status: os.stat_result) -> None:
    """Require ``path`` to still name the regular file held by the descriptor."""
    try:
        path_status = os.lstat(path)
    except OSError:
        raise ValueError(_UNSTABLE_PATH_ERROR) from None
    if not stat.S_ISREG(path_status.st_mode):
        raise ValueError(_SAFE_OPEN_ERROR)
    if _descriptor_identity(path_status) != _descriptor_identity(descriptor_status):
        raise ValueError(_UNSTABLE_PATH_ERROR)


def _read_bounded_descriptor(file_descriptor: int, *, byte_limit: int) -> bytes:
    """Read at most ``byte_limit + 1`` bytes from an opened descriptor."""
    remaining = byte_limit + 1
    chunks: list[bytes] = []
    while remaining:
        chunk = os.read(file_descriptor, min(_READ_CHUNK_BYTES, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    content = b"".join(chunks)
    if len(content) > byte_limit:
        raise ValueError(f"JSON input exceeds maximum allowed size {byte_limit} bytes")
    return content


def _read_stable_regular_file(path: Path, *, byte_limit: int) -> bytes:
    """Open, identify, and bounded-read one stable regular file."""
    try:
        file_descriptor = os.open(path, _open_flags())
    except OSError:
        raise ValueError(_SAFE_OPEN_ERROR) from None
    try:
        descriptor_status = os.fstat(file_descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise ValueError(_SAFE_OPEN_ERROR)
        _validate_path_identity(path, descriptor_status)
        content = _read_bounded_descriptor(file_descriptor, byte_limit=byte_limit)
        _validate_path_identity(path, descriptor_status)
        return content
    finally:
        os.close(file_descriptor)


def _validate_json_depth(content: bytes, *, max_depth: int) -> None:
    """Reject object or array nesting deeper than ``max_depth``."""
    depth = 0
    in_string = False
    escaped = False
    for byte in content:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x5B, 0x7B):
            depth += 1
            if depth > max_depth:
                raise ValueError(
                    f"JSON input exceeds maximum allowed depth {max_depth}"
                )
        elif byte in (0x5D, 0x7D) and depth:
            depth -= 1


def _reject_nonfinite_constant(_: str) -> None:
    """Reject Python's non-standard JSON numeric constants without reflection."""
    raise ValueError(_NONFINITE_NUMBER_ERROR)


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting repeated member names."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(_DUPLICATE_MEMBER_ERROR)
        result[key] = value
    return result


def _loads_interoperable_json(content: str) -> Any:
    """Decode one JSON value using unambiguous RFC-compatible semantics."""
    return json.loads(
        content,
        object_pairs_hook=_reject_duplicate_members,
        parse_constant=_reject_nonfinite_constant,
    )


def read_json_object(
    path: Path,
    *,
    max_bytes: int = MAX_JSON_BYTES,
    max_depth: int = MAX_JSON_DEPTH,
) -> dict[str, Any]:
    """Read a stable, bounded UTF-8 JSON object from ``path``.

    Args:
        path: Leaf file to open without following a symbolic link where
            supported by the operating system.
        max_bytes: Inclusive maximum number of bytes accepted.
        max_depth: Inclusive maximum object/array nesting depth.

    Returns:
        The decoded JSON object.

    Raises:
        ValueError: If limits are invalid, the input is not a stable regular
            file, the byte/depth limit is exceeded, or decoder recursion fails.
        ValueError: Also raised when the input is not valid UTF-8 or JSON.
        RuntimeError: If the decoded root is not an object.
    """
    byte_limit = _positive_limit(max_bytes, "max_bytes")
    depth_limit = _positive_limit(max_depth, "max_depth")
    content = _read_stable_regular_file(path, byte_limit=byte_limit)
    _validate_json_depth(content, max_depth=depth_limit)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("JSON input is not valid UTF-8") from None
    try:
        payload = _loads_interoperable_json(text)
    except json.JSONDecodeError:
        raise ValueError("JSON input is not valid JSON") from None
    except RecursionError as exc:
        raise ValueError("JSON input exceeds decoder recursion capacity") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("JSON artifact must be an object")
    return payload


def parse_json_bounded(
    content: str,
    *,
    max_bytes: int = MAX_JSON_BYTES,
    max_depth: int = MAX_JSON_DEPTH,
) -> Any:
    """Parse a stable, bounded UTF-8 JSON string.

    Args:
        content: Exact built-in string containing the JSON.
        max_bytes: Inclusive maximum number of bytes accepted.
        max_depth: Inclusive maximum object/array nesting depth.

    Raises:
        TypeError: If ``content`` is not an exact built-in string.
        ValueError: If limits are invalid or exceeded, JSON syntax is invalid,
            or decoder recursion fails.
    """
    byte_limit = _positive_limit(max_bytes, "max_bytes")
    depth_limit = _positive_limit(max_depth, "max_depth")
    if type(content) is not str:
        raise TypeError("content must be str")

    encoded = content.encode("utf-8")
    if len(encoded) > byte_limit:
        raise ValueError(f"JSON input exceeds maximum allowed size {byte_limit} bytes")

    _validate_json_depth(encoded, max_depth=depth_limit)

    try:
        return _loads_interoperable_json(content)
    except json.JSONDecodeError:
        raise ValueError("JSON input is not valid JSON") from None
    except RecursionError as exc:
        raise ValueError("JSON input exceeds decoder recursion capacity") from exc


__all__ = ["MAX_JSON_BYTES", "MAX_JSON_DEPTH", "read_json_object", "parse_json_bounded"]
