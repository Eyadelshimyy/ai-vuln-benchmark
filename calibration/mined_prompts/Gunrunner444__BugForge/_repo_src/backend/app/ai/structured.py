"""Robust structured-output parsing for local and cloud models.

Does not assume ``response_format={"type":"json_object"}`` is honoured.
Malformed model text is never treated as verified evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class StructuredParseError(ValueError):
    """Raised when model text cannot be parsed as the expected JSON object."""


def extract_json_object(
    text: str,
    *,
    required_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Parse a JSON object from raw model text.

    Tries strict JSON, then fenced blocks, then the first balanced ``{...}``.
    Validates required keys. Never returns a half-parsed mapping that could
    be mistaken for evidence.
    """
    if not text or not text.strip():
        raise StructuredParseError("empty model output")

    candidates = list(_candidate_blobs(text))
    last_error: Exception | None = None
    for blob in candidates:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(data, dict):
            last_error = StructuredParseError("JSON root is not an object")
            continue
        missing = [key for key in required_keys if key not in data]
        if missing:
            last_error = StructuredParseError(f"missing keys: {missing}")
            continue
        return data

    detail = f" ({last_error})" if last_error else ""
    raise StructuredParseError(f"malformed structured output{detail}")


def _candidate_blobs(text: str) -> list[str]:
    stripped = text.strip()
    blobs: list[str] = [stripped]
    for match in _FENCE.finditer(stripped):
        blobs.append(match.group(1).strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        blobs.append(stripped[start : end + 1])
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for blob in blobs:
        if blob and blob not in seen:
            seen.add(blob)
            unique.append(blob)
    return unique
