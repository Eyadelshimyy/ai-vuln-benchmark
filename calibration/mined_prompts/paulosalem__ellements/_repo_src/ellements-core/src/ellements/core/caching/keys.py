"""Stable cache-key generation for LLM requests.

The single public helper :func:`build_cache_key` produces a short
hex digest that uniquely identifies a request payload modulo
non-semantic noise:

* Mappings are JSON-encoded with ``sort_keys=True``, so dict ordering
  cannot perturb the digest.
* Falsy "no override" fields (``model=None``, ``response_model=None``,
  empty extras) are normalized to a fixed sentinel so different
  call-sites that ultimately produce identical requests collide on
  the same key.
* Pydantic ``BaseModel`` types are represented by their fully
  qualified name, since two distinct classes with identical fields
  should not share cached entries.

Callers should treat the returned key as **opaque**.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def build_cache_key(
    *,
    method: str,
    model: str,
    messages: Any,
    temperature: float,
    max_tokens: int | None,
    extra: Mapping[str, Any] | None = None,
    response_model: type | None = None,
) -> str:
    """Return a stable hex digest for an LLM request.

    Args:
        method: The :class:`ellements.core.LLMClient` method name
            being cached (e.g. ``"complete"``,
            ``"complete_structured"``).
        model: Final resolved model identifier (after any
            per-call override).
        messages: The normalized messages list / Conversation /
            string. Serialized with :func:`_normalize` so unhashable
            container types still hash consistently.
        temperature: Sampling temperature.
        max_tokens: Optional max tokens cap; ``None`` is preserved.
        extra: Extra kwargs (e.g. ``top_p``, ``stop``) the caller
            passed through. Order-independent.
        response_model: Optional Pydantic model class (for
            ``complete_structured``); represented by its fully
            qualified name.

    Returns:
        Hex digest (SHA-256, 64 chars) of the canonicalized payload.
    """
    payload = {
        "method": method,
        "model": model,
        "messages": _normalize(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra": _normalize(dict(extra or {})),
        "response_model": (
            f"{response_model.__module__}.{response_model.__qualname__}"
            if response_model is not None
            else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize(value: Any) -> Any:
    """Render *value* into a JSON-friendly, order-insensitive shape."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if hasattr(value, "model_dump"):
        return _normalize(value.model_dump())
    return repr(value)


def _json_default(value: Any) -> Any:
    """Fallback for ``json.dumps`` when ``_normalize`` returned something exotic."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return repr(value)


__all__ = ["build_cache_key"]
