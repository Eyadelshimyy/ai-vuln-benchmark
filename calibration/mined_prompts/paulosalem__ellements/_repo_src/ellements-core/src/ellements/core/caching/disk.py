"""JSON-file disk cache backend.

Each cache entry lives at ``<root>/<key>.json``. The file body is
a single JSON object::

    {"value": <whatever>, "created_at": 1700000000.0, "expires_at": null}

This backend is fully durable across process restarts. It is **not**
designed for high write throughput — it does one open/write per
:meth:`set`. For high-volume workloads, prefer
:class:`InMemoryCache` or build a Redis/SQLite backend on top of the
:class:`Cache` Protocol.

Values must be JSON-serializable (Pydantic ``BaseModel`` instances
are accepted because they expose ``.model_dump()``). Round-tripping
preserves only JSON-native types: dicts, lists, strings, numbers,
``True``/``False``/``None``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .cache import CacheEntry


class JsonDiskCache:
    """Durable JSON-on-disk cache satisfying :class:`Cache`.

    Args:
        root: Directory where entries are stored. Created on
            construction if missing.
        default_ttl: Optional default time-to-live (seconds). When a
            caller omits ``ttl=`` on :meth:`set`, this value is used.
            ``None`` (the default) means entries don't expire on
            their own.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        default_ttl: float | None = None,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            self._root.chmod(0o700)
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> CacheEntry | None:
        """Return the cached entry for *key*, or ``None`` if missing,
        unreadable, or expired (in which case the file is removed)."""
        path = self._path_for(key)
        if not path.exists():
            return None
        async with self._lock:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            expires_at = data.get("expires_at")
            if expires_at is not None and expires_at <= time.time():
                path.unlink(missing_ok=True)
                return None
            return CacheEntry(
                value=data.get("value"),
                created_at=float(data.get("created_at", time.time())),
            )

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: float | None = None,
    ) -> None:
        """Store *value* as a JSON file under ``<root>/<key>.json``.

        Uses ``ttl`` when provided; falls back to ``default_ttl``.
        Pydantic ``BaseModel`` values are converted via
        ``.model_dump()`` before serialisation.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()
        expires_at = now + effective_ttl if effective_ttl is not None else None
        payload = {
            "value": _to_json_native(value),
            "created_at": now,
            "expires_at": expires_at,
        }
        async with self._lock:
            path = self._path_for(key)
            temporary = self._root / f".{path.name}.{uuid4().hex}.tmp"
            try:
                temporary.write_text(json.dumps(payload), encoding="utf-8")
                if os.name != "nt":
                    temporary.chmod(0o600)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

    async def delete(self, key: str) -> None:
        """Remove the file for *key* if present; no-op otherwise."""
        async with self._lock:
            self._path_for(key).unlink(missing_ok=True)

    async def clear(self) -> None:
        """Remove every ``*.json`` file under ``root``."""
        async with self._lock:
            for path in self._root.glob("*.json"):
                path.unlink(missing_ok=True)

    def _path_for(self, key: str) -> Path:
        return self._root / f"{key}.json"


def _to_json_native(value: Any) -> Any:
    """Convert *value* into a JSON-native shape.

    Pydantic models are converted via ``.model_dump()``; dataclasses
    that expose ``.__dict__`` are converted to dicts. Everything else
    is passed through and will raise ``TypeError`` at write time if
    it is not JSON-serializable — a deliberate fail-loud choice so
    misuses surface immediately rather than corrupting the cache.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


__all__ = ["JsonDiskCache"]
