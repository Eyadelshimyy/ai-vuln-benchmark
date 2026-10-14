"""Process-local in-memory cache backend.

Backed by a plain ``dict`` plus a single :class:`asyncio.Lock` for
mutation safety. Supports optional TTL with lazy expiry on read.

Suitable for tests, demos, and short-lived processes; if you need
durability across runs, use :class:`ellements.core.caching.JsonDiskCache`
or build your own backend on top of the :class:`Cache` Protocol.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .cache import CacheEntry


@dataclass(slots=True)
class _Record:
    entry: CacheEntry
    expires_at: float | None


class InMemoryCache:
    """Thread/task-safe in-memory cache satisfying :class:`Cache`.

    Args:
        default_ttl: Default time-to-live (seconds) applied when
            callers omit ``ttl=`` on :meth:`set`. ``None`` (the
            default) means entries never expire on their own.
        max_entries: Optional cap on the number of entries. When the
            cap is exceeded, the **oldest** entry (by ``created_at``)
            is evicted on each new write. ``None`` disables eviction.
    """

    def __init__(
        self,
        *,
        default_ttl: float | None = None,
        max_entries: int | None = None,
    ) -> None:
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._store: dict[str, _Record] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> CacheEntry | None:
        """Return the cached entry for *key*, or ``None`` if it's missing
        or has expired (in which case the entry is also evicted)."""
        async with self._lock:
            record = self._store.get(key)
            if record is None:
                return None
            if record.expires_at is not None and record.expires_at <= time.time():
                del self._store[key]
                return None
            return record.entry

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: float | None = None,
    ) -> None:
        """Store *value* under *key*.

        Uses ``ttl`` when provided; falls back to ``default_ttl`` from
        construction. Triggers eviction of the oldest entry when
        ``max_entries`` is set and the cap is exceeded.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()
        expires_at = now + effective_ttl if effective_ttl is not None else None
        entry = CacheEntry(value=value, created_at=now)
        async with self._lock:
            self._store[key] = _Record(entry=entry, expires_at=expires_at)
            self._enforce_capacity()

    async def delete(self, key: str) -> None:
        """Remove *key* if present; no-op otherwise."""
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        """Drop every entry from the cache."""
        async with self._lock:
            self._store.clear()

    def _enforce_capacity(self) -> None:
        if self._max_entries is None:
            return
        overflow = len(self._store) - self._max_entries
        if overflow <= 0:
            return
        ordered = sorted(self._store.items(), key=lambda kv: kv[1].entry.created_at)
        for key, _ in ordered[:overflow]:
            del self._store[key]


__all__ = ["InMemoryCache"]
