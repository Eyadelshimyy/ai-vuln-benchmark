"""Cache Protocol and shared value type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A single cache hit.

    Backends return :data:`None` on a miss and an :class:`CacheEntry`
    on a hit so callers never have to disambiguate "cached ``None``"
    from "absent". The dataclass is frozen + slotted to keep cache
    handling allocation-light.

    Attributes:
        value: The cached value. May be any JSON-serializable object
            (or any object the backend can round-trip — e.g. an
            in-memory backend accepts anything picklable / addressable
            by reference).
        created_at: Wall-clock timestamp (Unix seconds) when the entry
            was first written. Useful for callers that want to expose
            cache age in UIs.
    """

    value: Any
    created_at: float


@runtime_checkable
class Cache(Protocol):
    """Async-safe key→value cache.

    Implementations must tolerate concurrent ``get`` / ``set`` calls
    from many tasks. Backends are free to interpret keys as opaque
    strings (the in-package helpers always produce stable, hashed
    keys via :func:`build_cache_key`).

    The TTL on :meth:`set` is **per-entry**: when present it overrides
    any default TTL the backend may have. When ``ttl`` is ``None`` the
    backend's default policy applies (which may be "never expire").
    """

    async def get(self, key: str) -> CacheEntry | None:
        """Return the entry stored at *key*, or ``None`` on a miss."""

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: float | None = None,
    ) -> None:
        """Store *value* under *key*.

        Args:
            key: Opaque key. Treated as a string by every backend.
            value: Object to store. Backends may impose serialization
                constraints (e.g. JSON-only).
            ttl: Optional time-to-live in seconds. When omitted the
                backend's default is used.
        """

    async def delete(self, key: str) -> None:
        """Remove the entry at *key* if it exists. No-op on miss."""

    async def clear(self) -> None:
        """Drop every entry held by the backend."""


__all__ = ["Cache", "CacheEntry"]
