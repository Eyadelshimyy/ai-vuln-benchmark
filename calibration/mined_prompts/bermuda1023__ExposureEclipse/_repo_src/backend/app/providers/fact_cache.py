"""Process-wide fact cache for multi-EDM workloads.

Hundreds of pre-aggregated EDMs still mean a lot of rows. This cache:

* Holds normalized fact lists keyed by ``dataset_id`` (one EDM cut).
* Caps memory with an LRU eviction policy (``max_datasets``).
* Optional TTL so stale SQL pulls refresh after ERT re-runs.
* Single-flight loading: concurrent requests for the same dataset share one load.

The cache is process-local (fine for a long-running uvicorn demo; on
serverless each cold start starts empty — warmup endpoint helps).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    value: T
    loaded_at: float
    row_count: int


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    loads: int = 0
    load_errors: int = 0
    evictions: int = 0
    datasets_cached: int = 0
    rows_cached: int = 0
    max_datasets: int = 0
    ttl_seconds: float = 0.0
    in_flight: int = 0


class FactCache(Generic[T]):
    """Thread-safe LRU + TTL cache with single-flight ``get_or_load``."""

    def __init__(self, *, max_datasets: int = 1024, ttl_seconds: float = 3600.0) -> None:
        self._max = max(1, max_datasets)
        self._ttl = max(0.0, ttl_seconds)
        self._lock = threading.RLock()
        self._data: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._in_flight: dict[str, threading.Event] = {}
        self._in_flight_errors: dict[str, BaseException] = {}
        self._hits = 0
        self._misses = 0
        self._loads = 0
        self._load_errors = 0
        self._evictions = 0

    def _is_fresh(self, entry: _Entry[T]) -> bool:
        if self._ttl <= 0:
            return True
        return (time.monotonic() - entry.loaded_at) <= self._ttl

    def get(self, key: str) -> T | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if not self._is_fresh(entry):
                del self._data[key]
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T, *, row_count: int | None = None) -> None:
        with self._lock:
            if key in self._data:
                del self._data[key]
            n = row_count if row_count is not None else (
                len(value) if hasattr(value, "__len__") else 0  # type: ignore[arg-type]
            )
            self._data[key] = _Entry(value=value, loaded_at=time.monotonic(), row_count=n)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)
                self._evictions += 1

    def invalidate(self, key: str | None = None) -> int:
        """Drop one key, or all keys when ``key`` is None. Returns count dropped."""
        with self._lock:
            if key is None:
                n = len(self._data)
                self._data.clear()
                return n
            if key in self._data:
                del self._data[key]
                return 1
            return 0

    def get_or_load(self, key: str, loader: Callable[[], T]) -> T:
        """Return cached value or run ``loader`` once (single-flight per key)."""
        cached = self.get(key)
        if cached is not None:
            return cached

        leader = False
        event: threading.Event
        with self._lock:
            entry = self._data.get(key)
            if entry is not None and self._is_fresh(entry):
                self._hits += 1
                self._data.move_to_end(key)
                return entry.value

            if key in self._in_flight:
                event = self._in_flight[key]
            else:
                event = threading.Event()
                self._in_flight[key] = event
                self._in_flight_errors.pop(key, None)
                leader = True
                self._misses += 1

        if not leader:
            if not event.wait(timeout=300.0):
                raise TimeoutError(f"Timed out waiting for in-flight fact load: {key}")
            cached = self.get(key)
            if cached is not None:
                return cached
            with self._lock:
                err = self._in_flight_errors.pop(key, None)
            if err is not None:
                raise err
            # The leader exited without publishing a value or error. Retry once
            # through the normal election path; do not start a competing load.
            return self.get_or_load(key, loader)

        try:
            value = loader()
            n = len(value) if hasattr(value, "__len__") else 0  # type: ignore[arg-type]
            self.set(key, value, row_count=n)
            with self._lock:
                self._loads += 1
                self._in_flight_errors.pop(key, None)
            return value
        except BaseException as exc:
            with self._lock:
                self._load_errors += 1
                self._in_flight_errors[key] = exc
            raise
        finally:
            with self._lock:
                self._in_flight.pop(key, None)
            event.set()

    def stats(self) -> CacheStats:
        with self._lock:
            rows = sum(e.row_count for e in self._data.values())
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                loads=self._loads,
                load_errors=self._load_errors,
                evictions=self._evictions,
                datasets_cached=len(self._data),
                rows_cached=rows,
                max_datasets=self._max,
                ttl_seconds=self._ttl,
                in_flight=len(self._in_flight),
            )

    def cached_keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())


# Process-wide default used by providers (one cache across mock/sql in hybrid).
_GLOBAL_FACT_CACHE: FactCache | None = None
_GLOBAL_LOCK = threading.Lock()


def get_global_fact_cache(
    *, max_datasets: int = 256, ttl_seconds: float = 3600.0
) -> FactCache:
    """Return the process-wide fact cache, creating it on first use.

    Note: first caller wins for max/ttl sizing. Configure via Settings before
    any provider constructs.
    """
    global _GLOBAL_FACT_CACHE
    with _GLOBAL_LOCK:
        if _GLOBAL_FACT_CACHE is None:
            _GLOBAL_FACT_CACHE = FactCache(
                max_datasets=max_datasets, ttl_seconds=ttl_seconds
            )
        return _GLOBAL_FACT_CACHE


def reset_global_fact_cache() -> None:
    """Test helper — drop the process-wide cache instance."""
    global _GLOBAL_FACT_CACHE
    with _GLOBAL_LOCK:
        _GLOBAL_FACT_CACHE = None
