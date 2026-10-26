"""Small process-local TTL cache.

Live NHC products change every advisory. ``functools.lru_cache`` kept the
first response (including a failed ``None``) for the life of the process,
so a blip during genesis left models, the cone, and the storm list stuck
until restart. This cache expires, and callers must not store failures.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TtlCache(Generic[K, V]):
    def __init__(self, ttl_s: float, maxsize: int = 32) -> None:
        self.ttl_s = ttl_s
        self.maxsize = maxsize
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()

    def get(self, key: K) -> V | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if time.monotonic() > expires:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: K, value: V) -> None:
        self._data[key] = (time.monotonic() + self.ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()
