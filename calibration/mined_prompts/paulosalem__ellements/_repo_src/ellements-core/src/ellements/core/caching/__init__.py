"""Optional caching layer for LLM calls.

This subpackage offers three orthogonal pieces:

1. :class:`Cache` — a tiny async Protocol with ``get`` / ``set`` /
   ``delete`` / ``clear`` semantics. Anything that satisfies it can be
   plugged into the rest of the package.
2. Concrete backends: :class:`InMemoryCache` (process-local dict with
   optional TTL) and :class:`JsonDiskCache` (one JSON file per key in
   a directory; survives process restarts).
3. :class:`CachingLLMClient` — a thin wrapper around any
   :class:`ellements.core.LLMClientProtocol` that memoizes
   :meth:`complete` and :meth:`complete_structured` responses keyed
   off the canonical request payload.

Caching is **opt-in**: nothing in ``ellements.core.LLMClient`` itself
caches. Apps wrap their client when they want repeatable, cheap
responses (smoke tests, demos, deterministic CI). Streams, tool calls,
and image generation are deliberately **not** cached — they have
side-effects or unbounded outputs that don't map cleanly onto a
key→value model.

Example::

    from ellements.core import LLMClient
    from ellements.core.caching import CachingLLMClient, InMemoryCache

    client = CachingLLMClient(
        inner=LLMClient(model="openai/gpt-4o-mini"),
        cache=InMemoryCache(default_ttl=3600),
    )
    answer1 = await client.complete("Hello?")  # hits the network
    answer2 = await client.complete("Hello?")  # served from cache
"""

from __future__ import annotations

from .cache import Cache, CacheEntry
from .client import CachingLLMClient
from .disk import JsonDiskCache
from .keys import build_cache_key
from .memory import InMemoryCache

__all__ = [
    "Cache",
    "CacheEntry",
    "CachingLLMClient",
    "InMemoryCache",
    "JsonDiskCache",
    "build_cache_key",
]
