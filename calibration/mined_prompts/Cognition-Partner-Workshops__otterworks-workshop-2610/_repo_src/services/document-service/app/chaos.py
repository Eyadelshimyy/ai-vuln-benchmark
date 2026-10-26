"""Runtime feature/chaos flags backed by the tenant's Redis.

Flag reads are cached in-process for a short window so request handlers do not
pay a Redis round trip per call, and an unreachable Redis costs at most one
bounded connect attempt per flag per back-off window rather than stalling
every request. Flags read as off whenever Redis cannot be reached.
"""

import os
import time

import redis as redis_lib

_redis_client: redis_lib.Redis | None = None
_flag_cache: dict[str, tuple[float, bool]] = {}

FLAG_SLOW_QUERIES = "chaos:document-service:slow_queries"
FLAG_REQUEST_LOG = "chaos:document-service:request_log"
FLAG_RENDER_CACHE = "chaos:document-service:render_cache"

FLAG_CACHE_SECONDS = 2.0
FLAG_FAILURE_BACKOFF_SECONDS = 15.0
REDIS_TIMEOUT_SECONDS = 0.25


def get_redis() -> redis_lib.Redis:
    """Return a shared Redis client (lazy-initialised)."""
    global _redis_client
    if _redis_client is None:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        _redis_client = redis_lib.Redis(
            host=host,
            port=port,
            decode_responses=True,
            socket_timeout=REDIS_TIMEOUT_SECONDS,
            socket_connect_timeout=REDIS_TIMEOUT_SECONDS,
        )
    return _redis_client


def flag_active(key: str) -> bool:
    """Return True if the given flag is set in Redis (cached; off when Redis is unreachable)."""
    now = time.monotonic()
    cached = _flag_cache.get(key)
    if cached is not None and cached[0] > now:
        return cached[1]
    try:
        active = bool(get_redis().exists(key))
        ttl = FLAG_CACHE_SECONDS
    except Exception:
        active = False
        ttl = FLAG_FAILURE_BACKOFF_SECONDS
    _flag_cache[key] = (now + ttl, active)
    return active
