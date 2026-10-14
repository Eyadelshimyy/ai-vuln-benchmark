import hashlib
import json
import logging

import redis.asyncio as redis
from redis.exceptions import RedisError

from app.config import settings

log = logging.getLogger(__name__)
_client: redis.Redis | None = None


async def connect() -> None:
    global _client
    kwargs: dict = {"decode_responses": True}
    # Railway Redis uses rediss:// (TLS). Skip cert verification for managed Redis.
    if settings.redis_url.startswith("rediss://"):
        import ssl
        kwargs["ssl_cert_reqs"] = ssl.CERT_NONE
    client = redis.from_url(settings.redis_url, **kwargs)
    try:
        await client.ping()
        _client = client
        log.info("Redis connected: rate limiting and caching active")
    except Exception as exc:
        _client = None
        log.warning(
            "Redis unavailable (%s: %s): caching and rate limiting disabled",
            type(exc).__name__, str(exc)[:120],
        )


async def disconnect() -> None:
    global _client
    if _client:
        await _client.aclose()
        _client = None


def _cache_key(prefix: str, payload: dict) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{prefix}:{digest}"


async def get_cached(prefix: str, payload: dict) -> str | None:
    if not _client:
        return None
    try:
        return await _client.get(_cache_key(prefix, payload))
    except RedisError:
        return None


async def set_cached(prefix: str, payload: dict, value: str) -> None:
    if not _client:
        return
    try:
        await _client.setex(
            _cache_key(prefix, payload),
            settings.cache_ttl_seconds,
            value,
        )
    except RedisError:
        return


async def increment_counter(key: str, window_seconds: int) -> int:
    """Increment a sliding-window counter. Returns current count, or 0 if Redis is unavailable."""
    if not _client:
        log.debug("increment_counter: Redis not connected, returning 0 for key=%s", key)
        return 0
    try:
        count = await _client.incr(key)
        if count == 1:
            await _client.expire(key, window_seconds)
        log.debug("increment_counter: key=%s count=%d", key, count)
        return count
    except RedisError as exc:
        log.warning("increment_counter Redis error: %s", exc)
        return 0
