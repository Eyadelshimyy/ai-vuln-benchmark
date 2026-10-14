"""Token-bucket :class:`RateLimiterProtocol` implementation."""

from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """Classic token-bucket limiter satisfying :class:`RateLimiterProtocol`.

    The bucket is refilled continuously at ``rate_per_second`` units
    per second, up to a maximum of ``capacity`` units. Each
    :meth:`acquire` call removes ``cost`` units; if the bucket lacks
    enough, the caller sleeps until refill catches up.

    Args:
        rate_per_second: Refill rate. Must be positive. A value of
            ``2`` means up to two single-unit calls per second
            (steady state).
        capacity: Maximum bucket size. Bursts up to this size are
            allowed when the bucket starts full / has been idle.
            Defaults to ``rate_per_second`` (no extra burst budget).

    Raises:
        ValueError: If *rate_per_second* or *capacity* is not
            positive, or if a single :meth:`acquire` call requests a
            *cost* larger than the bucket's *capacity* (such a call
            could never be satisfied).
    """

    def __init__(
        self,
        rate_per_second: float,
        *,
        capacity: float | None = None,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError(
                f"rate_per_second must be positive, got {rate_per_second}."
            )
        if capacity is None:
            capacity = rate_per_second
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self._rate = rate_per_second
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: float = 1.0) -> None:
        """Block until *cost* tokens are available, then deduct them.

        Returns immediately if *cost* is zero or negative. Raises
        :class:`ValueError` if *cost* exceeds the bucket's capacity
        (such a request could never be satisfied).
        """
        if cost <= 0:
            return
        if cost > self._capacity:
            raise ValueError(
                f"acquire(cost={cost}) exceeds capacity={self._capacity}; "
                f"this call could never be satisfied."
            )
        while True:
            async with self._lock:
                self._refill_locked()
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                deficit = cost - self._tokens
                wait_seconds = deficit / self._rate
            await asyncio.sleep(wait_seconds)

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now


__all__ = ["TokenBucketRateLimiter"]
