"""Structural protocol for rate limiters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RateLimiterProtocol(Protocol):
    """Async-safe permit acquisition.

    A limiter exposes a single coroutine: :meth:`acquire`. Callers
    ``await`` it before performing the rate-limited action and the
    limiter blocks until enough capacity is available.

    Implementations are free to model "capacity" however they want
    (requests-per-second, tokens-per-minute, bytes-per-hour, ...).
    The ``cost`` argument lets callers indicate that a single
    operation consumes multiple units of capacity (e.g. a long
    prompt counts as 50 tokens).
    """

    async def acquire(self, cost: float = 1.0) -> None:
        """Block until *cost* units of capacity become available."""


__all__ = ["RateLimiterProtocol"]
