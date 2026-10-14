"""Optional client-side rate limiting for LLM calls.

Two pieces:

1. :class:`RateLimiterProtocol` — a tiny async Protocol with a single
   :meth:`acquire` method. Anything implementing it is a valid
   limiter.
2. :class:`TokenBucketRateLimiter` — a classic token-bucket
   implementation: refills at a configurable rate, accepts bursts up
   to a configurable capacity. Bounded contention via a single
   :class:`asyncio.Lock`.
3. :class:`RateLimitedLLMClient` — a wrapper around any
   :class:`ellements.core.LLMClientProtocol` that calls
   ``await limiter.acquire(...)`` before each request.

The wrapper itself satisfies :class:`LLMClientProtocol` so it slots
into any consumer structurally.

Example::

    from ellements.core import LLMClient
    from ellements.core.rate_limit import (
        RateLimitedLLMClient,
        TokenBucketRateLimiter,
    )

    limiter = TokenBucketRateLimiter(rate_per_second=2.0, capacity=4)
    client = RateLimitedLLMClient(
        inner=LLMClient(model="openai/gpt-4o-mini"),
        limiter=limiter,
    )
"""

from __future__ import annotations

from .bucket import TokenBucketRateLimiter
from .client import RateLimitedLLMClient
from .protocol import RateLimiterProtocol

__all__ = [
    "RateLimitedLLMClient",
    "RateLimiterProtocol",
    "TokenBucketRateLimiter",
]
