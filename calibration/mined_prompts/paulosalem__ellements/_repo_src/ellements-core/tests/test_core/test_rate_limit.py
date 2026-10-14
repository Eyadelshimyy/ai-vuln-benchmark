"""Tests for the optional rate-limit layer."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from ellements.core.rate_limit import (
    RateLimitedLLMClient,
    RateLimiterProtocol,
    TokenBucketRateLimiter,
)

# ── Protocol conformance ──────────────────────────────────────────────


def test_token_bucket_satisfies_rate_limiter_protocol() -> None:
    assert isinstance(TokenBucketRateLimiter(1.0), RateLimiterProtocol)


# ── TokenBucketRateLimiter ────────────────────────────────────────────


def test_token_bucket_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError, match="rate_per_second must be positive"):
        TokenBucketRateLimiter(0)


def test_token_bucket_rejects_nonpositive_capacity() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        TokenBucketRateLimiter(1, capacity=0)


@pytest.mark.asyncio
async def test_token_bucket_serves_initial_burst_without_waiting() -> None:
    limiter = TokenBucketRateLimiter(rate_per_second=1.0, capacity=3)
    start = time.monotonic()
    for _ in range(3):
        await limiter.acquire()
    assert time.monotonic() - start < 0.05


@pytest.mark.asyncio
async def test_token_bucket_throttles_once_bucket_drained() -> None:
    limiter = TokenBucketRateLimiter(rate_per_second=10.0, capacity=1)
    await limiter.acquire()  # drains the bucket
    start = time.monotonic()
    await limiter.acquire()  # must wait ~0.1s for one refill
    elapsed = time.monotonic() - start
    assert 0.05 <= elapsed < 0.5


@pytest.mark.asyncio
async def test_token_bucket_acquire_ignores_zero_cost() -> None:
    limiter = TokenBucketRateLimiter(rate_per_second=1.0, capacity=1)
    await limiter.acquire(cost=0)
    await limiter.acquire(cost=0)
    # bucket still full (only zero-cost acquires happened)
    start = time.monotonic()
    await limiter.acquire(cost=1.0)
    assert time.monotonic() - start < 0.05


@pytest.mark.asyncio
async def test_token_bucket_rejects_cost_larger_than_capacity() -> None:
    limiter = TokenBucketRateLimiter(rate_per_second=1.0, capacity=2)
    with pytest.raises(ValueError, match="exceeds capacity"):
        await limiter.acquire(cost=3.0)


@pytest.mark.asyncio
async def test_token_bucket_serializes_concurrent_acquires() -> None:
    limiter = TokenBucketRateLimiter(rate_per_second=20.0, capacity=2)
    start = time.monotonic()
    await asyncio.gather(*(limiter.acquire() for _ in range(6)))
    elapsed = time.monotonic() - start
    # 2 free + 4 throttled @ 20/s = ~0.2s
    assert 0.15 <= elapsed < 0.6


# ── RateLimitedLLMClient ──────────────────────────────────────────────


class _FakeLLM:
    """Minimal LLMClientProtocol stub for rate-limit tests."""

    model: str = "fake/model"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, messages: Any, **kwargs: Any) -> str:
        self.calls.append("complete")
        return "ok"

    async def complete_structured(self, messages: Any, response_model: type, **kwargs: Any) -> Any:
        self.calls.append("complete_structured")
        return response_model()

    async def complete_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("complete_with_tools")

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("stream")
        async def _gen() -> Any:
            yield "x"
        async for chunk in _gen():
            yield chunk

    async def continue_conversation(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append("continue_conversation")
        return "ok"

    async def loglikelihood(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("loglikelihood")
        return (0.0, True)

    async def generate_image(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("generate_image")


class _CountingLimiter:
    """RateLimiterProtocol implementation that records each acquire."""

    def __init__(self) -> None:
        self.acquires: list[float] = []

    async def acquire(self, cost: float = 1.0) -> None:
        self.acquires.append(cost)


@pytest.mark.asyncio
async def test_rate_limited_client_acquires_before_complete() -> None:
    fake = _FakeLLM()
    limiter = _CountingLimiter()
    client = RateLimitedLLMClient(inner=fake, limiter=limiter)
    await client.complete("hi")
    assert limiter.acquires == [1.0]
    assert fake.calls == ["complete"]


@pytest.mark.asyncio
async def test_rate_limited_client_uses_per_method_cost_overrides() -> None:
    fake = _FakeLLM()
    limiter = _CountingLimiter()
    client = RateLimitedLLMClient(
        inner=fake,
        limiter=limiter,
        cost_per_method={"generate_image": 5.0},
    )
    await client.generate_image("a cat")
    await client.complete("hi")
    assert limiter.acquires == [5.0, 1.0]


@pytest.mark.asyncio
async def test_rate_limited_client_uses_custom_default_cost() -> None:
    fake = _FakeLLM()
    limiter = _CountingLimiter()
    client = RateLimitedLLMClient(inner=fake, limiter=limiter, default_cost=0.5)
    await client.complete("hi")
    await client.loglikelihood("ctx", "cont")
    assert limiter.acquires == [0.5, 0.5]


@pytest.mark.asyncio
async def test_rate_limited_client_throttles_under_load() -> None:
    fake = _FakeLLM()
    limiter = TokenBucketRateLimiter(rate_per_second=20.0, capacity=2)
    client = RateLimitedLLMClient(inner=fake, limiter=limiter)
    start = time.monotonic()
    await asyncio.gather(*(client.complete("hi") for _ in range(6)))
    elapsed = time.monotonic() - start
    # 2 free + 4 throttled @ 20/s = ~0.2s
    assert 0.15 <= elapsed < 0.6
    assert fake.calls.count("complete") == 6


@pytest.mark.asyncio
async def test_rate_limited_client_proxies_model() -> None:
    fake = _FakeLLM()
    client = RateLimitedLLMClient(inner=fake, limiter=_CountingLimiter())
    assert client.model == "fake/model"


@pytest.mark.asyncio
async def test_rate_limited_client_stream_yields_chunks() -> None:
    fake = _FakeLLM()
    limiter = _CountingLimiter()
    client = RateLimitedLLMClient(inner=fake, limiter=limiter)
    chunks = [chunk async for chunk in client.stream("hi")]
    assert chunks == ["x"]
    assert limiter.acquires == [1.0]
