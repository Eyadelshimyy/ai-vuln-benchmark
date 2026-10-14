"""Tests for the optional caching layer."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from ellements.core.caching import (
    Cache,
    CacheEntry,
    CachingLLMClient,
    InMemoryCache,
    JsonDiskCache,
    build_cache_key,
)
from pydantic import BaseModel

# ── Protocol structural conformance ──────────────────────────────────


def test_in_memory_cache_satisfies_cache_protocol() -> None:
    assert isinstance(InMemoryCache(), Cache)


def test_json_disk_cache_satisfies_cache_protocol(tmp_path: Path) -> None:
    assert isinstance(JsonDiskCache(tmp_path), Cache)


# ── build_cache_key ──────────────────────────────────────────────────


def test_keys_are_stable_across_runs() -> None:
    k1 = build_cache_key(
        method="complete",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=None,
    )
    k2 = build_cache_key(
        method="complete",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=None,
    )
    assert k1 == k2
    assert len(k1) == 64


def test_keys_differ_for_different_models() -> None:
    base = {
        "method": "complete",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": None,
    }
    k1 = build_cache_key(model="m1", **base)
    k2 = build_cache_key(model="m2", **base)
    assert k1 != k2


def test_keys_ignore_extras_ordering() -> None:
    base = dict(
        method="complete",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=None,
    )
    k1 = build_cache_key(**base, extra={"top_p": 0.9, "stop": ["END"]})
    k2 = build_cache_key(**base, extra={"stop": ["END"], "top_p": 0.9})
    assert k1 == k2


def test_keys_capture_response_model() -> None:
    class _A(BaseModel):
        x: int

    class _B(BaseModel):
        x: int

    base = dict(
        method="complete_structured",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=None,
    )
    assert build_cache_key(**base, response_model=_A) != build_cache_key(
        **base, response_model=_B
    )


# ── InMemoryCache ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_in_memory_cache_round_trip() -> None:
    cache = InMemoryCache()
    await cache.set("k", "value")
    hit = await cache.get("k")
    assert hit is not None
    assert hit.value == "value"
    assert isinstance(hit, CacheEntry)


@pytest.mark.asyncio
async def test_in_memory_cache_miss_returns_none() -> None:
    cache = InMemoryCache()
    assert await cache.get("missing") is None


@pytest.mark.asyncio
async def test_in_memory_cache_delete_then_get() -> None:
    cache = InMemoryCache()
    await cache.set("k", 1)
    await cache.delete("k")
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_in_memory_cache_clear() -> None:
    cache = InMemoryCache()
    await cache.set("k1", 1)
    await cache.set("k2", 2)
    await cache.clear()
    assert await cache.get("k1") is None
    assert await cache.get("k2") is None


@pytest.mark.asyncio
async def test_in_memory_cache_ttl_expiry() -> None:
    cache = InMemoryCache(default_ttl=0.05)
    await cache.set("k", "value")
    assert (await cache.get("k")) is not None
    await asyncio.sleep(0.1)
    assert (await cache.get("k")) is None


@pytest.mark.asyncio
async def test_in_memory_cache_max_entries_evicts_oldest() -> None:
    cache = InMemoryCache(max_entries=2)
    await cache.set("k1", 1)
    await asyncio.sleep(0.01)
    await cache.set("k2", 2)
    await asyncio.sleep(0.01)
    await cache.set("k3", 3)
    assert await cache.get("k1") is None
    assert (await cache.get("k2")) is not None
    assert (await cache.get("k3")) is not None


# ── JsonDiskCache ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_json_disk_cache_round_trip(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)
    await cache.set("k", {"x": 1})
    hit = await cache.get("k")
    assert hit is not None
    assert hit.value == {"x": 1}
    if os.name != "nt":
        assert tmp_path.stat().st_mode & 0o777 == 0o700
        assert (tmp_path / "k.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_json_disk_cache_persists_across_instances(tmp_path: Path) -> None:
    await JsonDiskCache(tmp_path).set("k", [1, 2, 3])
    hit = await JsonDiskCache(tmp_path).get("k")
    assert hit is not None
    assert hit.value == [1, 2, 3]


@pytest.mark.asyncio
async def test_json_disk_cache_delete_removes_file(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)
    await cache.set("k", "x")
    await cache.delete("k")
    assert not (tmp_path / "k.json").exists()


@pytest.mark.asyncio
async def test_json_disk_cache_clear_removes_all(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)
    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.clear()
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.asyncio
async def test_json_disk_cache_ttl_expiry(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path, default_ttl=0.05)
    await cache.set("k", "x")
    await asyncio.sleep(0.1)
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_json_disk_cache_accepts_pydantic_models(tmp_path: Path) -> None:
    class _Verdict(BaseModel):
        score: float
        reasoning: str

    cache = JsonDiskCache(tmp_path)
    await cache.set("k", _Verdict(score=0.9, reasoning="great"))
    hit = await cache.get("k")
    assert hit is not None
    assert hit.value == {"score": 0.9, "reasoning": "great"}


# ── CachingLLMClient ─────────────────────────────────────────────────


class _FakeLLM:
    """Minimal LLMClientProtocol-compatible stub for caching tests."""

    model: str = "fake/model"

    def __init__(self) -> None:
        self.complete_calls = 0
        self.structured_calls = 0

    async def complete(
        self,
        messages: Any,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        self.complete_calls += 1
        return f"answer-{self.complete_calls}"

    async def complete_structured(
        self,
        messages: Any,
        response_model: type,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        self.structured_calls += 1
        return response_model(answer=f"struct-{self.structured_calls}")

    async def complete_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def continue_conversation(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    async def loglikelihood(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def generate_image(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _StructuredAnswer(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_caching_client_memoizes_complete_at_zero_temperature() -> None:
    fake = _FakeLLM()
    client = CachingLLMClient(inner=fake, cache=InMemoryCache())
    a = await client.complete("hello", temperature=0.0)
    b = await client.complete("hello", temperature=0.0)
    assert a == b == "answer-1"
    assert fake.complete_calls == 1


@pytest.mark.asyncio
async def test_caching_client_skips_cache_for_nonzero_temperature_by_default() -> None:
    fake = _FakeLLM()
    client = CachingLLMClient(inner=fake, cache=InMemoryCache())
    await client.complete("hi", temperature=0.5)
    await client.complete("hi", temperature=0.5)
    assert fake.complete_calls == 2


@pytest.mark.asyncio
async def test_caching_client_can_cache_high_temperature_when_opted_in() -> None:
    fake = _FakeLLM()
    client = CachingLLMClient(
        inner=fake,
        cache=InMemoryCache(),
        only_cache_low_temperature=False,
    )
    a = await client.complete("hi", temperature=0.9)
    b = await client.complete("hi", temperature=0.9)
    assert a == b
    assert fake.complete_calls == 1


@pytest.mark.asyncio
async def test_caching_client_memoizes_structured() -> None:
    fake = _FakeLLM()
    client = CachingLLMClient(inner=fake, cache=InMemoryCache())
    a = await client.complete_structured("q", _StructuredAnswer, temperature=0.0)
    b = await client.complete_structured("q", _StructuredAnswer, temperature=0.0)
    assert a == b == _StructuredAnswer(answer="struct-1")
    assert fake.structured_calls == 1


@pytest.mark.asyncio
async def test_caching_client_keys_differ_for_different_prompts() -> None:
    fake = _FakeLLM()
    client = CachingLLMClient(inner=fake, cache=InMemoryCache())
    await client.complete("hello", temperature=0.0)
    await client.complete("goodbye", temperature=0.0)
    assert fake.complete_calls == 2


@pytest.mark.asyncio
async def test_caching_client_proxies_inner_model_attribute() -> None:
    fake = _FakeLLM()
    client = CachingLLMClient(inner=fake, cache=InMemoryCache())
    assert client.model == "fake/model"


@pytest.mark.asyncio
async def test_caching_client_with_disk_cache(tmp_path: Path) -> None:
    fake = _FakeLLM()
    cache = JsonDiskCache(tmp_path)
    client = CachingLLMClient(inner=fake, cache=cache)
    a = await client.complete("hello", temperature=0.0)
    # New client + new in-memory state, but same disk cache:
    client2 = CachingLLMClient(inner=_FakeLLM(), cache=JsonDiskCache(tmp_path))
    b = await client2.complete("hello", temperature=0.0)
    assert a == b
