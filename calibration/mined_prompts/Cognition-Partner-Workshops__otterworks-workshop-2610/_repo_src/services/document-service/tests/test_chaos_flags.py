"""Chaos flag reads are cached and fail closed when Redis is unreachable."""

import pytest
import redis as redis_lib

from app import chaos


class _CountingRedis:
    def __init__(self, active: bool) -> None:
        self.active = active
        self.calls = 0

    def exists(self, key: str) -> int:
        self.calls += 1
        return int(self.active)


class _DownRedis:
    calls = 0

    def exists(self, key: str) -> int:
        self.calls += 1
        raise redis_lib.ConnectionError("connection refused")


@pytest.fixture(autouse=True)
def clear_flag_cache() -> None:
    chaos._flag_cache.clear()


def test_flag_read_is_cached_within_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _CountingRedis(active=True)
    monkeypatch.setattr(chaos, "get_redis", lambda: fake)
    now = 1000.0
    monkeypatch.setattr(chaos.time, "monotonic", lambda: now)

    assert chaos.flag_active("k") is True
    assert chaos.flag_active("k") is True
    assert fake.calls == 1

    now += chaos.FLAG_CACHE_SECONDS + 0.01
    fake.active = False
    assert chaos.flag_active("k") is False
    assert fake.calls == 2


def test_unreachable_redis_reads_off_and_backs_off(monkeypatch: pytest.MonkeyPatch) -> None:
    down = _DownRedis()
    monkeypatch.setattr(chaos, "get_redis", lambda: down)
    now = 1000.0
    monkeypatch.setattr(chaos.time, "monotonic", lambda: now)

    assert chaos.flag_active("k") is False
    for _ in range(50):
        assert chaos.flag_active("k") is False
    assert down.calls == 1

    now += chaos.FLAG_FAILURE_BACKOFF_SECONDS + 0.01
    assert chaos.flag_active("k") is False
    assert down.calls == 2


def test_redis_client_uses_bounded_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chaos, "_redis_client", None)
    client = chaos.get_redis()
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["socket_timeout"] == chaos.REDIS_TIMEOUT_SECONDS
    assert kwargs["socket_connect_timeout"] == chaos.REDIS_TIMEOUT_SECONDS
    assert chaos.REDIS_TIMEOUT_SECONDS < 1
