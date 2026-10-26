"""Multi-EDM readiness: fact cache, parallel load, SQL row map, admin hooks."""

from __future__ import annotations

import threading
import time
import types

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.providers.connection_registry import ConnectionRegistry, ServerConnectionConfig
from app.providers.fact_cache import FactCache, reset_global_fact_cache
from app.providers.mock import MockExposureDataProvider
from app.providers.parallel_load import load_many
from app.providers.sql_row_map import map_sql_row_to_fact
from app.providers import clear_provider_state
from app.config import get_settings


@pytest.fixture(autouse=True)
def _isolate_cache() -> None:
    clear_provider_state()
    get_settings.cache_clear()
    yield
    clear_provider_state()
    get_settings.cache_clear()


# ───────────────────────── FactCache ─────────────────────────


def test_fact_cache_get_or_load_single_flight() -> None:
    cache: FactCache[list[int]] = FactCache(max_datasets=8, ttl_seconds=0)
    loads = {"n": 0}
    barrier = threading.Barrier(4)

    def loader() -> list[int]:
        loads["n"] += 1
        time.sleep(0.05)
        return [1, 2, 3]

    def worker() -> list[int]:
        barrier.wait()
        return cache.get_or_load("ds-a", loader)

    threads: list[threading.Thread] = []
    results: list[list[int]] = []

    def run() -> None:
        results.append(worker())

    for _ in range(4):
        t = threading.Thread(target=run)
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert loads["n"] == 1
    assert all(r == [1, 2, 3] for r in results)
    assert cache.stats().loads == 1
    assert cache.stats().hits >= 1


def test_fact_cache_lru_eviction() -> None:
    cache: FactCache[list[str]] = FactCache(max_datasets=2, ttl_seconds=0)
    cache.set("a", ["a"])
    cache.set("b", ["b"])
    cache.set("c", ["c"])  # evicts a
    assert cache.get("a") is None
    assert cache.get("b") == ["b"]
    assert cache.get("c") == ["c"]
    assert cache.stats().evictions >= 1


def test_fact_cache_ttl_expiry() -> None:
    cache: FactCache[list[int]] = FactCache(max_datasets=4, ttl_seconds=0.05)
    cache.set("x", [1])
    assert cache.get("x") == [1]
    time.sleep(0.07)
    assert cache.get("x") is None


# ───────────────────────── parallel load ─────────────────────────


def test_load_many_parallel_and_error_isolation() -> None:
    def loader(k: str) -> list[str]:
        if k == "bad":
            raise RuntimeError("boom")
        time.sleep(0.02)
        return [k]

    out = load_many(
        ["a", "b", "bad", "c"],
        loader,
        max_workers=4,
        empty=[],
        on_error="empty",
    )
    assert out["a"] == ["a"]
    assert out["b"] == ["b"]
    assert out["c"] == ["c"]
    assert out["bad"] == []


def test_portfolio_dataset_ids_no_orphans() -> None:
    settings = get_settings()
    cache: FactCache = FactCache(max_datasets=64, ttl_seconds=0)
    provider = MockExposureDataProvider(
        settings.mock_data_dir, fact_cache=cache, max_workers=4
    )
    ids = provider.portfolio_dataset_ids(in_force_only=True)
    assert "ds-farmers-27-ws" not in ids  # orphan per-peril stem
    facts = provider.get_portfolio_facts()
    assert all(f.dataset_id != "ds-farmers-27-ws" for f in facts)


# ───────────────────────── SQL row map ─────────────────────────


def test_map_sql_row_ert_style() -> None:
    row = {
        "Aggregation": "State",
        "Country": "US",
        "CountryName": "United States",
        "Statecode": "FL",
        "StateName": "FLORIDA",
        "Peril": "WS",
        "Building": 100.0,
        "Contents": 50.0,
        "BI": 25.0,
        "TIV": 175.0,
        "#Location": 10,
        "EXPLIM_GR": 160.0,
        "EXPLIM_NET": 140.0,
        "OccupancyGroup": "Res-SFD",
        "Portname": "09302025",
    }
    fact = map_sql_row_to_fact(
        row,
        dataset_id="ds-x",
        server_name="BERMUDA-SQL01",
        database_name="Re_BER_27_X_EDM_01",
        source_table_name="Re_BER_27_X_EDM_01__EVOLUTION",
    )
    assert fact.geography_id == "US-FL"
    assert fact.peril == "WS" or getattr(fact.peril, "value", None) == "WS"
    assert fact.tiv == 175.0
    assert fact.location_count == 10
    seg = fact.occupancy_segment
    assert seg == "RESIDENTIAL" or getattr(seg, "value", None) == "RESIDENTIAL"
    assert fact.dataset_id == "ds-x"


def test_map_sql_row_demo_camel_case() -> None:
    row = {
        "datasetId": "ds-y",
        "portname": "12312025",
        "aggregation": "COUNTY",
        "geographyLevel": "COUNTY",
        "country": "US",
        "statecode": "TX",
        "county": "48201",
        "geographyId": "US-TX-48201",
        "peril": "EQ",
        "building": 1,
        "contents": 2,
        "bi": 3,
        "tiv": 6,
        "locationCount": 1,
        "currency": "USD",
        "occupancySegment": "COMMERCIAL",
    }
    fact = map_sql_row_to_fact(
        row,
        dataset_id="ds-fallback",
        server_name="S",
        database_name="D",
        source_table_name="T",
    )
    assert fact.geography_id == "US-TX-48201"
    assert fact.peril == "EQ" or getattr(fact.peril, "value", None) == "EQ"
    seg = fact.occupancy_segment
    assert seg == "COMMERCIAL" or getattr(seg, "value", None) == "COMMERCIAL"


# ───────────────────────── connection registry ─────────────────────────


def test_connection_registry_odbc_string() -> None:
    reg = ConnectionRegistry()
    reg.register(
        ServerConnectionConfig(
            server_name="BERMUDA-SQL01",
            host="sql01.local",
            port=1433,
            user="u",
            password="p",
        )
    )
    cfg = reg.get("BERMUDA-SQL01")
    assert cfg is not None
    s = cfg.odbc_connect_string("Re_BER_27_Farmers_BDA_EDM_01")
    assert "SERVER=sql01.local,1433" in s
    assert "DATABASE=Re_BER_27_Farmers_BDA_EDM_01" in s
    assert "UID=u" in s


def test_connection_registry_connects_distinct_databases_concurrently(
    monkeypatch,
) -> None:
    reg = ConnectionRegistry()
    reg.register(ServerConnectionConfig(server_name="S", host="sql.local"))
    state = {"active": 0, "max_active": 0}
    state_lock = threading.Lock()
    first_started = threading.Event()

    class FakeCursor:
        def execute(self, _sql):
            return None

        def close(self):
            return None

    class FakeConnection:
        timeout = 0

        def cursor(self):
            return FakeCursor()

        def close(self):
            return None

    def fake_connect(_connection_string, timeout):
        assert timeout == 15
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            first_started.set()
        time.sleep(0.08)
        with state_lock:
            state["active"] -= 1
        return FakeConnection()

    monkeypatch.setitem(
        __import__("sys").modules,
        "pyodbc",
        types.SimpleNamespace(connect=fake_connect),
    )
    threads = [
        threading.Thread(target=reg.connect, args=("S", database))
        for database in ("EDM_A", "EDM_B")
    ]
    threads[0].start()
    assert first_started.wait(timeout=1)
    threads[1].start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert state["max_active"] == 2


# ───────────────────────── mock lazy + parallel ─────────────────────────


def test_mock_lazy_load_and_cache_hits() -> None:
    settings = get_settings()
    cache: FactCache = FactCache(max_datasets=32, ttl_seconds=0)
    provider = MockExposureDataProvider(
        settings.mock_data_dir, fact_cache=cache, max_workers=4
    )
    # Cold
    f1 = provider.get_facts_for_dataset("ds-coastalre-27-ws")
    assert len(f1) > 0
    assert cache.stats().loads == 1
    # Warm
    f2 = provider.get_facts_for_dataset("ds-coastalre-27-ws")
    assert len(f2) == len(f1)
    assert cache.stats().hits >= 1


def test_mock_bulk_load_many_datasets() -> None:
    settings = get_settings()
    cache: FactCache = FactCache(max_datasets=64, ttl_seconds=0)
    provider = MockExposureDataProvider(
        settings.mock_data_dir, fact_cache=cache, max_workers=8
    )
    ids = [
        "ds-coastalre-27-ws",
        "ds-coastalre-26-ws",
        "ds-zenith-27-eq",
        "ds-munich-27-ws",
    ]
    batch = provider.get_facts_for_datasets(ids)
    assert set(batch.by_id.keys()) == set(ids)
    assert all(len(batch.by_id[i]) > 0 for i in ids)
    assert batch.errors == []


def test_warmup_endpoint() -> None:
    client = TestClient(app)
    resp = client.post(
        "/api/admin/cache/warmup",
        json={"inForceOnly": False, "datasetIds": ["ds-coastalre-27-ws", "ds-zenith-27-eq"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["totalDatasets"] == 2
    assert body["totalRows"] > 0
    assert body["loaded"]["ds-coastalre-27-ws"] > 0

    stats = client.get("/api/admin/cache")
    assert stats.status_code == 200
    assert stats.json()["datasetsCached"] >= 1


def test_connections_endpoint_mock_provider() -> None:
    client = TestClient(app)
    resp = client.get("/api/admin/connections")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataProvider"] == "mock"
    assert body["serverCount"] == 0
    assert "factCache" in body


def test_health_includes_cache_fields() -> None:
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "factCacheMaxDatasets" in body
    assert "factLoadMaxWorkers" in body
