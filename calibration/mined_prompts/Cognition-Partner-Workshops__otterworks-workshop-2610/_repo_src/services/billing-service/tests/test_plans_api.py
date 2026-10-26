from __future__ import annotations

from datetime import date
from uuid import UUID

import psycopg
from fastapi.testclient import TestClient

import app.main as main
from app.domain import SubscriptionRow

TENANT = UUID("00000000-0000-0000-0000-000000000001")
PLAN = UUID("10000000-0000-0000-0000-000000000002")
CREATED = SubscriptionRow(
    UUID("20000000-0000-0000-0000-000000000003"),
    TENANT,
    PLAN,
    date(2026, 3, 1),
    None,
    "active",
    None,
)


class FakeConnection:
    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_repeated_plan_change_returns_conflict(monkeypatch) -> None:
    calls = 0

    def fake_change_plan(*_args: object) -> tuple[list[SubscriptionRow], SubscriptionRow]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise psycopg.errors.UniqueViolation("duplicate subscription")
        return [CREATED], CREATED

    monkeypatch.setattr(main, "migrate", lambda: None)
    monkeypatch.setattr(main, "connect", FakeConnection)
    monkeypatch.setattr(main, "change_plan", fake_change_plan)
    with TestClient(main.app) as client:
        payload = {"plan_id": str(PLAN), "effective_on": "2026-03-01"}
        first = client.post(f"/api/tenants/{TENANT}/plan-change", json=payload)
        second = client.post(f"/api/tenants/{TENANT}/plan-change", json=payload)

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"] == "this plan change has already been requested"
