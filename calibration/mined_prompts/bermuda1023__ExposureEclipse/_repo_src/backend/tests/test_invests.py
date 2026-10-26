"""Invest slots (cyclone numbers 90-99) are not a live-storm product.

The storm list does not probe them, and a bookmarked id is not a bundle.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.invests import is_invest_id
from app.services.live_hurricane import storm_for_impact

client = TestClient(app)


def test_is_invest_id_recognises_90s_cy() -> None:
    assert is_invest_id("AL912026")
    assert is_invest_id("EP992024")
    assert not is_invest_id("AL092024")
    assert not is_invest_id("AL152024")
    assert not is_invest_id("garbage")


def test_storms_list_omits_invests(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api import live as live_api

    monkeypatch.setattr(live_api, "fetch_active_summaries", lambda: [])
    monkeypatch.setattr(live_api, "fetch_active_typhoons", lambda: [])
    body = client.get("/api/live/storms").json()
    assert "invests" not in body
    assert body["active"] == []
    assert body["replay"] == []
    assert body["hasActive"] is False
    assert body["note"]


def test_invest_id_is_not_a_selectable_storm() -> None:
    r = client.get(
        "/api/live/storms/AL912026",
        params={
            "includeObs": "false",
            "includeAlerts": "false",
            "includeLand": "false",
            "includeSst": "false",
        },
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_invest_id_is_not_an_impact_storm() -> None:
    assert storm_for_impact("AL912026") is None
