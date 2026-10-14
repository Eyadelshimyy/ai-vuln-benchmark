"""Phase 52 hardening API: stateful status and bundles are readable; control errors map to 409."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

import app.discovery.bounty.service as service_module
from tests.conftest import OPERATOR_HEADERS
from tests.test_phase51.phase51_support import create_body

pytestmark = pytest.mark.asyncio

BASE = "/api/v1/bounty"


@pytest.fixture(autouse=True)
def _fresh_service():
    service_module._SERVICE = None
    yield
    service_module._SERVICE = None


async def test_stateful_status_bundle_and_paused_refusal(client: AsyncClient) -> None:
    resp = await client.post(f"{BASE}/campaigns", json=create_body(), headers=OPERATOR_HEADERS)
    assert resp.status_code == 200, resp.text
    campaign_id = resp.json()["campaign_id"]

    status = await client.get(f"{BASE}/campaigns/{campaign_id}/stateful", headers=OPERATOR_HEADERS)
    assert status.status_code == 200
    body = status.json()
    assert body["verified"] is False and body["executions"] == {} and body["bundles"] == {}

    missing = await client.get(
        f"{BASE}/campaigns/{campaign_id}/repro-bundles/rb_nope", headers=OPERATOR_HEADERS
    )
    assert missing.status_code == 404

    paused = await client.post(
        f"{BASE}/campaigns/{campaign_id}/pause", json={"reason": "t"}, headers=OPERATOR_HEADERS
    )
    assert paused.status_code == 200
    refused = await client.post(
        f"{BASE}/campaigns/{campaign_id}/stateful-execute",
        json={"rounds": 1},
        headers=OPERATOR_HEADERS,
    )
    assert refused.status_code == 409

    unauthenticated = await client.get(f"{BASE}/campaigns/{campaign_id}/stateful")
    assert unauthenticated.status_code in {401, 403}
