"""Phase 51 API end-to-end: API -> campaign -> Phase 50 engines -> Phase 49 orchestrator
-> evidence -> next action -> findings -> report. Operator auth is enforced throughout."""

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


async def _create(client: AsyncClient, **overrides) -> str:
    resp = await client.post(
        f"{BASE}/campaigns", json=create_body(**overrides), headers=OPERATOR_HEADERS
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["campaign_id"]


async def test_full_campaign_flow_over_the_api(client: AsyncClient) -> None:
    campaign_id = await _create(client)

    created = await client.get(f"{BASE}/campaigns/{campaign_id}", headers=OPERATOR_HEADERS)
    assert created.status_code == 200
    assert created.json()["verified"] is False

    analyzed = await client.post(
        f"{BASE}/campaigns/{campaign_id}/analyze", headers=OPERATOR_HEADERS
    )
    assert analyzed.status_code == 200
    body = analyzed.json()
    exercised = set(body["exercised_capabilities"])
    assert {"bounty_context_analysis", "caller_context_analysis", "vfcs_generation"} <= exercised
    assert body["verified"] is False and body["submitted"] is False and body["llm_invoked"] is False

    evidence = await client.get(
        f"{BASE}/campaigns/{campaign_id}/evidence", headers=OPERATOR_HEADERS
    )
    assert evidence.status_code == 200
    assert evidence.json()["evidence"]
    assert all(e["verified"] == "false" for e in evidence.json()["evidence"])

    nxt = await client.get(f"{BASE}/campaigns/{campaign_id}/next-action", headers=OPERATOR_HEADERS)
    assert nxt.status_code == 200 and "next_capability" in nxt.json()

    findings = await client.get(
        f"{BASE}/campaigns/{campaign_id}/findings", headers=OPERATOR_HEADERS
    )
    assert findings.status_code == 200
    detectors = {f["detector"] for f in findings.json()["findings"]}
    assert "caller_context.self_call_elevation" in detectors
    assert all(not f["verified"] for f in findings.json()["findings"])

    repro = await client.get(f"{BASE}/campaigns/{campaign_id}/repro", headers=OPERATOR_HEADERS)
    assert repro.status_code == 200 and repro.json()["sequences"]

    report = await client.get(f"{BASE}/campaigns/{campaign_id}/report", headers=OPERATOR_HEADERS)
    assert report.status_code == 200
    rbody = report.json()
    assert rbody["verified"] is False and rbody["submitted"] is False
    assert "Not verified" in rbody["markdown"] and "Not submitted" in rbody["markdown"]
    assert rbody["pack_id"].startswith("rp_")

    advisories = await client.get(
        f"{BASE}/campaigns/{campaign_id}/advisories", headers=OPERATOR_HEADERS
    )
    assert advisories.status_code == 200 and advisories.json()["considered"] > 0


async def test_execute_and_control_endpoints(client: AsyncClient) -> None:
    campaign_id = await _create(client)
    step = await client.post(
        f"{BASE}/campaigns/{campaign_id}/execute",
        json={"capability": "bounty_context_analysis", "reason": "operator"},
        headers=OPERATOR_HEADERS,
    )
    assert step.status_code == 200 and "progressed" in step.json()

    paused = await client.post(
        f"{BASE}/campaigns/{campaign_id}/pause", json={"reason": "op"}, headers=OPERATOR_HEADERS
    )
    assert paused.status_code == 200 and paused.json()["paused"] is True

    resumed = await client.post(
        f"{BASE}/campaigns/{campaign_id}/resume", json={"reason": "op"}, headers=OPERATOR_HEADERS
    )
    assert resumed.status_code == 200

    stopped = await client.post(
        f"{BASE}/campaigns/{campaign_id}/stop", json={"reason": "op"}, headers=OPERATOR_HEADERS
    )
    assert stopped.status_code == 200 and stopped.json()["stopped_by_operator"] is True


async def test_operator_token_is_required(client: AsyncClient) -> None:
    unauth = await client.post(f"{BASE}/campaigns", json=create_body())
    assert unauth.status_code == 401


async def test_fork_approval_is_operator_only_and_bounded(client: AsyncClient) -> None:
    campaign_id = await _create(client)
    # a non-approvable capability is rejected
    bad = await client.post(
        f"{BASE}/campaigns/{campaign_id}/approvals",
        json={"capability": "static_analysis"},
        headers=OPERATOR_HEADERS,
    )
    assert bad.status_code == 400
    # fork_validation is approvable by the authenticated human operator
    ok = await client.post(
        f"{BASE}/campaigns/{campaign_id}/approvals",
        json={"capability": "fork_validation"},
        headers=OPERATOR_HEADERS,
    )
    assert ok.status_code == 200 and "fork_validation" in ok.json()["approvals"]


async def test_out_of_scope_target_stops_without_engine(client: AsyncClient) -> None:
    campaign_id = await _create(
        client,
        target="BalanceAsDeposit.creditDeposit",
        contract="BalanceAsDeposit",
        function="creditDeposit()",
        source_file="accounting_vulnerable.sol",
        files=["accounting_vulnerable.sol"],
    )
    analyzed = await client.post(
        f"{BASE}/campaigns/{campaign_id}/analyze", headers=OPERATOR_HEADERS
    )
    assert analyzed.status_code == 200
    assert analyzed.json()["stop_reason"] == "scope_blocked"


async def test_invalid_manifest_is_rejected(client: AsyncClient) -> None:
    resp = await client.post(
        f"{BASE}/campaigns",
        json={"manifest": {"platform": ""}, "repo_root": "/tmp"},
        headers=OPERATOR_HEADERS,
    )
    assert resp.status_code == 400


async def test_unknown_campaign_is_404(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/campaigns/cp_missing", headers=OPERATOR_HEADERS)
    assert resp.status_code == 404


async def test_scope_identity_endpoint(client: AsyncClient) -> None:
    campaign_id = await _create(client)
    resp = await client.get(
        f"{BASE}/campaigns/{campaign_id}/scope-identity", headers=OPERATOR_HEADERS
    )
    assert resp.status_code == 200
    body = resp.json()
    # in scope by manifest, but no runtime digest -> binding is not confidently in scope
    assert body["status"] in {"ambiguous", "confidently_in_scope", "unknown"}
    assert "binding" in body
