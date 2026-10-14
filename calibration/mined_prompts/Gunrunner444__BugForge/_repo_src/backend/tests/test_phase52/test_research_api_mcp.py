"""Phase 52 Slice E: research coverage, gaps, properties, observations, feedback
status, bundle metadata, engine availability and the evidence graph are exposed via
the API and the MCP server -- as reads. Nothing exposed here can change scope,
auth, budgets, approvals, deployment, policy or verification."""

from __future__ import annotations

import copy

import httpx
import pytest
from httpx import AsyncClient
from starlette.testclient import TestClient

import app.discovery.bounty.service as service_module
from app.cursor_control.mcp_server import (
    TOOL_DEFINITIONS,
    BugForgeApi,
    LocalApiError,
    dispatch_tool,
)
from app.main import app
from tests.conftest import OPERATOR_HEADERS
from tests.test_phase50.phase50_support import MANIFEST_DATA
from tests.test_phase51.phase51_support import create_body
from tests.test_phase52.phase52_support import requires_forge

BASE = "/api/v1/bounty"
RESEARCH = ("coverage", "plan", "engines", "properties", "evidence-graph", "modules")
READ_TOOLS = (
    "bugforge_campaign_stateful_status",
    "bugforge_campaign_research_coverage",
    "bugforge_campaign_research_plan",
    "bugforge_campaign_engines",
    "bugforge_campaign_properties",
    "bugforge_campaign_evidence_graph",
    "bugforge_campaign_modules",
)


@pytest.fixture(autouse=True)
def _fresh_service():
    service_module._SERVICE = None
    yield
    service_module._SERVICE = None


def _aa_body() -> dict[str, object]:
    manifest = copy.deepcopy(MANIFEST_DATA)
    manifest["in_scope"] = [{"kind": "contract", "identifier": "LooseAccount"}]
    manifest["out_of_scope"] = []
    manifest["known_issues"] = []
    return create_body(
        manifest=manifest,
        target="LooseAccount.initialize",
        contract="LooseAccount",
        function="initialize(address)",
        source_file="aa_vulnerable.sol",
        files=["aa_vulnerable.sol"],
    )


@pytest.mark.asyncio
async def test_research_endpoints_are_authenticated_reads(client: AsyncClient) -> None:
    resp = await client.post(f"{BASE}/campaigns", json=_aa_body(), headers=OPERATOR_HEADERS)
    assert resp.status_code == 200, resp.text
    cid = resp.json()["campaign_id"]
    before = (await client.get(f"{BASE}/campaigns/{cid}/state", headers=OPERATOR_HEADERS)).json()
    bodies = {}
    for name in RESEARCH:
        url = f"{BASE}/campaigns/{cid}/research/{name}"
        ok = await client.get(url, headers=OPERATOR_HEADERS)
        assert ok.status_code == 200, (name, ok.text)
        bodies[name] = ok.json()
        assert bodies[name]["campaign_id"] == cid
        assert bodies[name].get("verified", False) is False
        assert (await client.get(url)).status_code in {401, 403}
        assert (await client.post(url, json={}, headers=OPERATOR_HEADERS)).status_code == 405
    unknown = await client.get(
        f"{BASE}/campaigns/cp_nope/research/coverage", headers=OPERATOR_HEADERS
    )
    assert unknown.status_code == 404

    assert {"RESEARCH_COVERAGE", "RESEARCH_GAPS", "ledger_digest"} <= set(bodies["coverage"])
    assert bodies["coverage"]["RESEARCH_GAPS"]  # nothing executed yet: gaps are visible
    assert {"last_run", "current"} <= set(bodies["engines"])
    current = {e["name"]: e["status"] for e in bodies["engines"]["current"]}
    assert current["fork_replay"] == "blocked_by_policy"
    assert "usable" not in {current.get("echidna"), current.get("medusa")}  # GET runs no smoke
    props = bodies["properties"]["properties"]
    assert props and all(p["observation"] is None and p["verified"] is False for p in props)
    assert {n["kind"] for n in bodies["evidence-graph"]["nodes"]} >= {"static_candidate"}
    assert {m["module"] for m in bodies["modules"]["modules"]} >= {"erc4337", "bridge"}
    assert next(m for m in bodies["modules"]["modules"] if m["module"] == "erc4337")["triggered"]

    after = (await client.get(f"{BASE}/campaigns/{cid}/state", headers=OPERATOR_HEADERS)).json()
    assert after == before  # reading research state changes nothing


def _bridged_api() -> BugForgeApi:
    test_client = TestClient(app)

    def handler(request: httpx.Request) -> httpx.Response:
        resp = test_client.request(
            request.method,
            request.url.raw_path.decode(),
            content=request.content,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        )
        return httpx.Response(resp.status_code, content=resp.content, headers=dict(resp.headers))

    return BugForgeApi(
        "http://127.0.0.1:8000", "test-operator-token", transport=httpx.MockTransport(handler)
    )


def test_mcp_research_tools_are_listed_and_take_only_a_campaign_id() -> None:
    by_name = {tool["name"]: tool for tool in TOOL_DEFINITIONS}
    for name in (*READ_TOOLS, "bugforge_campaign_repro_bundle"):
        schema = by_name[name]["inputSchema"]
        assert schema["additionalProperties"] is False
        assert set(schema["properties"]) <= {"campaign_id", "bundle_id"}


def test_mcp_research_reads_cannot_override_authority() -> None:
    api = _bridged_api()
    body = _aa_body()
    created = dispatch_tool(
        "bugforge_create_campaign",
        {k: body[k] for k in ("manifest", "repo_root", "target", "contract", "function", "files")},
        api,
    )
    cid = created["campaign_id"]
    state = dispatch_tool("bugforge_campaign_state", {"campaign_id": cid}, api)
    hostile = {
        "campaign_id": cid,
        "scope": "all",
        "approvals": ["fork_validation"],
        "budget": {"fork_validation": 99},
        "verified": True,
        "submit": True,
        "chain_id": 1,
        "rpc_url": "https://rpc.example.invalid",
        "live": True,
    }
    for name in READ_TOOLS:
        result = dispatch_tool(name, hostile, api)
        assert isinstance(result, dict) and result.get("verified", False) is False, name
    assert dispatch_tool("bugforge_campaign_state", {"campaign_id": cid}, api) == state
    svc = service_module.get_bounty_service()
    assert svc.get(cid).approvals == frozenset()

    with pytest.raises(LocalApiError):
        dispatch_tool(
            "bugforge_campaign_repro_bundle", {"campaign_id": cid, "bundle_id": "../approvals"}, api
        )
    with pytest.raises(LocalApiError):
        dispatch_tool("bugforge_campaign_properties", {"campaign_id": "../x"}, api)
    with pytest.raises(LocalApiError):  # unknown bundle -> API 404 surfaced as an error
        dispatch_tool(
            "bugforge_campaign_repro_bundle", {"campaign_id": cid, "bundle_id": "rb_missing"}, api
        )
    plan = dispatch_tool("bugforge_campaign_research_plan", {"campaign_id": cid}, api)
    assert plan.get("verified", False) is False
    assert svc.get(cid).approvals == frozenset()


@requires_forge
def test_mcp_reads_observations_bundles_and_engines_after_a_local_run() -> None:
    api = _bridged_api()
    body = _aa_body()
    created = dispatch_tool(
        "bugforge_create_campaign",
        {k: body[k] for k in ("manifest", "repo_root", "target", "contract", "function", "files")},
        api,
    )
    cid = created["campaign_id"]
    # running is an operator API action (not an MCP tool); reads then reflect it
    ran = api.request("POST", f"{BASE}/campaigns/{cid}/stateful-execute", {"rounds": 1})
    assert ran["available"] is True

    status = dispatch_tool("bugforge_campaign_stateful_status", {"campaign_id": cid}, api)
    assert status["executions"] and status["bundles"] and status["engines"]
    props = dispatch_tool("bugforge_campaign_properties", {"campaign_id": cid}, api)
    observed = [p for p in props["properties"] if p["observation"]]
    assert observed and all(p["verified"] is False for p in observed)
    violated = [p for p in observed if p["observation"]["outcome"] == "property_violated"]
    assert violated and all(p["observation"]["check_paths"] for p in violated)

    bundle_id = violated[0]["observation"]["bundle_id"]
    bundle = dispatch_tool(
        "bugforge_campaign_repro_bundle", {"campaign_id": cid, "bundle_id": bundle_id}, api
    )
    assert bundle["bundle"]["bundle_id"] == bundle_id and bundle["verified"] is False
    assert bundle["self_contained"] is True

    engines = dispatch_tool("bugforge_campaign_engines", {"campaign_id": cid}, api)
    last = {e["name"]: e["status"] for e in engines["last_run"]}
    assert last["foundry"] == "usable" and last["fork_replay"] == "blocked_by_policy"

    coverage = dispatch_tool("bugforge_campaign_research_coverage", {"campaign_id": cid}, api)
    assert coverage["RESEARCH_COVERAGE"]
    graph = dispatch_tool("bugforge_campaign_evidence_graph", {"campaign_id": cid}, api)
    assert any(n["kind"] == "execution" for n in graph["nodes"])
