"""Phase 51 MCP end-to-end: a Cursor tool call reaches the real campaign API.

The MCP server is not an AI. It is allow-listed, loopback-only, and cannot grant
an approval, change scope, raise a budget, or verify a finding. This drives the
real FastAPI app through a bridged transport so the path MCP -> API -> campaign ->
Phase 50 engines -> Phase 49 orchestrator -> evidence -> report is exercised.
"""

from __future__ import annotations

import httpx
import pytest
from starlette.testclient import TestClient

import app.discovery.bounty.service as service_module
from app.cursor_control.mcp_server import (
    FORBIDDEN_MCP_TOOLS,
    TOOL_DEFINITIONS,
    BugForgeApi,
    LocalApiError,
    dispatch_tool,
    handle_message,
)
from app.main import app
from tests.test_phase51.phase51_support import FIXTURES, create_body


@pytest.fixture(autouse=True)
def _fresh_service():
    service_module._SERVICE = None
    yield
    service_module._SERVICE = None


def _bridged_api() -> BugForgeApi:
    """A BugForgeApi whose transport forwards to the real ASGI app via TestClient."""
    test_client = TestClient(app)

    def handler(request: httpx.Request) -> httpx.Response:
        relative = str(request.url).split(request.url.host, 1)[-1]
        relative = relative.split("/", 1)[-1]
        path = request.url.raw_path.decode()
        resp = test_client.request(
            request.method,
            path,
            content=request.content,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        )
        return httpx.Response(resp.status_code, content=resp.content, headers=dict(resp.headers))

    return BugForgeApi(
        "http://127.0.0.1:8000",
        "test-operator-token",
        transport=httpx.MockTransport(handler),
    )


def test_bounty_tools_are_listed() -> None:
    api = _bridged_api()
    listed = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, api)
    names = {tool["name"] for tool in listed["result"]["tools"]}
    for expected in {
        "bugforge_create_campaign",
        "bugforge_campaign_analyze",
        "bugforge_campaign_findings",
        "bugforge_campaign_report",
    }:
        assert expected in names


def test_mcp_create_analyze_and_report_flow() -> None:
    api = _bridged_api()
    body = create_body()
    created = dispatch_tool(
        "bugforge_create_campaign",
        {
            "manifest": body["manifest"],
            "repo_root": str(FIXTURES),
            "target": body["target"],
            "contract": body["contract"],
            "function": body["function"],
            "source_file": body["source_file"],
            "files": body["files"],
            "max_rounds": 12,
        },
        api,
    )
    campaign_id = created["campaign_id"]
    assert campaign_id.startswith("cp_")
    assert created["verified"] is False and created["llm_invoked"] is False

    analyzed = dispatch_tool("bugforge_campaign_analyze", {"campaign_id": campaign_id}, api)
    assert "caller_context_analysis" in set(analyzed["exercised_capabilities"])
    assert analyzed["verified"] is False

    findings = dispatch_tool("bugforge_campaign_findings", {"campaign_id": campaign_id}, api)
    assert findings["findings"] and all(not f["verified"] for f in findings["findings"])

    report = dispatch_tool("bugforge_campaign_report", {"campaign_id": campaign_id}, api)
    assert report["verified"] is False and report["submitted"] is False
    assert "Not verified" in report["markdown"]


def test_mcp_cannot_reach_the_approvals_path() -> None:
    api = _bridged_api()
    # The approvals path is not reachable at all from the MCP client.
    with pytest.raises(LocalApiError):
        api.request(
            "POST", "/api/v1/bounty/campaigns/cp_x/approvals", {"capability": "fork_validation"}
        )


def test_mcp_forbidden_tools_cover_approval_and_scope() -> None:
    assert "grant_approval" in FORBIDDEN_MCP_TOOLS
    assert "change_scope" in FORBIDDEN_MCP_TOOLS
    assert "grant_campaign_approval" in FORBIDDEN_MCP_TOOLS
    api = _bridged_api()
    with pytest.raises(LocalApiError):
        dispatch_tool("grant_campaign_approval", {"campaign_id": "cp_x"}, api)


def test_mcp_token_is_never_echoed_in_errors() -> None:
    api = _bridged_api()
    message = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "bugforge_campaign_state",
                "arguments": {"campaign_id": "cp_missing"},
            },
        },
        api,
    )
    text = str(message)
    assert "test-operator-token" not in text


def test_tool_count_includes_session_and_campaign_tools() -> None:
    names = {tool["name"] for tool in TOOL_DEFINITIONS}
    assert "bugforge_status" in names  # Phase 49/50 session tools preserved
    assert "bugforge_create_campaign" in names  # Phase 51 campaign tools added


def _create_via_mcp(api: BugForgeApi) -> str:
    body = create_body()
    created = dispatch_tool(
        "bugforge_create_campaign",
        {
            "manifest": body["manifest"],
            "repo_root": str(FIXTURES),
            "target": body["target"],
            "contract": body["contract"],
            "function": body["function"],
            "source_file": body["source_file"],
            "files": body["files"],
            "max_rounds": 12,
        },
        api,
    )
    return str(created["campaign_id"])


def test_mcp_exposes_read_tools_for_selection_scope_and_advisories() -> None:
    api = _bridged_api()
    cid = _create_via_mcp(api)

    selection = dispatch_tool("bugforge_campaign_source_selection", {"campaign_id": cid}, api)
    assert "selected" in selection or "mode" in selection

    scope = dispatch_tool("bugforge_campaign_scope_identity", {"campaign_id": cid}, api)
    assert isinstance(scope, dict)

    advisories = dispatch_tool("bugforge_campaign_advisories", {"campaign_id": cid}, api)
    assert isinstance(advisories, dict)

    progress = dispatch_tool("bugforge_campaign_progress", {"campaign_id": cid}, api)
    assert isinstance(progress, dict)


def test_mcp_vfcs_feedback_steers_without_widening_scope() -> None:
    api = _bridged_api()
    cid = _create_via_mcp(api)
    result = dispatch_tool(
        "bugforge_campaign_vfcs_feedback",
        {"campaign_id": cid, "signals": [{"sequence_id": "seq-1", "outcome": "reverted"}]},
        api,
    )
    assert isinstance(result, dict)
    # feedback never verifies or submits
    assert result.get("verified", False) is False


def test_mcp_pause_resume_stop_flow_through_the_state_machine() -> None:
    api = _bridged_api()
    cid = _create_via_mcp(api)

    paused = dispatch_tool(
        "bugforge_campaign_pause", {"campaign_id": cid, "reason": "operator"}, api
    )
    assert isinstance(paused, dict)

    resumed = dispatch_tool("bugforge_campaign_resume", {"campaign_id": cid}, api)
    assert isinstance(resumed, dict)

    stopped = dispatch_tool(
        "bugforge_campaign_stop", {"campaign_id": cid, "reason": "operator"}, api
    )
    assert isinstance(stopped, dict)
    # once stopped, resume is refused by the real state machine
    with pytest.raises(LocalApiError):
        dispatch_tool("bugforge_campaign_resume", {"campaign_id": cid}, api)


def test_mcp_new_campaign_tools_are_listed() -> None:
    names = {tool["name"] for tool in TOOL_DEFINITIONS}
    for expected in {
        "bugforge_campaign_source_selection",
        "bugforge_campaign_scope_identity",
        "bugforge_campaign_advisories",
        "bugforge_campaign_vfcs_feedback",
        "bugforge_campaign_pause",
        "bugforge_campaign_resume",
        "bugforge_campaign_stop",
        "bugforge_campaign_progress",
    }:
        assert expected in names
