"""Security testing HTTP API."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_security_testing_session_and_authorize(client: AsyncClient) -> None:
    created = await client.post(
        "/api/v1/security-testing/sessions",
        json={
            "project_id": "proj-lab",
            "mode": "lab",
            "program_name": "local-lab",
            "includes": [{"identifier": "127.0.0.1", "allow_active_testing": True}],
            "allow_active_testing": True,
            "dry_run": True,
            "allowed_methods": ["GET", "HEAD", "OPTIONS"],
        },
    )
    assert created.status_code == 200
    body = created.json()
    assert body["session"]["mode"] == "lab"
    assert "hack everything" not in body["notes"].lower()
    tools = await client.get("/api/v1/security-testing/tools")
    assert tools.status_code == 200
    assert "zap" in tools.json()["security_tools"]
    assert "nuclei" in tools.json()["security_tools"]
    allowed = await client.post(
        "/api/v1/security-testing/sessions/proj-lab/authorize",
        json={"target": "http://127.0.0.1/health", "method": "GET", "tool": "browser"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["allowed"] is True
    denied = await client.post(
        "/api/v1/security-testing/sessions/proj-lab/authorize",
        json={"target": "https://example.com/", "method": "GET", "tool": "browser"},
    )
    assert denied.json()["allowed"] is False
    report = await client.post(
        "/api/v1/security-testing/sessions/proj-lab/approvals",
        headers={"X-BugForge-Operator-Token": "test-operator-token"},
        json={"kind": "submit_hackerone_report", "operator": "alice"},
    )
    assert report.status_code == 200
    missing = await client.post(
        "/api/v1/security-testing/sessions/proj-lab/approvals",
        json={"kind": "submit_hackerone_report", "operator": "alice"},
    )
    assert missing.status_code == 401
    ai_report = await client.post(
        "/api/v1/security-testing/sessions/proj-lab/approvals",
        headers={"X-BugForge-Operator-Token": "wrong"},
        json={"kind": "submit_hackerone_report", "operator": "ai"},
    )
    assert ai_report.status_code == 401
    audit = await client.get("/api/v1/security-testing/sessions/proj-lab/audit")
    assert audit.status_code == 200
    assert audit.json()["chain_valid"] is True
