"""Phase 2 API surface for AI status and security analyzers."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_ai_status_includes_thinking_and_analyzers() -> None:
    client = TestClient(app)
    resp = client.get("/api/v1/ai/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "thinking_enabled" in data
    assert "language_analyzers" in data
    assert "security_analysis_status" in data
    assert "python" in data["language_analyzers"]


def test_security_status_lists_rules_and_languages() -> None:
    client = TestClient(app)
    resp = client.get("/api/v1/security/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "available"
    ids = {item["language_id"] for item in data["language_analyzers"]}
    assert "python" in ids
    assert "javascript" in ids
    assert "ruby" in ids
    assert any(rid.startswith("sec.taint") for rid in data["rule_ids"])
    assert "HackerOne" in data["notes"]
    assert "ScopeGuard" in data["notes"]
