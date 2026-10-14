"""API tests for discovery endpoints (Phase 11).

Uses SQLite in-memory database (via aiosqlite) and does not require
a real PostgreSQL connection or GitHub token.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class TestAIStatusEndpoint:
    def test_status_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/ai/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "provider" in data
        assert "model" in data
        assert "is_local" in data
        assert "reachable" in data
        assert "configured" in data

    def test_status_does_not_leak_api_key(self, client: TestClient) -> None:
        resp = client.get("/api/v1/ai/status")
        body = resp.text
        # Ensure no token/key values appear in the response
        assert "sk-" not in body
        assert "API_KEY" not in body
        assert "api_key" not in body

    def test_ai_test_endpoint_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/ai/test",
            json={"prompt": "hello"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "provider" in data
        assert "response" in data

    def test_ai_test_prompt_too_long(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/ai/test",
            json={"prompt": "x" * 600},
        )
        # Pydantic should reject max_length=500 prompt
        assert resp.status_code == 422

    def test_discovery_settings_no_credentials(self, client: TestClient) -> None:
        resp = client.get("/api/v1/discovery/settings")
        assert resp.status_code == 200
        data = resp.json()
        body_text = resp.text
        assert "GITHUB_TOKEN" not in body_text
        assert "github_token" not in body_text
        assert "ai_api_key" not in body_text
        assert "discovery_min_stars" in data

    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"].startswith("1.")
