from __future__ import annotations

import pytest


@pytest.fixture
def sample_project_payload(tmp_path):
    return {
        "name": "Test Project",
        "description": "A test project",
        "repository_path": str(tmp_path),
    }


class TestProjectCRUD:
    async def test_create_project(self, client, sample_project_payload) -> None:
        response = await client.post("/api/v1/projects", json=sample_project_payload)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Project"
        assert data["description"] == "A test project"
        assert "id" in data
        assert "created_at" in data

    async def test_list_projects_empty(self, client) -> None:
        response = await client.get("/api/v1/projects")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data

    async def test_list_projects_with_entry(self, client, sample_project_payload) -> None:
        await client.post("/api/v1/projects", json=sample_project_payload)
        response = await client.get("/api/v1/projects")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1

    async def test_get_project_by_id(self, client, sample_project_payload) -> None:
        create_resp = await client.post("/api/v1/projects", json=sample_project_payload)
        project_id = create_resp.json()["id"]

        response = await client.get(f"/api/v1/projects/{project_id}")
        assert response.status_code == 200
        assert response.json()["id"] == project_id

    async def test_get_nonexistent_project(self, client) -> None:
        response = await client.get("/api/v1/projects/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    async def test_delete_project(self, client, sample_project_payload) -> None:
        create_resp = await client.post("/api/v1/projects", json=sample_project_payload)
        project_id = create_resp.json()["id"]

        delete_resp = await client.delete(f"/api/v1/projects/{project_id}")
        assert delete_resp.status_code == 204

        get_resp = await client.get(f"/api/v1/projects/{project_id}")
        assert get_resp.status_code == 404

    async def test_create_project_invalid_path(self, client) -> None:
        response = await client.post(
            "/api/v1/projects",
            json={"name": "Bad", "repository_path": "/does/not/exist/12345"},
        )
        assert response.status_code == 422

    async def test_create_project_relative_path(self, client) -> None:
        response = await client.post(
            "/api/v1/projects",
            json={"name": "Bad", "repository_path": "relative/path"},
        )
        assert response.status_code == 422

    async def test_create_project_missing_name(self, client, tmp_path) -> None:
        response = await client.post(
            "/api/v1/projects",
            json={"repository_path": str(tmp_path)},
        )
        assert response.status_code == 422

    async def test_pagination_offset_limit(self, client, tmp_path) -> None:
        for i in range(3):
            await client.post(
                "/api/v1/projects",
                json={"name": f"Project {i}", "repository_path": str(tmp_path)},
            )
        response = await client.get("/api/v1/projects?offset=0&limit=2")
        assert response.status_code == 200
        assert len(response.json()["items"]) <= 2
