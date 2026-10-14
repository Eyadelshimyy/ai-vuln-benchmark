"""API tests for test-run endpoints."""

from __future__ import annotations

from pathlib import Path


def _build_sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "tests" / "test_sample.py").write_text(
        "def test_pass(): assert 1 + 1 == 2\ndef test_fail(): assert 1 == 2\n"
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sample'\n")
    return tmp_path


class TestTestRunAPI:
    async def test_start_test_run_returns_202(self, client, tmp_path: Path) -> None:
        repo = _build_sample_repo(tmp_path)
        proj_resp = await client.post(
            "/api/v1/projects",
            json={"name": "TR Project", "repository_path": str(repo)},
        )
        assert proj_resp.status_code == 201
        project_id = proj_resp.json()["id"]

        run_resp = await client.post(f"/api/v1/projects/{project_id}/tests/run")
        assert run_resp.status_code == 202
        data = run_resp.json()
        assert "id" in data
        assert data["status"] in ("pending", "running", "completed", "failed")
        assert data["project_id"] == project_id

    async def test_get_test_run(self, client, tmp_path: Path) -> None:
        repo = _build_sample_repo(tmp_path)
        proj_resp = await client.post(
            "/api/v1/projects",
            json={"name": "TR Project 2", "repository_path": str(repo)},
        )
        project_id = proj_resp.json()["id"]
        run_resp = await client.post(f"/api/v1/projects/{project_id}/tests/run")
        run_id = run_resp.json()["id"]

        get_resp = await client.get(f"/api/v1/test-runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == run_id

    async def test_get_test_run_results(self, client, tmp_path: Path) -> None:
        repo = _build_sample_repo(tmp_path)
        proj_resp = await client.post(
            "/api/v1/projects",
            json={"name": "TR Project 3", "repository_path": str(repo)},
        )
        project_id = proj_resp.json()["id"]
        run_resp = await client.post(f"/api/v1/projects/{project_id}/tests/run")
        run_id = run_resp.json()["id"]

        results_resp = await client.get(f"/api/v1/test-runs/{run_id}/results")
        assert results_resp.status_code == 200
        data = results_resp.json()
        assert "items" in data
        assert "total" in data

    async def test_list_project_test_runs(self, client, tmp_path: Path) -> None:
        repo = _build_sample_repo(tmp_path)
        proj_resp = await client.post(
            "/api/v1/projects",
            json={"name": "TR Project 4", "repository_path": str(repo)},
        )
        project_id = proj_resp.json()["id"]
        await client.post(f"/api/v1/projects/{project_id}/tests/run")

        list_resp = await client.get(f"/api/v1/projects/{project_id}/test-runs")
        assert list_resp.status_code == 200
        data = list_resp.json()
        assert data["total"] >= 1

    async def test_start_test_run_nonexistent_project(self, client) -> None:
        resp = await client.post("/api/v1/projects/00000000-0000-0000-0000-000000000000/tests/run")
        assert resp.status_code == 404

    async def test_get_nonexistent_test_run(self, client) -> None:
        resp = await client.get("/api/v1/test-runs/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404
