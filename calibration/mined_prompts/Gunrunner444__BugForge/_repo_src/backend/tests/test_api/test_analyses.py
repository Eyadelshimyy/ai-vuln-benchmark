from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


def _build_sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "calc.py").write_text(
        textwrap.dedent(
            """
            def add(a: int, b: int) -> int:
                return a + b
            """
        )
    )
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "tests" / "test_calc.py").write_text(
        textwrap.dedent(
            """
            from src.calc import add

            def test_add():
                assert add(1, 2) == 3
            """
        )
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'sample'\n")
    return tmp_path


@pytest.fixture
def repo_path(tmp_path: Path) -> Path:
    return _build_sample_repo(tmp_path)


class TestAnalysisEndpoints:
    async def test_start_analysis_returns_202(self, client, repo_path: Path) -> None:
        create_resp = await client.post(
            "/api/v1/projects",
            json={"name": "Repo", "repository_path": str(repo_path)},
        )
        assert create_resp.status_code == 201
        project_id = create_resp.json()["id"]

        analysis_resp = await client.post(f"/api/v1/projects/{project_id}/analyze")
        assert analysis_resp.status_code == 202
        data = analysis_resp.json()
        assert "id" in data
        assert data["status"] in ("pending", "running", "completed")
        assert data["project_id"] == project_id

    async def test_get_analysis(self, client, repo_path: Path) -> None:
        create_resp = await client.post(
            "/api/v1/projects",
            json={"name": "Repo", "repository_path": str(repo_path)},
        )
        project_id = create_resp.json()["id"]
        analysis_resp = await client.post(f"/api/v1/projects/{project_id}/analyze")
        analysis_id = analysis_resp.json()["id"]

        get_resp = await client.get(f"/api/v1/analyses/{analysis_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == analysis_id

    async def test_get_nonexistent_analysis(self, client) -> None:
        response = await client.get("/api/v1/analyses/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    async def test_analyze_nonexistent_project(self, client) -> None:
        response = await client.post(
            "/api/v1/projects/00000000-0000-0000-0000-000000000000/analyze"
        )
        assert response.status_code == 404

    async def test_list_project_analyses(self, client, repo_path: Path) -> None:
        create_resp = await client.post(
            "/api/v1/projects",
            json={"name": "Repo", "repository_path": str(repo_path)},
        )
        project_id = create_resp.json()["id"]
        await client.post(f"/api/v1/projects/{project_id}/analyze")

        list_resp = await client.get(f"/api/v1/projects/{project_id}/analyses")
        assert list_resp.status_code == 200
        data = list_resp.json()
        assert "items" in data
        assert data["total"] >= 1
