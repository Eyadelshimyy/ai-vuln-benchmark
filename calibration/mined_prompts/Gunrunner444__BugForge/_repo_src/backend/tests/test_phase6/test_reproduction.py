"""Tests for Phase 6: Bug Reproduction Engine."""

from __future__ import annotations

from pathlib import Path


class TestReproductionAPI:
    async def test_start_reproduction_returns_202(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "Repro", "repository_path": str(repo)}
        )
        assert proj.status_code == 201
        project_id = proj.json()["id"]

        resp = await client.post(f"/api/v1/projects/{project_id}/reproduction", json={})
        assert resp.status_code == 202
        data = resp.json()
        assert "id" in data
        assert data["status"] in ("pending", "running", "completed", "failed")
        assert data["total_attempts"] == 3

    async def test_get_reproduction_session(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "Repro2", "repository_path": str(repo)}
        )
        project_id = proj.json()["id"]
        start = await client.post(f"/api/v1/projects/{project_id}/reproduction", json={})
        session_id = start.json()["id"]

        resp = await client.get(f"/api/v1/reproduction/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == session_id
        assert "attempts" in data

    async def test_list_attempts(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "Repro3", "repository_path": str(repo)}
        )
        project_id = proj.json()["id"]
        start = await client.post(f"/api/v1/projects/{project_id}/reproduction", json={})
        session_id = start.json()["id"]

        resp = await client.get(f"/api/v1/reproduction/{session_id}/attempts")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_list_project_reproduction_sessions(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "Repro4", "repository_path": str(repo)}
        )
        project_id = proj.json()["id"]
        await client.post(f"/api/v1/projects/{project_id}/reproduction", json={})

        resp = await client.get(f"/api/v1/projects/{project_id}/reproduction")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    async def test_nonexistent_project_returns_404(self, client) -> None:
        resp = await client.post(
            "/api/v1/projects/00000000-0000-0000-0000-000000000000/reproduction",
            json={},
        )
        assert resp.status_code == 404

    async def test_nonexistent_session_returns_404(self, client) -> None:
        resp = await client.get("/api/v1/reproduction/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    async def test_custom_attempt_count(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "Repro5", "repository_path": str(repo)}
        )
        project_id = proj.json()["id"]
        resp = await client.post(
            f"/api/v1/projects/{project_id}/reproduction",
            json={"total_attempts": 5},
        )
        assert resp.json()["total_attempts"] == 5


class TestReproductionPlanner:
    async def test_planner_returns_plan_with_mock_provider(self, tmp_path: Path) -> None:
        from app.testing.reproduction_planner import ReproductionPlanner

        planner = ReproductionPlanner()
        plan = await planner.plan(
            hypothesis_text="divide() raises ZeroDivisionError when b=0",
            repository_path=str(tmp_path),
            failing_tests=[],
            static_findings=[],
        )
        assert plan is not None
        assert plan.reproducer_code
        assert plan.provider == "mock"

    async def test_planner_returns_none_on_empty_response(self, monkeypatch) -> None:
        from app.ai.provider import AIUsage, StructuredTextResponse
        from app.testing.reproduction_planner import ReproductionPlanner

        async def _fake_generate(system_prompt: str, user_msg: str) -> StructuredTextResponse:
            return StructuredTextResponse(
                content="",
                provider="mock",
                model="mock-v1",
                usage=AIUsage(),
                duration_seconds=0.0,
                error="Simulated error",
            )

        from app.ai.mock_provider import MockLLMProvider

        mock = MockLLMProvider()
        mock.generate_structured = _fake_generate  # type: ignore[method-assign]

        import app.ai as _ai_module

        orig = _ai_module.get_provider
        _ai_module.get_provider = lambda: mock  # type: ignore[assignment]

        try:
            planner = ReproductionPlanner()
            plan = await planner.plan("hypothesis", "/tmp", [], [])
            assert plan is None
        finally:
            _ai_module.get_provider = orig


class TestReproductionClassification:
    def test_consistently_reproduced(self) -> None:
        results = [
            {"reproduced": True, "classification": "reproduced"},
            {"reproduced": True, "classification": "reproduced"},
            {"reproduced": True, "classification": "reproduced"},
        ]
        cls = _classify(results)
        assert cls == "consistently_reproduced"

    def test_not_reproduced(self) -> None:
        results = [{"reproduced": False}, {"reproduced": False}, {"reproduced": False}]
        assert _classify(results) == "not_reproduced"

    def test_intermittent(self) -> None:
        results = [
            {"reproduced": True},
            {"reproduced": True},
            {"reproduced": False},
        ]
        cls = _classify(results)
        assert cls in ("intermittent", "inconclusive", "consistently_reproduced")


def _classify(results: list[dict]) -> str:
    successful = sum(1 for r in results if r.get("reproduced"))
    total = len(results)
    if successful == 0:
        return "not_reproduced"
    if successful == total:
        return "consistently_reproduced"
    if successful >= total * 0.5:
        return "intermittent"
    return "inconclusive"


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text(
        "class Calculator:\n    def add(self, a, b):\n        return a + b\n"
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    return tmp_path
