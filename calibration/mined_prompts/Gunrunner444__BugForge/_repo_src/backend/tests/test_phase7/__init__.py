"""Tests for Phase 7: Automated Repair Engine."""

from __future__ import annotations

from pathlib import Path

# ── API tests ─────────────────────────────────────────────────────────────────


class TestRepairAPI:
    async def test_start_repair_returns_202(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "Repair1", "repository_path": str(repo)}
        )
        assert proj.status_code == 201
        project_id = proj.json()["id"]

        resp = await client.post(f"/api/v1/projects/{project_id}/repair", json={})
        assert resp.status_code == 202
        data = resp.json()
        assert "id" in data
        assert data["status"] in ("pending", "running", "completed", "failed")
        assert data["total_candidates"] == 0

    async def test_get_repair_session(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "Repair2", "repository_path": str(repo)}
        )
        project_id = proj.json()["id"]
        start = await client.post(f"/api/v1/projects/{project_id}/repair", json={})
        session_id = start.json()["id"]

        resp = await client.get(f"/api/v1/repair/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == session_id
        assert "candidates" in data

    async def test_list_candidates(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "Repair3", "repository_path": str(repo)}
        )
        project_id = proj.json()["id"]
        start = await client.post(f"/api/v1/projects/{project_id}/repair", json={})
        session_id = start.json()["id"]

        resp = await client.get(f"/api/v1/repair/{session_id}/candidates")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_list_project_repair_sessions(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "Repair4", "repository_path": str(repo)}
        )
        project_id = proj.json()["id"]
        await client.post(f"/api/v1/projects/{project_id}/repair", json={})

        resp = await client.get(f"/api/v1/projects/{project_id}/repair")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    async def test_nonexistent_project_returns_404(self, client) -> None:
        resp = await client.post(
            "/api/v1/projects/00000000-0000-0000-0000-000000000000/repair",
            json={},
        )
        assert resp.status_code == 404

    async def test_nonexistent_session_returns_404(self, client) -> None:
        resp = await client.get("/api/v1/repair/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    async def test_nonexistent_candidate_returns_404(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "Repair5", "repository_path": str(repo)}
        )
        project_id = proj.json()["id"]
        start = await client.post(f"/api/v1/projects/{project_id}/repair", json={})
        session_id = start.json()["id"]

        resp = await client.get(
            f"/api/v1/repair/{session_id}/candidates/00000000-0000-0000-0000-000000000000"
        )
        assert resp.status_code == 404

    async def test_max_candidates_parameter(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "Repair6", "repository_path": str(repo)}
        )
        project_id = proj.json()["id"]
        resp = await client.post(
            f"/api/v1/projects/{project_id}/repair",
            json={"max_candidates": 2},
        )
        assert resp.status_code == 202


# ── Patch validator tests ─────────────────────────────────────────────────────


class TestPatchValidator:
    def test_valid_patch(self) -> None:
        from app.testing.patch_validator import validate_patch

        diff = (
            "--- a/src/module.py\n"
            "+++ b/src/module.py\n"
            "@@ -1,3 +1,4 @@\n"
            " def foo():\n"
            "-    return None\n"
            "+    return 42\n"
        )
        result = validate_patch(diff, ["src/module.py"])
        assert result.valid

    def test_empty_diff_rejected(self) -> None:
        from app.testing.patch_validator import validate_patch

        result = validate_patch("", ["src/module.py"])
        assert not result.valid
        assert result.error is not None

    def test_empty_changed_files_rejected(self) -> None:
        from app.testing.patch_validator import validate_patch

        diff = "--- a/src/module.py\n+++ b/src/module.py\n@@ -1 +1 @@\n-x\n+y\n"
        result = validate_patch(diff, [])
        assert not result.valid

    def test_absolute_path_rejected(self) -> None:
        from app.testing.patch_validator import validate_patch

        diff = "--- a/src/module.py\n+++ b/src/module.py\n@@ -1 +1 @@\n-x\n+y\n"
        result = validate_patch(diff, ["/absolute/path.py"])
        assert not result.valid
        assert "absolute" in (result.error or "").lower()

    def test_path_traversal_rejected(self) -> None:
        from app.testing.patch_validator import validate_patch

        diff = "--- a/../etc/passwd\n+++ b/../etc/passwd\n@@ -1 +1 @@\n-x\n+y\n"
        result = validate_patch(diff, ["../etc/passwd"])
        assert not result.valid

    def test_dangerous_import_in_added_lines_rejected(self) -> None:
        from app.testing.patch_validator import validate_patch

        diff = (
            "--- a/src/module.py\n"
            "+++ b/src/module.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def foo():\n"
            "+    import subprocess\n"
            "     pass\n"
        )
        result = validate_patch(diff, ["src/module.py"])
        assert not result.valid

    def test_dangerous_call_in_added_lines_rejected(self) -> None:
        from app.testing.patch_validator import validate_patch

        diff = (
            "--- a/src/module.py\n"
            "+++ b/src/module.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def foo():\n"
            "+    os.system('cmd')\n"
            "     pass\n"
        )
        result = validate_patch(diff, ["src/module.py"])
        assert not result.valid

    def test_diff_references_undeclared_file_rejected(self) -> None:
        from app.testing.patch_validator import validate_patch

        diff = "--- a/src/module.py\n+++ b/src/module.py\n@@ -1 +1 @@\n-x\n+y\n"
        result = validate_patch(diff, ["src/other.py"])
        assert not result.valid


# ── Patch planner tests ───────────────────────────────────────────────────────


class TestPatchPlanner:
    async def test_planner_returns_plan_with_mock_provider(self, tmp_path: Path) -> None:
        from app.testing.patch_planner import PatchPlanner

        planner = PatchPlanner()
        plan = await planner.plan(
            hypothesis_text="divide() raises ZeroDivisionError when b=0",
            reproduction_summary="Classification: consistently_reproduced",
            affected_files_content={},
            static_findings=[],
            existing_test_failures=[],
        )
        assert plan is not None
        assert plan.patch_diff
        assert plan.provider == "mock"

    async def test_planner_returns_none_on_error(self, monkeypatch) -> None:
        from app.ai.provider import AIUsage, StructuredTextResponse
        from app.testing.patch_planner import PatchPlanner

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
        mock.generate_patch = _fake_generate  # type: ignore[method-assign]

        import app.ai as _ai_module

        orig = _ai_module.get_provider
        _ai_module.get_provider = lambda: mock  # type: ignore[assignment]

        try:
            planner = PatchPlanner()
            plan = await planner.plan("hypothesis", "", {}, [], [])
            assert plan is None
        finally:
            _ai_module.get_provider = orig


# ── Repair workspace tests ────────────────────────────────────────────────────


class TestRepairWorkspace:
    def test_create_and_destroy(self, tmp_path: Path) -> None:
        from app.testing.repair_workspace import RepairWorkspace

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "module.py").write_text("x = 1\n")

        ws = RepairWorkspace.create(str(tmp_path))
        assert Path(ws.repo_root).is_dir()
        assert (Path(ws.repo_root) / "src" / "module.py").exists()

        ws.destroy()
        assert not Path(ws.workspace_dir).exists()

    def test_context_manager(self, tmp_path: Path) -> None:
        from app.testing.repair_workspace import RepairWorkspace

        (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")

        workspace_dir = None
        with RepairWorkspace.create(str(tmp_path)) as ws:
            workspace_dir = ws.workspace_dir
            assert Path(ws.repo_root).is_dir()

        assert workspace_dir is not None
        assert not Path(workspace_dir).exists()

    def test_apply_patch_simple_replacement(self, tmp_path: Path) -> None:
        from app.testing.repair_workspace import RepairWorkspace

        (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")

        with RepairWorkspace.create(str(tmp_path)) as ws:
            diff = (
                "--- a/calc.py\n"
                "+++ b/calc.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def add(a, b):\n"
                "-    return a - b\n"
                "+    return a + b\n"
            )
            ok, err = ws.apply_patch(diff)
            assert ok, err
            patched = (Path(ws.repo_root) / "calc.py").read_text()
            assert "a + b" in patched

    def test_apply_empty_patch_fails(self, tmp_path: Path) -> None:
        from app.testing.repair_workspace import RepairWorkspace

        (tmp_path / "module.py").write_text("x = 1\n")
        with RepairWorkspace.create(str(tmp_path)) as ws:
            ok, err = ws.apply_patch("")
            assert not ok

    def test_nonexistent_repo_raises(self) -> None:
        from app.testing.repair_workspace import RepairWorkspace

        try:
            RepairWorkspace.create("/nonexistent/path/that/does/not/exist")
            assert False, "Should have raised"
        except ValueError:
            pass


# ── Repair scoring tests ──────────────────────────────────────────────────────


class TestRepairScoring:
    def test_bug_fixed_no_regressions_is_max(self) -> None:
        from app.services.repair_service import RepairService

        svc = RepairService()
        score = svc._compute_score(bug_fixed=True, no_regressions=True, regression_count=0)
        assert score == 1.0

    def test_bug_fixed_with_regressions(self) -> None:
        from app.services.repair_service import RepairService

        svc = RepairService()
        score = svc._compute_score(bug_fixed=True, no_regressions=False, regression_count=2)
        assert score < 1.0
        assert score > 0.0

    def test_bug_not_fixed_is_zero(self) -> None:
        from app.services.repair_service import RepairService

        svc = RepairService()
        score = svc._compute_score(bug_fixed=False, no_regressions=True, regression_count=0)
        assert score == 0.3

    def test_unknown_bug_status_partial(self) -> None:
        from app.services.repair_service import RepairService

        svc = RepairService()
        # None means we don't know (no reproducer available)
        score = svc._compute_score(bug_fixed=None, no_regressions=True, regression_count=0)
        assert 0.0 <= score <= 1.0

    def test_many_regressions_capped(self) -> None:
        from app.services.repair_service import RepairService

        svc = RepairService()
        score = svc._compute_score(bug_fixed=True, no_regressions=False, regression_count=100)
        assert score >= 0.0


# ── Reproduction classification tests (v0.6 fix verification) ─────────────────


class TestReproductionClassification:
    def test_exit_code_1_with_matching_evidence_is_reproduced(self) -> None:
        from app.services.reproduction_service import _classify_attempt

        class FakeResult:
            exit_code = 1
            timed_out = False
            error_message = None
            stdout = "ZeroDivisionError: division by zero"
            stderr = ""

        cls, reproduced, evidence = _classify_attempt(
            FakeResult(), expected_failure_pattern="ZeroDivisionError"
        )
        assert cls == "reproduced"
        assert reproduced is True
        assert evidence is not None

    def test_exit_code_1_without_matching_evidence_is_no_evidence(self) -> None:
        from app.services.reproduction_service import _classify_attempt

        class FakeResult:
            exit_code = 1
            timed_out = False
            error_message = None
            stdout = "AssertionError: unrelated test failure"
            stderr = ""

        cls, reproduced, evidence = _classify_attempt(
            FakeResult(), expected_failure_pattern="ZeroDivisionError"
        )
        assert cls == "no_evidence"
        assert reproduced is False

    def test_exit_code_1_no_pattern_is_no_evidence(self) -> None:
        from app.services.reproduction_service import _classify_attempt

        class FakeResult:
            exit_code = 1
            timed_out = False
            error_message = None
            stdout = "FAILED test_something"
            stderr = ""

        cls, reproduced, evidence = _classify_attempt(FakeResult(), expected_failure_pattern=None)
        assert cls == "no_evidence"
        assert reproduced is False

    def test_exit_code_0_is_failed(self) -> None:
        from app.services.reproduction_service import _classify_attempt

        class FakeResult:
            exit_code = 0
            timed_out = False
            error_message = None
            stdout = ""
            stderr = ""

        cls, reproduced, _ = _classify_attempt(FakeResult(), expected_failure_pattern="error")
        assert cls == "failed"
        assert reproduced is False

    def test_timeout_is_classified_correctly(self) -> None:
        from app.services.reproduction_service import _classify_attempt

        class FakeResult:
            exit_code = -1
            timed_out = True
            error_message = None
            stdout = ""
            stderr = ""

        cls, reproduced, _ = _classify_attempt(FakeResult(), expected_failure_pattern=None)
        assert cls == "timeout"
        assert reproduced is False

    def test_environment_error(self) -> None:
        from app.services.reproduction_service import _classify_attempt

        class FakeResult:
            exit_code = 2
            timed_out = False
            error_message = "Docker not found"
            stdout = ""
            stderr = ""

        cls, reproduced, _ = _classify_attempt(FakeResult(), expected_failure_pattern=None)
        assert cls == "environment_error"
        assert reproduced is False


# ── Validator adversarial tests ────────────────────────────────────────────────


class TestValidatorAdversarial:
    def test_import_os_rejected(self) -> None:
        from app.testing.test_validator import validate_test_code

        code = "import os\n\ndef test_bad():\n    os.system('rm -rf /')\n    assert True\n"
        result = validate_test_code(code)
        assert not result.valid

    def test_from_subprocess_import_rejected(self) -> None:
        from app.testing.test_validator import validate_test_code

        code = "from subprocess import run\n\ndef test_bad():\n    run(['ls'])\n    assert True\n"
        result = validate_test_code(code)
        assert not result.valid

    def test_import_socket_rejected(self) -> None:
        from app.testing.test_validator import validate_test_code

        code = "import socket\n\ndef test_net():\n    s = socket.socket()\n    assert True\n"
        result = validate_test_code(code)
        assert not result.valid

    def test_wildcard_import_rejected(self) -> None:
        from app.testing.test_validator import validate_test_code

        code = "from some_module import *\n\ndef test_wild():\n    assert True\n"
        result = validate_test_code(code)
        assert not result.valid

    def test_eval_call_rejected(self) -> None:
        from app.testing.test_validator import validate_test_code

        code = "def test_bad():\n    eval('1+1')\n    assert True\n"
        result = validate_test_code(code)
        assert not result.valid

    def test_safe_code_accepted(self) -> None:
        from app.testing.test_validator import validate_test_code

        code = (
            "import pytest\n\n"
            "def test_divide_by_zero():\n"
            "    with pytest.raises(ZeroDivisionError):\n"
            "        result = 1 / 0\n"
        )
        result = validate_test_code(code)
        assert result.valid


# ── Production executor guard test ────────────────────────────────────────────


class TestExecutorProductionGuard:
    def test_production_raises_when_docker_unavailable(self, monkeypatch) -> None:
        import shutil

        from app.core.config import settings
        from app.execution import ExecutorFactory

        monkeypatch.setattr(shutil, "which", lambda _: None)
        monkeypatch.setattr(settings, "environment", "production")

        try:
            ExecutorFactory.create(prefer_docker=True)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as exc:
            assert "Docker is required" in str(exc)

    def test_development_falls_back_gracefully(self, monkeypatch) -> None:
        import shutil

        from app.core.config import settings
        from app.execution import ExecutorFactory, LocalTestExecutor

        monkeypatch.setattr(shutil, "which", lambda _: None)
        monkeypatch.setattr(settings, "environment", "development")

        executor = ExecutorFactory.create(prefer_docker=True)
        assert isinstance(executor, LocalTestExecutor)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text(
        "class Calculator:\n    def add(self, a, b):\n        return a + b\n"
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    return tmp_path
