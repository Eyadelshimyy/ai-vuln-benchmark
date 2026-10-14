"""Tests for Phase 5: test generation, validator, and test-gen API."""

from __future__ import annotations

from pathlib import Path


class TestValidator:
    def test_valid_test_passes(self) -> None:
        from app.testing.test_validator import validate_test_code

        code = "def test_add():\n    assert 1 + 1 == 2\n"
        result = validate_test_code(code)
        assert result.valid is True
        assert "test_add" in (result.test_function_names or [])

    def test_empty_code_invalid(self) -> None:
        from app.testing.test_validator import validate_test_code

        assert validate_test_code("").valid is False

    def test_syntax_error_invalid(self) -> None:
        from app.testing.test_validator import validate_test_code

        assert validate_test_code("def broken(\n").valid is False

    def test_no_test_function_invalid(self) -> None:
        from app.testing.test_validator import validate_test_code

        result = validate_test_code("x = 1\n")
        assert result.valid is False

    def test_dangerous_exec_rejected(self) -> None:
        from app.testing.test_validator import validate_test_code

        code = "def test_bad():\n    exec('import os')\n    assert True\n"
        result = validate_test_code(code)
        assert result.valid is False

    def test_pytest_raises_counts_as_assertion(self) -> None:
        from app.testing.test_validator import validate_test_code

        code = (
            "import pytest\n"
            "def test_raises():\n"
            "    with pytest.raises(ValueError):\n"
            "        raise ValueError()\n"
        )
        result = validate_test_code(code)
        assert result.valid is True

    def test_class_based_test(self) -> None:
        from app.testing.test_validator import validate_test_code

        code = "class TestFoo:\n    def test_bar(self):\n        assert True\n"
        result = validate_test_code(code)
        assert result.valid is True


class TestQualityScorer:
    def test_with_assertion_scores_well(self) -> None:
        from app.testing.test_validator import compute_quality_score

        code = "def test_add():\n    assert 1 + 1 == 2\n"
        score, _ = compute_quality_score(code, "add")
        assert score > 0.5

    def test_trivial_assert_true_penalized(self) -> None:
        from app.testing.test_validator import compute_quality_score

        code = "def test_trivial():\n    assert True\n"
        score, notes = compute_quality_score(code, "add")
        assert score < 0.8
        assert "trivial" in notes.lower()

    def test_no_assertions_penalized(self) -> None:
        from app.testing.test_validator import compute_quality_score

        code = "def test_nothing():\n    pass\n"
        score, _ = compute_quality_score(code, "foo")
        assert score < 0.7

    def test_missing_target_symbol_penalized(self) -> None:
        from app.testing.test_validator import compute_quality_score

        code = "def test_other():\n    assert 1 == 1\n"
        score, notes = compute_quality_score(code, "Calculator.divide")
        assert "divide" in notes.lower() or score < 1.0


class TestPathNormalization:
    def test_relative_path_correct(self, tmp_path: Path) -> None:
        from app.core.paths import to_relative_path

        root = tmp_path
        child = tmp_path / "src" / "calc.py"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        result = to_relative_path(child, root)
        assert result == "src/calc.py"

    def test_outside_root_raises(self, tmp_path: Path) -> None:
        from app.core.paths import to_relative_path

        root = tmp_path / "project"
        root.mkdir()
        outside = tmp_path / "other.py"
        outside.touch()
        import pytest

        with pytest.raises(ValueError, match="not inside"):
            to_relative_path(outside, root)

    def test_is_within_repo(self, tmp_path: Path) -> None:
        from app.core.paths import is_within_repo

        root = tmp_path
        child = tmp_path / "src" / "x.py"
        child.parent.mkdir()
        child.touch()
        assert is_within_repo(child, root) is True
        assert is_within_repo(tmp_path.parent / "other.py", root) is False


class TestTestGenerationAPI:
    async def test_start_test_generation_returns_202(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "TG", "repository_path": str(repo)}
        )
        assert proj.status_code == 201
        project_id = proj.json()["id"]

        resp = await client.post(f"/api/v1/projects/{project_id}/test-generation", json={})
        assert resp.status_code == 202
        data = resp.json()
        assert "id" in data
        assert data["status"] in ("pending", "running", "completed", "failed")

    async def test_get_test_generation_session(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "TG2", "repository_path": str(repo)}
        )
        project_id = proj.json()["id"]
        start = await client.post(f"/api/v1/projects/{project_id}/test-generation", json={})
        session_id = start.json()["id"]

        resp = await client.get(f"/api/v1/test-generation/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == session_id

    async def test_list_generated_tests(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects", json={"name": "TG3", "repository_path": str(repo)}
        )
        project_id = proj.json()["id"]
        start = await client.post(f"/api/v1/projects/{project_id}/test-generation", json={})
        session_id = start.json()["id"]

        resp = await client.get(f"/api/v1/test-generation/{session_id}/tests")
        assert resp.status_code == 200
        assert "items" in resp.json()

    async def test_start_generation_nonexistent_project(self, client) -> None:
        resp = await client.post(
            "/api/v1/projects/00000000-0000-0000-0000-000000000000/test-generation",
            json={},
        )
        assert resp.status_code == 404

    async def test_get_nonexistent_session(self, client) -> None:
        resp = await client.get("/api/v1/test-generation/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text(
        "class Calculator:\n    def add(self, a, b):\n        return a + b\n"
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    return tmp_path
