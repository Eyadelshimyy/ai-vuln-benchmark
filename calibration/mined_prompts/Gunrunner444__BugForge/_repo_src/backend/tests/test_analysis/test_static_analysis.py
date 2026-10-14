"""Tests for Phase 3 static analysis rules and engine."""

from __future__ import annotations

import textwrap
from pathlib import Path

from app.analysis.finding import Finding
from app.analysis.python_analyzer import (
    BareExceptRule,
    BroadExceptionCatchRule,
    ComparisonToNoneRule,
    HardcodedCredentialRule,
    MutableDefaultArgumentRule,
    UnreachableCodeRule,
)


def _findings(rule, code: str, tmp_path: Path) -> list[Finding]:
    f = tmp_path / "test_file.py"
    f.write_text(textwrap.dedent(code))
    return rule.check_file(f, f.read_text())


class TestMutableDefaultArgument:
    def test_list_default_detected(self, tmp_path: Path) -> None:
        findings = _findings(MutableDefaultArgumentRule(), "def foo(x=[]): pass", tmp_path)
        assert len(findings) == 1
        assert findings[0].category == "mutable_default_argument"

    def test_dict_default_detected(self, tmp_path: Path) -> None:
        findings = _findings(MutableDefaultArgumentRule(), "def foo(x={}): pass", tmp_path)
        assert len(findings) == 1

    def test_immutable_default_clean(self, tmp_path: Path) -> None:
        findings = _findings(
            MutableDefaultArgumentRule(), "def foo(x=None): pass\ndef bar(y=42): pass", tmp_path
        )
        assert findings == []

    def test_syntax_error_returns_empty(self, tmp_path: Path) -> None:
        findings = _findings(MutableDefaultArgumentRule(), "def broken(", tmp_path)
        assert findings == []


class TestBareExcept:
    def test_bare_except_detected(self, tmp_path: Path) -> None:
        code = "try:\n    pass\nexcept:\n    pass\n"
        findings = _findings(BareExceptRule(), code, tmp_path)
        assert len(findings) == 1
        assert findings[0].category == "bare_except"

    def test_named_except_clean(self, tmp_path: Path) -> None:
        code = "try:\n    pass\nexcept ValueError:\n    pass\n"
        findings = _findings(BareExceptRule(), code, tmp_path)
        assert findings == []


class TestBroadExceptionCatch:
    def test_broad_except_no_reraise(self, tmp_path: Path) -> None:
        code = "try:\n    pass\nexcept Exception:\n    pass\n"
        findings = _findings(BroadExceptionCatchRule(), code, tmp_path)
        assert len(findings) == 1

    def test_broad_except_with_reraise_clean(self, tmp_path: Path) -> None:
        code = "try:\n    pass\nexcept Exception:\n    raise\n"
        findings = _findings(BroadExceptionCatchRule(), code, tmp_path)
        assert findings == []

    def test_specific_exception_clean(self, tmp_path: Path) -> None:
        code = "try:\n    pass\nexcept ValueError:\n    pass\n"
        findings = _findings(BroadExceptionCatchRule(), code, tmp_path)
        assert findings == []


class TestComparisonToNone:
    def test_eq_none_detected(self, tmp_path: Path) -> None:
        findings = _findings(ComparisonToNoneRule(), "if x == None: pass", tmp_path)
        assert len(findings) == 1
        assert findings[0].suggested_fix

    def test_neq_none_detected(self, tmp_path: Path) -> None:
        findings = _findings(ComparisonToNoneRule(), "if x != None: pass", tmp_path)
        assert len(findings) == 1

    def test_is_none_clean(self, tmp_path: Path) -> None:
        findings = _findings(
            ComparisonToNoneRule(), "if x is None: pass\nif y is not None: pass", tmp_path
        )
        assert findings == []


class TestHardcodedCredential:
    def test_password_string_detected(self, tmp_path: Path) -> None:
        findings = _findings(HardcodedCredentialRule(), 'password = "supersecret123"', tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == "high"

    def test_placeholder_ignored(self, tmp_path: Path) -> None:
        findings = _findings(HardcodedCredentialRule(), 'password = "change_me"', tmp_path)
        assert findings == []

    def test_env_var_clean(self, tmp_path: Path) -> None:
        findings = _findings(
            HardcodedCredentialRule(), 'password = os.environ.get("PASSWORD")', tmp_path
        )
        assert findings == []


class TestUnreachableCode:
    def test_after_return_detected(self, tmp_path: Path) -> None:
        code = "def foo():\n    return 1\n    x = 2\n"
        findings = _findings(UnreachableCodeRule(), code, tmp_path)
        assert len(findings) == 1

    def test_normal_flow_clean(self, tmp_path: Path) -> None:
        code = "def foo():\n    x = 1\n    return x\n"
        findings = _findings(UnreachableCodeRule(), code, tmp_path)
        assert findings == []


class TestStaticAnalysisEngine:
    def test_engine_runs_all_rules(self, tmp_path: Path) -> None:
        from app.analysis.engine import StaticAnalysisEngine

        py_file = tmp_path / "code.py"
        py_file.write_text("def foo(x=[]):\n    if x == None: pass\n")
        engine = StaticAnalysisEngine()
        findings = engine.analyze_repository(tmp_path, [py_file])
        categories = {f.category for f in findings}
        assert "mutable_default_argument" in categories
        assert "comparison_to_none" in categories

    def test_engine_skips_detection_only_formats(self, tmp_path: Path) -> None:
        from app.analysis.engine import StaticAnalysisEngine

        md_file = tmp_path / "README.md"
        md_file.write_text("# var x = 1\n")
        findings = StaticAnalysisEngine().analyze_repository(tmp_path, [md_file])
        assert findings == []

    def test_engine_handles_syntax_error_gracefully(self, tmp_path: Path) -> None:
        from app.analysis.engine import StaticAnalysisEngine

        bad_file = tmp_path / "broken.py"
        bad_file.write_text("def broken(")
        findings = StaticAnalysisEngine().analyze_repository(tmp_path, [bad_file])
        assert isinstance(findings, list)

    def test_engine_custom_rules(self, tmp_path: Path) -> None:
        from app.analysis.engine import StaticAnalysisEngine

        py_file = tmp_path / "code.py"
        py_file.write_text("def foo(x=[]): pass\n")
        findings = StaticAnalysisEngine(rules=[MutableDefaultArgumentRule()]).analyze_repository(
            tmp_path, [py_file]
        )
        assert len(findings) == 1


class TestFindings:
    """Test the findings API endpoint."""

    async def test_findings_endpoint_returns_200(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj_resp = await client.post(
            "/api/v1/projects",
            json={"name": "SA Project", "repository_path": str(repo)},
        )
        assert proj_resp.status_code == 201
        project_id = proj_resp.json()["id"]

        analysis_resp = await client.post(f"/api/v1/projects/{project_id}/analyze")
        assert analysis_resp.status_code == 202
        analysis_id = analysis_resp.json()["id"]

        resp = await client.get(f"/api/v1/analyses/{analysis_id}/findings")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data

    async def test_findings_nonexistent_analysis(self, client) -> None:
        resp = await client.get("/api/v1/analyses/00000000-0000-0000-0000-000000000000/findings")
        assert resp.status_code == 404


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "buggy.py").write_text("def foo(x=[]):\n    if x == None: pass\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sample'\n")
    return tmp_path
