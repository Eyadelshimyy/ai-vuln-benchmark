"""JavaScript/TypeScript quality static-analysis rules."""

from __future__ import annotations

from pathlib import Path

from app.analysis.base import AnalyzerRule
from app.analysis.engine import StaticAnalysisEngine
from app.analysis.finding import Finding
from app.analysis.javascript_analyzer import (
    EmptyCatchRule,
    LooseEqualityRule,
    VarDeclarationRule,
    WithStatementRule,
)
from app.domain.language import LanguageCapability
from app.plugins import get_plugin_catalog, reset_plugin_catalog


def _findings(rule: AnalyzerRule, code: str, tmp_path: Path, name: str = "a.js") -> list[Finding]:
    path = tmp_path / name
    path.write_text(code)
    return rule.check_file(path, path.read_text())


class TestLooseEquality:
    def test_eq_detected(self, tmp_path: Path) -> None:
        findings = _findings(LooseEqualityRule(), "if (x == 1) {}\n", tmp_path)
        assert len(findings) == 1
        assert findings[0].category == "js_loose_equality"
        assert findings[0].suggested_fix
        assert findings[0].severity == "low"
        assert findings[0].confidence == "high"

    def test_null_idiom_clean(self, tmp_path: Path) -> None:
        findings = _findings(
            LooseEqualityRule(),
            "if (x == null) {}\nif (y != undefined) {}\n",
            tmp_path,
        )
        assert findings == []

    def test_strict_clean(self, tmp_path: Path) -> None:
        findings = _findings(LooseEqualityRule(), "if (x === 1) {}\n", tmp_path)
        assert findings == []

    def test_string_lookalike_clean(self, tmp_path: Path) -> None:
        findings = _findings(LooseEqualityRule(), 'const s = "x == 1";\n', tmp_path)
        assert findings == []


class TestVarDeclaration:
    def test_var_detected(self, tmp_path: Path) -> None:
        findings = _findings(VarDeclarationRule(), "var x = 1;\n", tmp_path)
        assert len(findings) == 1
        assert findings[0].category == "js_var_declaration"

    def test_let_const_clean(self, tmp_path: Path) -> None:
        findings = _findings(VarDeclarationRule(), "const x = 1;\nlet y = 2;\n", tmp_path)
        assert findings == []

    def test_comment_clean(self, tmp_path: Path) -> None:
        findings = _findings(VarDeclarationRule(), "/* var x = 1 */\nconst y = 2;\n", tmp_path)
        assert findings == []


class TestEmptyCatch:
    def test_empty_detected(self, tmp_path: Path) -> None:
        findings = _findings(EmptyCatchRule(), "try { f(); } catch (e) {}\n", tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == "medium"

    def test_intentional_empty_catch_clean(self, tmp_path: Path) -> None:
        findings = _findings(
            EmptyCatchRule(),
            "try { f(); } catch (e) { /* intentionally empty */ }\n",
            tmp_path,
        )
        assert findings == []

    def test_catalog_is_code_quality(self, tmp_path: Path) -> None:
        findings = _findings(VarDeclarationRule(), "var x = 1;\n", tmp_path)
        assert findings[0].catalog == "code_quality"


class TestWithStatement:
    def test_with_detected(self, tmp_path: Path) -> None:
        findings = _findings(WithStatementRule(), "with (obj) { x = 1; }\n", tmp_path)
        assert len(findings) == 1

    def test_comment_clean(self, tmp_path: Path) -> None:
        findings = _findings(WithStatementRule(), "// with (obj) {}\nconst x = 1;\n", tmp_path)
        assert findings == []


class TestEngineIntegration:
    def setup_method(self) -> None:
        reset_plugin_catalog()

    def teardown_method(self) -> None:
        reset_plugin_catalog()

    def test_js_and_ts_are_static_analyzers(self) -> None:
        catalog = get_plugin_catalog()
        assert catalog.languages.get("javascript").supports(LanguageCapability.STATIC_ANALYSIS)
        assert catalog.languages.get("typescript").supports(LanguageCapability.STATIC_ANALYSIS)
        assert catalog.languages.get("ruby").supports(LanguageCapability.STATIC_ANALYSIS)

    def test_engine_reports_js_issues(self, tmp_path: Path) -> None:
        path = tmp_path / "app.js"
        path.write_text("var x = 1;\nif (x == 2) {}\n")
        findings = StaticAnalysisEngine().analyze_repository(tmp_path, [path])
        categories = {f.category for f in findings}
        assert "js_var_declaration" in categories
        assert "js_loose_equality" in categories

    def test_engine_skips_detection_only(self, tmp_path: Path) -> None:
        path = tmp_path / "README.md"
        path.write_text("# var x = 1\n")
        findings = StaticAnalysisEngine().analyze_repository(tmp_path, [path])
        assert findings == []

    def test_rule_override(self, tmp_path: Path) -> None:
        path = tmp_path / "app.js"
        path.write_text("var x = 1;\nif (x == 2) {}\n")
        findings = StaticAnalysisEngine(rules=[VarDeclarationRule()]).analyze_repository(
            tmp_path, [path]
        )
        assert len(findings) == 1
        assert findings[0].category == "js_var_declaration"
