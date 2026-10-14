"""Security rule engine, taint, correlation, and false-positive checks."""

from __future__ import annotations

from pathlib import Path

from app.domain.findings import FindingStatus
from app.domain.security import EvidenceTier, VulnerabilityClass
from app.plugins import reset_plugin_catalog
from app.security.engine import SecurityAnalysisEngine
from app.security.rules.catalog import builtin_security_rules


def setup_function() -> None:
    reset_plugin_catalog()


def teardown_function() -> None:
    reset_plugin_catalog()


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "security"


def _copy(tmp: Path, rel: str) -> Path:
    src = FIXTURE_ROOT / rel
    dest = tmp / src.name
    dest.write_text(src.read_text(encoding="utf-8"))
    return dest


def test_builtin_rules_are_documented() -> None:
    for rule in builtin_security_rules():
        assert rule.rule_id
        assert rule.documentation.detects
        assert rule.documentation.evidence
        assert rule.documentation.limitations
        assert rule.documentation.false_positives


def test_python_sql_and_command_and_xss(tmp_path: Path) -> None:
    path = _copy(tmp_path, "python/vuln_app.py")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    classes = {obs.vulnerability_class for obs in result.observations}
    assert VulnerabilityClass.SQL_INJECTION in classes
    assert VulnerabilityClass.COMMAND_INJECTION in classes
    assert VulnerabilityClass.XSS in classes
    assert VulnerabilityClass.HARDCODED_SECRET in classes
    assert all(f.status is not FindingStatus.VERIFIED for f in result.findings)


def test_python_parameterized_query_is_quiet(tmp_path: Path) -> None:
    path = _copy(tmp_path, "python/safe_app.py")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    sql = [
        obs
        for obs in result.observations
        if obs.vulnerability_class is VulnerabilityClass.SQL_INJECTION
    ]
    assert sql == []


def test_javascript_command_and_sql(tmp_path: Path) -> None:
    path = _copy(tmp_path, "javascript/vuln_app.js")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    classes = {obs.vulnerability_class for obs in result.observations}
    assert VulnerabilityClass.COMMAND_INJECTION in classes
    assert VulnerabilityClass.SQL_INJECTION in classes


def test_javascript_safe_is_quiet(tmp_path: Path) -> None:
    path = _copy(tmp_path, "javascript/safe_app.js")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    noisy = {
        obs.vulnerability_class
        for obs in result.observations
        if obs.vulnerability_class
        in {VulnerabilityClass.SQL_INJECTION, VulnerabilityClass.COMMAND_INJECTION}
    }
    assert noisy == set()


def test_typescript_ssrf_and_sql(tmp_path: Path) -> None:
    path = _copy(tmp_path, "typescript/vuln_app.ts")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    classes = {obs.vulnerability_class for obs in result.observations}
    assert VulnerabilityClass.SSRF in classes
    assert VulnerabilityClass.SQL_INJECTION in classes


def test_ruby_sql_and_command(tmp_path: Path) -> None:
    path = _copy(tmp_path, "ruby/vuln_app.rb")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    classes = {obs.vulnerability_class for obs in result.observations}
    assert (
        VulnerabilityClass.SQL_INJECTION in classes or VulnerabilityClass.UNSAFE_REDIRECT in classes
    )
    assert VulnerabilityClass.COMMAND_INJECTION in classes


def test_c_command_injection(tmp_path: Path) -> None:
    path = _copy(tmp_path, "c/vuln_cmd.c")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    assert any(
        obs.vulnerability_class is VulnerabilityClass.COMMAND_INJECTION
        for obs in result.observations
    )
    safe = _copy(tmp_path, "c/safe_cmd.c")
    quiet = SecurityAnalysisEngine().analyze_repository(tmp_path, [safe])
    assert not any(
        obs.vulnerability_class is VulnerabilityClass.COMMAND_INJECTION
        and "safe_cmd" in obs.file_path
        for obs in quiet.observations
    )


def test_cpp_path_traversal(tmp_path: Path) -> None:
    path = _copy(tmp_path, "cpp/vuln_path.cpp")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    assert any(
        obs.vulnerability_class is VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL
        for obs in result.observations
    )


def test_correlation_marks_independent_rules(tmp_path: Path) -> None:
    path = _copy(tmp_path, "python/vuln_app.py")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    sql_clusters = [
        c for c in result.clusters if c.vulnerability_class is VulnerabilityClass.SQL_INJECTION
    ]
    assert sql_clusters
    # One rule can still be potential; multiple rules become corroborated.
    for finding in result.findings:
        assert finding.status in {FindingStatus.POTENTIAL, FindingStatus.CORROBORATED}
        assert finding.evidence_tier in {
            EvidenceTier.STATIC_INDICATOR,
            EvidenceTier.CORROBORATED,
        }
        assert finding.observation_refs


def test_keyword_sql_without_taint_is_not_enough(tmp_path: Path) -> None:
    path = tmp_path / "docs.py"
    path.write_text('"""This file mentions SQL injection in a comment."""\nVALUE = 1\n')
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    assert not any(
        obs.vulnerability_class is VulnerabilityClass.SQL_INJECTION for obs in result.observations
    )


def test_engine_does_not_special_case_language_names() -> None:
    import inspect

    import app.security.engine as engine_mod

    source = inspect.getsource(engine_mod)
    assert "javascript" not in source
    assert "if language" not in source
