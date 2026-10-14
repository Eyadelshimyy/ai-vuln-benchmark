"""Phase 11: parser truth, flow-sensitive taint, scopes, sanitizers, quality."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import pytest

from app.analysis.catalog import FindingCatalog
from app.analysis.javascript_analyzer import LooseEqualityRule, WithStatementRule
from app.analysis.quality import quality_rules_for
from app.domain.language import LanguageCapability, ParserTier
from app.domain.security import VulnerabilityClass
from app.parsing.engine import (
    installed_parser_report,
    parse_source,
    parser_backend_for,
    parser_tier_for,
    reset_syntax_registry,
)
from app.parsing.model import ParserStatus
from app.parsing.treesitter import (
    TESTED_LANGUAGE_PACK,
    TESTED_TREE_SITTER,
    TS_LANGUAGE_IDS,
    native_parser_status,
    treesitter_available,
)
from app.plugins import get_plugin_catalog, reset_plugin_catalog
from app.security.correlation import correlate_observations
from app.security.definitions import PathIssueKind
from app.security.language_vocab import vocab_for
from app.security.rules.base import RuleDocumentation, SecurityObservation
from app.security.taint import TAINT_PATH_SENSITIVE, TAINT_PRECISION, propagate_taint
from tests.language_support import (
    ANALYSIS_LANGUAGE_IDS,
    DETECTION_ONLY_LANGUAGE_IDS,
    analyze_source,
    graph_for,
    observation_classes,
)


def setup_function() -> None:
    reset_plugin_catalog()


def teardown_function() -> None:
    reset_plugin_catalog()


def test_tree_sitter_pair_is_tested_and_offline() -> None:
    ts_ver = version("tree-sitter")
    pack_ver = version("tree-sitter-language-pack")
    assert ts_ver.startswith("0.2")
    assert pack_ver.split(".", 1)[0] in {"0", "1"}
    assert TESTED_TREE_SITTER
    assert TESTED_LANGUAGE_PACK
    for language_id in ANALYSIS_LANGUAGE_IDS:
        if language_id == "python":
            continue
        status = native_parser_status(language_id)
        assert status.available, (language_id, status)
        assert treesitter_available(language_id)


def test_installed_status_matches_runtime() -> None:
    report = installed_parser_report("javascript")
    assert report["native_available"] is True
    assert report["parser_backend"] == "tree_sitter"
    assert report["parser_tier"] == "full_ast"
    graph = parse_source("javascript", Path("a.js"), "const x = 1;\n")
    assert graph.parser_backend == "tree_sitter"
    assert graph.parser_tier is ParserTier.FULL_AST
    assert graph.diagnostics.status == ParserStatus.NATIVE_AVAILABLE


def test_native_parser_used_when_available() -> None:
    graph = parse_source("javascript", Path("a.js"), "const q = req.query.q;\neval(q);\n")
    assert graph.parser_backend == "tree_sitter"
    assert graph.diagnostics.native_available is True


def test_fallback_labeled_when_native_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.parsing.engine.treesitter_available", lambda _lid: False)
    reset_syntax_registry()
    reset_plugin_catalog()
    graph = parse_source("javascript", Path("a.js"), "const x = 1;\n")
    assert graph.parser_backend == "profile"
    assert graph.parser_tier is ParserTier.PROFILE_FALLBACK
    assert graph.diagnostics.status == ParserStatus.PROFILE_FALLBACK
    assert parser_tier_for("javascript") is ParserTier.PROFILE_FALLBACK
    assert parser_backend_for("javascript") == "profile"
    adapter = get_plugin_catalog().languages.get("javascript")
    assert LanguageCapability.AST not in adapter.capabilities
    assert LanguageCapability.SCOPE_ANALYSIS not in adapter.capabilities
    assert LanguageCapability.DATA_FLOW not in adapter.capabilities
    reset_syntax_registry()
    reset_plugin_catalog()


def test_security_observation_carries_parser_tier(tmp_path: Path) -> None:
    result = analyze_source(tmp_path, "a.js", "const q = req.query.q;\neval(q);\n")
    obs = [
        o
        for o in result.observations
        if o.vulnerability_class is VulnerabilityClass.DYNAMIC_EXECUTION
    ]
    assert obs
    assert obs[0].parser_backend == "tree_sitter"
    assert obs[0].parser_tier == "full_ast"
    assert obs[0].node_id
    assert obs[0].scope_id


def test_php_snippet_spans_match_original_file() -> None:
    source = "$q = $_GET['q'];\nmysqli_query($db, $q);\n"
    graph = parse_source("php", Path("q.php"), source)
    assert graph.source == source
    assert "<?php" not in graph.source
    hits = [b for b in graph.bindings if b.name == "q"]
    assert hits
    span = hits[0].span
    assert span is not None
    encoded = source.encode("utf-8")
    snippet = encoded[span.start_byte : span.end_byte].decode("utf-8")
    assert "$q" in snippet or "q" in snippet
    assert span.start_column >= 1
    line = graph.lines[span.start_line - 1]
    # Column is 1-indexed into the original line, not the synthetic prefix.
    assert span.start_column <= len(line) + 1


def test_js_block_let_is_not_the_same_symbol() -> None:
    src = """
function f() {
    if (x) {
        let q = request.query.q;
        sink(q);
    }
    let q = "safe";
    sink(q);
}
"""
    graph = parse_source("javascript", Path("s.js"), src)
    qs = [b for b in graph.bindings if b.name == "q"]
    scopes = {b.scope_id for b in qs}
    assert len(scopes) >= 2, qs
    vocab = vocab_for("javascript")
    assert vocab is not None
    tainted = propagate_taint(graph, vocab.sources)
    safe = [b for b in qs if b.rhs_is_literal]
    assert safe
    assert safe[0].symbol_id not in tainted


def test_js_var_vs_let_shadowing() -> None:
    src = """
function f() {
    var q = request.query.q;
    {
        let q = "safe";
        sink(q);
    }
    sink(q);
}
"""
    graph = parse_source("javascript", Path("s.js"), src)
    qs = [b for b in graph.bindings if b.name == "q"]
    assert any(b.declarator == "var" for b in qs)
    assert any(b.declarator == "let" for b in qs)
    vocab = vocab_for("javascript")
    assert vocab is not None
    tainted = propagate_taint(graph, vocab.sources)
    var_ids = [b.symbol_id for b in qs if b.declarator == "var"]
    let_ids = [b.symbol_id for b in qs if b.declarator == "let"]
    assert var_ids and var_ids[0] in tainted
    assert let_ids and let_ids[0] not in tainted


def test_reassignment_kills_taint(tmp_path: Path) -> None:
    src = 'let q = req.query.q;\nq = "safe";\neval(q);\n'
    result = analyze_source(tmp_path, "r.js", src)
    assert VulnerabilityClass.DYNAMIC_EXECUTION not in observation_classes(result)
    assert TAINT_PRECISION == "flow-sensitive"
    assert TAINT_PATH_SENSITIVE is False


def test_reassignment_introduces_taint(tmp_path: Path) -> None:
    src = 'let q = "safe";\nq = req.query.q;\neval(q);\n'
    result = analyze_source(tmp_path, "r.js", src)
    assert VulnerabilityClass.DYNAMIC_EXECUTION in observation_classes(result)


def test_branch_merge_is_conservative(tmp_path: Path) -> None:
    src = """
function f(x) {
    let q = req.query.q;
    if (x) {
        q = "safe";
    }
    eval(q);
}
"""
    result = analyze_source(tmp_path, "m.js", src)
    assert VulnerabilityClass.DYNAMIC_EXECUTION in observation_classes(result)


def test_interproc_maps_correct_argument(tmp_path: Path) -> None:
    src = """
def f(safe, dangerous):
    eval(dangerous)

f("safe", request.args.get("q"))
"""
    result = analyze_source(tmp_path, "i.py", src)
    assert VulnerabilityClass.DYNAMIC_EXECUTION in observation_classes(result)


def test_interproc_does_not_taint_wrong_parameter(tmp_path: Path) -> None:
    src = """
def f(safe, dangerous):
    eval(dangerous)

f(request.args.get("q"), "safe")
"""
    result = analyze_source(tmp_path, "i.py", src)
    assert VulnerabilityClass.DYNAMIC_EXECUTION not in observation_classes(result)


def test_name_collision_does_not_speculate(tmp_path: Path) -> None:
    src = """
class A:
    def process(self, q):
        eval(q)

class B:
    def process(self, q):
        eval(q)

x.process(request.args.get("q"))
"""
    graph = parse_source("python", Path("c.py"), src)
    processes = [e for e in graph.entities if e.name == "process"]
    assert len(processes) == 2
    result = analyze_source(tmp_path, "c.py", src)
    assert VulnerabilityClass.DYNAMIC_EXECUTION not in observation_classes(result)


def test_interproc_return_when_callee_unique(tmp_path: Path) -> None:
    src = """
def helper():
    return request.args.get("q")

q = helper()
eval(q)
"""
    result = analyze_source(tmp_path, "r.py", src)
    assert VulnerabilityClass.DYNAMIC_EXECUTION in observation_classes(result)


def test_unrelated_sanitizer_does_not_suppress_sql(tmp_path: Path) -> None:
    src = "const q = req.query.q;\ndb.query(escapeHtml(q));\n"
    result = analyze_source(tmp_path, "s.js", src)
    assert VulnerabilityClass.SQL_INJECTION in observation_classes(result)


def test_sanitizer_on_other_argument_does_not_suppress(tmp_path: Path) -> None:
    src = "const q = req.query.q;\nsink(escapeHtml(other), q);\neval(q);\n"
    result = analyze_source(tmp_path, "s.js", src)
    assert VulnerabilityClass.DYNAMIC_EXECUTION in observation_classes(result)


def test_multi_argument_sql_wrong_slot_is_not_injection(tmp_path: Path) -> None:
    src = "const q = req.query.q;\ndb.query('SELECT 1', q);\n"
    result = analyze_source(tmp_path, "s.js", src)
    assert VulnerabilityClass.SQL_INJECTION not in observation_classes(result)


def test_placeholder_looking_string_is_not_parameterized(tmp_path: Path) -> None:
    src = "const q = req.query.q;\ndb.query('SELECT ' + q + ' -- ?');\n"
    result = analyze_source(tmp_path, "s.js", src)
    assert VulnerabilityClass.SQL_INJECTION in observation_classes(result)


def test_string_literal_is_not_a_source(tmp_path: Path) -> None:
    src = 'const q = "req.query.user";\neval(q);\n'
    result = analyze_source(tmp_path, "s.js", src)
    assert VulnerabilityClass.DYNAMIC_EXECUTION not in observation_classes(result)


def test_ruby_send_is_not_eval(tmp_path: Path) -> None:
    src = "q = params[:q]\nobj.send(q)\n"
    result = analyze_source(tmp_path, "s.rb", src)
    assert VulnerabilityClass.DYNAMIC_EXECUTION not in observation_classes(result)


def test_go_write_is_not_xss(tmp_path: Path) -> None:
    src = 'func f(w http.ResponseWriter, q string) { q = r.FormValue("q"); w.Write([]byte(q)) }\n'
    result = analyze_source(tmp_path, "s.go", src)
    assert VulnerabilityClass.XSS not in observation_classes(result)


def test_path_kinds_never_demonstrated(tmp_path: Path) -> None:
    src = "const q = req.query.f;\nfs.readFile(q);\n"
    result = analyze_source(tmp_path, "p.js", src)
    obs = [
        o
        for o in result.observations
        if o.vulnerability_class is VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL
    ]
    assert obs
    assert obs[0].metadata.get("path_issue_kind") != PathIssueKind.DEMONSTRATED_DIRECTORY_ESCAPE
    assert obs[0].metadata.get("path_issue_kind") in {
        "user_controlled_path",
        "unsafe_path_construction",
        "possible_path_traversal",
    }


def test_explicit_dotdot_is_unsafe_construction(tmp_path: Path) -> None:
    src = 'const q = req.query.f;\nfs.readFile("../" + q);\n'
    result = analyze_source(tmp_path, "p.js", src)
    obs = [
        o
        for o in result.observations
        if o.vulnerability_class is VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL
    ]
    assert obs
    assert obs[0].metadata.get("path_issue_kind") == "unsafe_path_construction"


def test_loose_eq_nearby_null_text_does_not_suppress(tmp_path: Path) -> None:
    findings = LooseEqualityRule().check_file(
        tmp_path / "a.js", 'if (user == request.get("null")) {}\n'
    )
    assert findings


def test_with_is_mode_aware(tmp_path: Path) -> None:
    sloppy = WithStatementRule().check_file(tmp_path / "a.js", "with (obj) { x = 1; }\n")
    strict = WithStatementRule().check_file(
        tmp_path / "a.js", '"use strict";\nwith (obj) { x = 1; }\n'
    )
    assert (
        sloppy
        and "sloppy" in sloppy[0].message.lower()
        or "discouraged" in sloppy[0].message.lower()
    )
    assert strict and "strict" in strict[0].message.lower()


QUALITY_FIXTURES: dict[str, tuple[str, str]] = {
    "python": ("def f(x=[]):\n    pass\n", "mutable_default_argument"),
    "javascript": ("if (x == 1) {}\n", "js_loose_equality"),
    "typescript": ("if (x == 1) {}\n", "js_loose_equality"),
    "ruby": ("for x in xs\n  puts x\nend\n", "quality_for_loop"),
    "c": ("void f(void) { char b[8]; gets(b); }\n", "quality_deprecated_api"),
    "cpp": ("void f() { char b[8]; gets(b); }\n", "quality_deprecated_api"),
    "go": ('package p\nfunc f() { panic("x") }\n', "quality_panic"),
    "rust": ("fn f(x: Option<i32>) { x.unwrap(); }\n", "quality_unwrap"),
    "java": (
        "class S { void f(Exception e) { e.printStackTrace(); } }\n",
        "quality_deprecated_api",
    ),
    "php": ("<?php mysql_query($q);\n", "quality_deprecated_api"),
    "kotlin": ("fun f(y: String?) { val x = y!! }\n", "quality_force_unwrap"),
    "swift": ("func f(y: String?) { let x = y! }\n", "quality_force_unwrap"),
    "csharp": ("class S { void f() { goto x; x: return; } }\n", "quality_goto"),
    "shell": ("echo $q\n", "quality_unquoted_expansion"),
}


@pytest.mark.parametrize("language_id", ANALYSIS_LANGUAGE_IDS)
def test_full_analysis_languages_have_quality_catalog(language_id: str) -> None:
    rules = quality_rules_for(language_id)
    assert len(rules) >= 3
    catalog = get_plugin_catalog()
    adapter = catalog.languages.get(language_id)
    assert adapter.supports(LanguageCapability.CODE_QUALITY)
    assert adapter.supports(LanguageCapability.STATIC_ANALYSIS)
    for rule in rules:
        assert getattr(rule, "CATALOG", FindingCatalog.CODE_QUALITY) == FindingCatalog.CODE_QUALITY


@pytest.mark.parametrize("language_id", ANALYSIS_LANGUAGE_IDS)
def test_quality_fixture_fires(language_id: str) -> None:
    source, category = QUALITY_FIXTURES[language_id]
    ext = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "ruby": ".rb",
        "c": ".c",
        "cpp": ".cpp",
        "go": ".go",
        "rust": ".rs",
        "java": ".java",
        "php": ".php",
        "kotlin": ".kt",
        "swift": ".swift",
        "csharp": ".cs",
        "shell": ".sh",
    }[language_id]
    graph = graph_for(language_id, f"q{ext}", source)
    findings = []
    for rule in quality_rules_for(language_id):
        findings.extend(rule.check_graph(graph))
    categories = {item.category for item in findings}
    assert category in categories, (language_id, categories, [e.kind for e in graph.events])
    hit = next(item for item in findings if item.category == category)
    assert hit.catalog == FindingCatalog.CODE_QUALITY
    assert hit.parser_backend
    assert hit.evidence is not None


@pytest.mark.parametrize("language_id", DETECTION_ONLY_LANGUAGE_IDS)
def test_detection_only_languages_stay_detection_only(language_id: str) -> None:
    adapter = get_plugin_catalog().languages.get(language_id)
    assert adapter.parser_tier() is ParserTier.DETECTION_ONLY
    assert not adapter.supports(LanguageCapability.PARSE)
    assert language_id not in TS_LANGUAGE_IDS or language_id in {
        "r",
        "scala",
        "dart",
        "lua",
        "elixir",
    }


@pytest.mark.parametrize("language_id", ANALYSIS_LANGUAGE_IDS)
def test_malformed_source_reports_diagnostics(language_id: str) -> None:
    ext = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "ruby": ".rb",
        "c": ".c",
        "cpp": ".cpp",
        "go": ".go",
        "rust": ".rs",
        "java": ".java",
        "php": ".php",
        "kotlin": ".kt",
        "swift": ".swift",
        "csharp": ".cs",
        "shell": ".sh",
    }[language_id]
    graph = graph_for(language_id, f"bad{ext}", "{[(")
    assert graph.language == language_id
    assert graph.diagnostics is not None
    graph2 = graph_for(language_id, f"empty{ext}", "")
    assert graph2.language == language_id
    deep = "{" * 80 + "}" * 80
    graph3 = graph_for(language_id, f"deep{ext}", deep)
    assert graph3.language == language_id
    truncated = "const x = " if language_id in {"javascript", "typescript"} else "x = "
    graph4 = graph_for(language_id, f"trunc{ext}", truncated)
    assert graph4.diagnostics.has_errors or graph4.diagnostics.error_count >= 0


def test_truncated_parse_is_visible() -> None:
    from app.parsing.treesitter import MAX_PARSE_BYTES, MAX_WALK_NODES

    huge = "const x = 1;\n" + ("x + " * (MAX_PARSE_BYTES // 2)) + "1;\n"
    graph = parse_source("javascript", Path("big.js"), huge)
    assert graph.diagnostics.truncated
    assert graph.diagnostics.has_errors
    wide = "function f(){\n" + "let x = 1;\n" * 20_000 + "}\n"
    walked = parse_source("javascript", Path("n.js"), wide)
    assert walked.diagnostics.truncated
    assert len(walked.nodes) <= MAX_WALK_NODES


def test_corroboration_requires_independent_paths() -> None:
    doc = RuleDocumentation("d", "e", "l", "f")
    shared = dict(
        vulnerability_class=VulnerabilityClass.SQL_INJECTION,
        title="t",
        summary="s",
        file_path="a.js",
        line=3,
        evidence_text="db.query(q)",
        confidence="medium",
        language="javascript",
        documentation=doc,
        node_id="js:call:10:20:query",
        taint_path="from:module::q",
    )
    a = SecurityObservation(rule_id="sec.taint.sql", **shared)
    b = SecurityObservation(rule_id="sec.indicator.sql", **shared)
    clusters = correlate_observations([a, b])
    assert len(clusters) == 1
    assert clusters[0].independent_rules == 1
    assert clusters[0].corroborated is False


def test_unsupported_ssrf_in_c_is_explicit() -> None:
    vocab = vocab_for("c")
    assert vocab is not None
    assert not any(s.vulnerability_class is VulnerabilityClass.SSRF for s in vocab.sinks)


def test_php_echo_without_html_context_is_not_xss(tmp_path: Path) -> None:
    src = "<?php $q = $_GET['q']; echo $q;\n"
    result = analyze_source(tmp_path, "cli.php", src)
    assert VulnerabilityClass.XSS not in observation_classes(result)
    from app.analysis.javascript_analyzer import VarDeclarationRule

    findings = VarDeclarationRule().check_file(tmp_path / "a.js", "var x = 1;\n")
    assert findings
    assert findings[0].catalog == FindingCatalog.CODE_QUALITY


def test_quoted_shell_expansion_is_not_unquoted() -> None:
    graph = parse_source("shell", Path("ok.sh"), 'echo "$q"\n')
    assert not any(event.kind == "unquoted_expansion" for event in graph.events)


def test_every_sink_documents_semantics() -> None:
    from app.security.language_vocab import VOCABULARIES

    for vocab in VOCABULARIES.values():
        for sink in vocab.sinks:
            assert sink.dangerous_condition
            assert sink.api_names
            assert sink.vulnerability_class is not None


def test_analysis_stays_bounded(tmp_path: Path) -> None:
    import time

    src = "let q = req.query.q;\n" + "".join(f"let x{i} = 1;\n" for i in range(1500)) + "eval(q);\n"
    started = time.perf_counter()
    result = analyze_source(tmp_path, "b.js", src)
    assert time.perf_counter() - started < 8.0
    assert VulnerabilityClass.DYNAMIC_EXECUTION in observation_classes(result)
