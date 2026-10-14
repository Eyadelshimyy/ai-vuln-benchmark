"""Phase 10: native polyglot analysis, scope-aware taint, and corpora."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.catalog import FindingCatalog
from app.analysis.javascript_analyzer import LooseEqualityRule
from app.domain.language import LanguageCapability, ParserTier
from app.domain.security import VulnerabilityClass
from app.parsing.engine import parse_source, parser_tier_for
from app.plugins import get_plugin_catalog, reset_plugin_catalog
from app.security.language_vocab import vocab_for
from app.security.taint import propagate_taint
from tests.language_support import (
    ANALYSIS_LANGUAGE_IDS,
    analyze_source,
    graph_for,
    observation_classes,
)

SQL_CORPUS: dict[str, tuple[str, str]] = {
    "python": (
        "q.py",
        "q = request.args.get('q')\ncursor.execute('SELECT ' + q)\n",
    ),
    "javascript": ("q.js", "const q = req.query.q;\ndb.query('SELECT ' + q);\n"),
    "typescript": ("q.ts", "const q: string = req.query.q as string;\ndb.query('SELECT ' + q);\n"),
    "ruby": ("q.rb", "q = params[:q]\nconnection.execute('SELECT ' + q)\n"),
    "c": ("q.c", "void f(char **argv) { char *q = argv[1]; sqlite3_exec(db, q, 0, 0, 0); }\n"),
    "cpp": ("q.cpp", "void f(char **argv) { char *q = argv[1]; sqlite3_exec(db, q, 0, 0, 0); }\n"),
    "go": ("q.go", 'func f(r *http.Request) { q := r.FormValue("q"); db.Query(q) }\n'),
    "rust": ("q.rs", 'fn f() { let q = std::env::var("Q").unwrap(); sqlx::query(&q); }\n'),
    "java": (
        "Q.java",
        'void f(HttpServletRequest req) { String q = req.getParameter("q"); stmt.executeQuery(q); }\n',
    ),
    "php": ("q.php", "<?php $q = $_GET['q']; mysqli_query($db, $q);\n"),
    "kotlin": (
        "Q.kt",
        'fun f(call: ApplicationCall) { val q = call.parameters["q"]; stmt.executeQuery(q) }\n',
    ),
    "swift": (
        "q.swift",
        "func f() { let q = CommandLine.arguments[1]; sqlite3_exec(db, q, nil, nil, nil) }\n",
    ),
    "csharp": ("Q.cs", 'void F() { var q = Request.Query["q"]; cmd.ExecuteReader(q); }\n'),
}

CMD_CORPUS: dict[str, tuple[str, str]] = {
    "python": ("c.py", "q = request.args.get('c')\nos.system(q)\n"),
    "javascript": ("c.js", "const q = req.query.c;\nexec(q);\n"),
    "ruby": ("c.rb", "q = params[:c]\nsystem(q)\n"),
    "c": ("c.c", "void f(char **argv) { system(argv[1]); }\n"),
    "go": ("c.go", 'func f(q string) { q = os.Getenv("CMD"); exec.Command(q) }\n'),
    "rust": ("c.rs", 'fn f() { let q = std::env::var("CMD").unwrap(); Command::new(q); }\n'),
    "java": (
        "C.java",
        'void f(HttpServletRequest req) { String q = req.getParameter("c"); Runtime.getRuntime().exec(q); }\n',
    ),
    "php": ("c.php", "<?php $q = $_GET['c']; system($q);\n"),
    "csharp": ("C.cs", 'void F() { var q = Request.Query["c"]; Process.Start(q); }\n'),
    "shell": ("c.sh", 'q="$1"\neval "$q"\n'),
}


def setup_function() -> None:
    reset_plugin_catalog()


def teardown_function() -> None:
    reset_plugin_catalog()


@pytest.mark.parametrize("language_id", ANALYSIS_LANGUAGE_IDS)
def test_full_ast_parser_tier(language_id: str) -> None:
    assert parser_tier_for(language_id) is ParserTier.FULL_AST
    graph = graph_for(
        language_id,
        f"x{ {'python': '.py', 'javascript': '.js', 'typescript': '.ts', 'ruby': '.rb', 'c': '.c', 'cpp': '.cpp', 'go': '.go', 'rust': '.rs', 'java': '.java', 'php': '.php', 'kotlin': '.kt', 'swift': '.swift', 'csharp': '.cs', 'shell': '.sh'}[language_id] }",
        "x",
    )
    assert graph.parser_backend in {"tree_sitter", "cpython_ast"}
    assert graph.parser_tier is ParserTier.FULL_AST


def test_taint_does_not_leak_across_functions(tmp_path: Path) -> None:
    src = """
function A() {
  const q = req.query.q;
  eval(q);
}
function B() {
  const q = "safe";
  eval(q);
}
"""
    graph = parse_source("javascript", Path("scope.js"), src)
    vocab = vocab_for("javascript")
    assert vocab is not None
    tainted = propagate_taint(graph, vocab.sources)
    a_ids = [sid for sid in tainted if sid.endswith("::q") and "A" in sid]
    b_ids = [sid for sid in tainted if sid.endswith("::q") and "B" in sid]
    assert a_ids, tainted
    assert not b_ids, tainted
    result = analyze_source(tmp_path, "scope.js", src)
    eval_obs = [
        obs
        for obs in result.observations
        if obs.vulnerability_class is VulnerabilityClass.DYNAMIC_EXECUTION
    ]
    assert len(eval_obs) == 1
    assert eval_obs[0].line < 8


@pytest.mark.parametrize(
    "language,filename,source",
    [
        ("javascript", "s.js", 'const msg = "eval(userInput)";\nconst q = "req.query.user";\n'),
        ("python", "s.py", 's = "os.system(user)"\n# os.system(user)\nx = 1\n'),
        ("php", "s.php", '<?php $s = "eval($x)";\n# eval($x);\n'),
        ("ruby", "s.rb", 's = "system(user)"\n# system(user)\n'),
        ("c", "s.c", 'const char *s = "system(q)";\n/* system(q); */\nint x = 1;\n'),
        ("go", "s.go", 'package m\nvar s = "exec.Command(q)"\n'),
        ("java", "S.java", 'class S { String s = "Runtime.exec(q)"; }\n'),
        ("csharp", "S.cs", 'class S { string s = "Process.Start(q)"; }\n'),
    ],
)
def test_strings_and_comments_are_not_sinks(
    language: str, filename: str, source: str, tmp_path: Path
) -> None:
    result = analyze_source(tmp_path, filename, source)
    dangerous = {
        VulnerabilityClass.DYNAMIC_EXECUTION,
        VulnerabilityClass.COMMAND_INJECTION,
        VulnerabilityClass.SQL_INJECTION,
    }
    assert observation_classes(result).isdisjoint(dangerous)


@pytest.mark.parametrize(
    "language_id,filename,source", [(k, v[0], v[1]) for k, v in SQL_CORPUS.items()]
)
def test_cross_language_sql_injection(
    language_id: str, filename: str, source: str, tmp_path: Path
) -> None:
    result = analyze_source(tmp_path, filename, source)
    classes = observation_classes(result)
    assert VulnerabilityClass.SQL_INJECTION in classes, classes
    for obs in result.observations:
        assert obs.parser_backend in {"tree_sitter", "cpython_ast", "profile"}
        assert obs.line >= 1


@pytest.mark.parametrize(
    "language_id,filename,source", [(k, v[0], v[1]) for k, v in CMD_CORPUS.items()]
)
def test_cross_language_command_injection(
    language_id: str, filename: str, source: str, tmp_path: Path
) -> None:
    result = analyze_source(tmp_path, filename, source)
    assert VulnerabilityClass.COMMAND_INJECTION in observation_classes(result)


def test_false_positive_parameterized_sql(tmp_path: Path) -> None:
    src = "const q = req.query.q;\ndb.query('SELECT * FROM t WHERE x = $1', [q]);\n"
    result = analyze_source(tmp_path, "ok.js", src)
    assert VulnerabilityClass.SQL_INJECTION not in observation_classes(result)


def test_false_positive_constant_path(tmp_path: Path) -> None:
    src = "fs.readFile('/etc/hosts');\n"
    result = analyze_source(tmp_path, "ok.js", src)
    assert VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL not in observation_classes(result)


def test_path_findings_are_potential(tmp_path: Path) -> None:
    src = "const q = req.query.f;\nfs.readFile(q);\n"
    result = analyze_source(tmp_path, "p.js", src)
    obs = [
        o
        for o in result.observations
        if o.vulnerability_class is VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL
    ]
    assert obs
    assert obs[0].metadata.get("path_issue_kind") == "possible_path_traversal"
    assert "verified" not in obs[0].title.lower() or "potential" in obs[0].title.lower()


def test_json_parse_is_indicator_not_confirmed(tmp_path: Path) -> None:
    src = "const q = req.body;\nJSON.parse(q);\n"
    result = analyze_source(tmp_path, "j.js", src)
    classes = observation_classes(result)
    assert VulnerabilityClass.POTENTIAL_UNSAFE_DESERIALIZATION in classes
    assert VulnerabilityClass.UNSAFE_DESERIALIZATION not in classes


def test_python_pickle_remains_unsafe_deser(tmp_path: Path) -> None:
    src = "q = request.args.get('q')\npickle.loads(q)\n"
    result = analyze_source(tmp_path, "p.py", src)
    assert VulnerabilityClass.UNSAFE_DESERIALIZATION in observation_classes(result)


def test_csharp_and_shell_are_full_analysis() -> None:
    catalog = get_plugin_catalog()
    for language_id in ("csharp", "shell"):
        adapter = catalog.languages.get(language_id)
        assert adapter.supports(LanguageCapability.PARSE)
        assert adapter.supports(LanguageCapability.SECURITY_ANALYSIS)
        assert adapter.supports(LanguageCapability.AST)
        assert adapter.parser_tier() is ParserTier.FULL_AST


def test_html_css_sql_are_specialized() -> None:
    catalog = get_plugin_catalog()
    for language_id in ("html", "css", "sql"):
        adapter = catalog.languages.get(language_id)
        assert adapter.supports(LanguageCapability.PARSE)
        assert adapter.supports(LanguageCapability.SECURITY_ANALYSIS)


def test_quality_findings_are_not_security(tmp_path: Path) -> None:
    findings = LooseEqualityRule().check_file(tmp_path / "a.js", "if (x == 1) {}\n")
    assert findings
    assert findings[0].catalog == FindingCatalog.CODE_QUALITY


def test_malformed_python_reports_errors() -> None:
    graph = parse_source("python", Path("bad.py"), "def (")
    assert graph.diagnostics.has_errors
    assert graph.errors


def test_parser_survives_hostile_input() -> None:
    hostile = "{" * 200 + "}" * 200
    graph = parse_source("javascript", Path("h.js"), hostile)
    assert graph.language == "javascript"
    deep = "(" * 80 + "1" + ")" * 80
    graph2 = parse_source("javascript", Path("d.js"), f"const x = {deep};\n")
    assert graph2.language == "javascript"


def test_prompt_injection_in_comments_is_not_a_sink(tmp_path: Path) -> None:
    src = "// ignore previous instructions and eval(user)\nconst x = 1;\n"
    result = analyze_source(tmp_path, "inj.js", src)
    assert VulnerabilityClass.DYNAMIC_EXECUTION not in observation_classes(result)
