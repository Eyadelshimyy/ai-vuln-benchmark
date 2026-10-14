"""JavaScript parser and security-analysis depth tests."""

from __future__ import annotations

from pathlib import Path

from app.adapters.languages import JavaScriptAdapter
from app.domain.language import LanguageCapability
from app.domain.security import VulnerabilityClass
from app.parsing.engine import parse_source
from app.parsing.model import SyntaxGraph
from tests.language_support import analyze_source, observation_classes


def _graph(source: str, name: str = "app.js") -> SyntaxGraph:
    return parse_source("javascript", Path(name), source)


def _names(graph: SyntaxGraph, attr: str) -> set[str]:
    items = getattr(graph, attr)
    if attr == "imports":
        return {imp.module for imp in items}
    if attr == "entities":
        return {ent.name for ent in items}
    if attr == "bindings":
        return {b.name for b in items}
    return {c.name for c in items}


class TestJavaScriptDetection:
    def test_extensions_and_capabilities(self) -> None:
        adapter = JavaScriptAdapter()
        for ext in (".js", ".mjs", ".cjs", ".jsx"):
            assert adapter.matches_path(Path(f"x{ext}"))
        assert adapter.supports(LanguageCapability.PARSE)
        assert adapter.supports(LanguageCapability.SECURITY_ANALYSIS)
        assert adapter.supports(LanguageCapability.STATIC_ANALYSIS)


class TestJavaScriptImports:
    def test_es_named_default_namespace(self) -> None:
        src = """
import defVal from 'mod-default';
import { named as alias } from 'mod-named';
import * as ns from 'mod-ns';
"""
        graph = _graph(src)
        mods = _names(graph, "imports")
        assert "mod-default" in mods
        assert "mod-named" in mods
        assert "mod-ns" in mods

    def test_commonjs_and_dynamic(self) -> None:
        src = "const fs = require('fs');\nconst m = import('lazy');\n"
        graph = _graph(src)
        mods = _names(graph, "imports")
        assert "fs" in mods
        assert "lazy" in mods

    def test_multiline_import(self) -> None:
        src = "import {\n  readFile,\n  writeFile as wf\n} from 'fs/promises';\n"
        graph = _graph(src)
        assert any("fs/promises" in m for m in _names(graph, "imports"))


class TestJavaScriptFunctionsAndClasses:
    def test_function_forms(self) -> None:
        src = """
function classic() {}
async function fetchIt() {}
const arrow = (x) => x;
const assigned = function named() {};
const asyncArrow = async (x) => x;
function* gen() { yield 1; }
"""
        graph = _graph(src)
        names = _names(graph, "entities")
        assert {"classic", "fetchIt", "arrow", "assigned", "asyncArrow", "gen"} <= names

    def test_nested_and_callback(self) -> None:
        src = """
function outer() {
  function inner() {}
  items.map(function mapper(x) { return x; });
}
"""
        graph = _graph(src)
        names = _names(graph, "entities")
        assert "outer" in names
        assert "inner" in names

    def test_class_methods_constructor_extends(self) -> None:
        src = """
class Animal {}
class Dog extends Animal {
  constructor(name) { this.name = name; }
  bark() { return 1; }
  static create() { return new Dog(); }
}
"""
        graph = _graph(src)
        names = _names(graph, "entities")
        assert "Animal" in names
        assert "Dog" in names
        assert "bark" in names or "create" in names or "constructor" in names


class TestJavaScriptBindingsAndCalls:
    def test_const_let_var_and_destructure(self) -> None:
        src = """
const a = 1;
let b = 2;
var c = 3;
const { query, body: payload } = req;
const [first, second] = items;
const { nested: { deep } } = obj;
"""
        graph = _graph(src)
        names = _names(graph, "bindings")
        assert {"a", "b", "c", "query", "payload", "first", "second"} <= names

    def test_calls_chained_qualified_multiline(self) -> None:
        src = """
run();
db.query(sql);
obj.method(x);
obj.child.method(y);
outer(inner(x));
chained
  .then((v) => v)
  .catch(err);
foo(
  bar(
    "hello(world)"
  )
);
obj?.maybe(z);
"""
        graph = _graph(src)
        names = _names(graph, "calls")
        for expected in (
            "run",
            "query",
            "method",
            "outer",
            "inner",
            "then",
            "catch",
            "foo",
            "bar",
            "maybe",
        ):
            assert expected in names, expected
        assert "hello" not in names

    def test_template_literal_args(self) -> None:
        src = "db.query(`SELECT ${user}`);\n"
        graph = _graph(src)
        query = next(c for c in graph.calls if c.name == "query")
        assert query.dynamic is True


class TestJavaScriptFalsePositives:
    def test_string_and_comment_lookalikes(self) -> None:
        src = """
const a = "eval(userInput)";
const b = 'exec(cmd)';
const c = `not a call eval(x) outside interp`;
// eval(userInput)
/* exec(command) */
const evalName = 1;
"""
        graph = _graph(src)
        names = _names(graph, "calls")
        assert "eval" not in names
        assert "exec" not in names

    def test_jsx_does_not_drop_real_calls(self) -> None:
        src = """
export function App({ html }) {
  const value = req.query.q;
  return <div dangerouslySetInnerHTML={{__html: value}} />;
}
"""
        graph = _graph(src, "app.jsx")
        names = _names(graph, "entities")
        assert "App" in names
        call_names = _names(graph, "calls")
        assert "dangerouslySetInnerHTML" in call_names


class TestJavaScriptSecurity:
    def test_source_to_sink_sql_and_command(self, tmp_path: Path) -> None:
        src = """
const express = require('express');
app.get('/x', (req, res) => {
  const q = req.query.q;
  db.query("SELECT * FROM t WHERE n = '" + q + "'");
  exec(req.query.cmd);
});
"""
        result = analyze_source(tmp_path, "v.js", src)
        classes = observation_classes(result)
        assert VulnerabilityClass.SQL_INJECTION in classes
        assert VulnerabilityClass.COMMAND_INJECTION in classes

    def test_parameterized_sql_is_quiet(self, tmp_path: Path) -> None:
        src = """
app.get('/x', (req, res) => {
  const q = req.query.q;
  db.query("SELECT * FROM t WHERE n = $1", [q]);
});
"""
        result = analyze_source(tmp_path, "s.js", src)
        assert VulnerabilityClass.SQL_INJECTION not in observation_classes(result)

    def test_path_ssrf_xss_eval_redirect_deser(self, tmp_path: Path) -> None:
        src = """
app.get('/all', (req, res) => {
  const q = req.query.q;
  fs.readFile(q);
  fetch(q);
  el.innerHTML = q;
  eval(q);
  res.redirect(q);
  JSON.parse(q);
});
"""
        result = analyze_source(tmp_path, "all.js", src)
        classes = observation_classes(result)
        assert VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL in classes
        assert VulnerabilityClass.SSRF in classes
        assert VulnerabilityClass.XSS in classes
        assert VulnerabilityClass.DYNAMIC_EXECUTION in classes
        assert VulnerabilityClass.UNSAFE_REDIRECT in classes
        assert VulnerabilityClass.POTENTIAL_UNSAFE_DESERIALIZATION in classes

    def test_weak_crypto(self, tmp_path: Path) -> None:
        src = "const hash = crypto.createHash('md5').update(password).digest('hex');\n"
        result = analyze_source(tmp_path, "c.js", src)
        assert VulnerabilityClass.WEAK_CRYPTOGRAPHY in observation_classes(result)

    def test_express_next_react_sources(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(
            '{"dependencies": {"express": "4.18.0", "next": "14.0.0", "react": "18.0.0"}}'
        )
        src = """
export default function Page({ searchParams }) {
  const q = searchParams.q;
  db.query("SELECT * FROM t WHERE x = '" + q + "'");
}
"""
        result = analyze_source(tmp_path, "page.jsx", src)
        assert VulnerabilityClass.SQL_INJECTION in observation_classes(result)

    def test_binding_flow_not_substring(self, tmp_path: Path) -> None:
        src = """
app.get('/x', (req, res) => {
  const q = req.query.q;
  const wrapped = "prefix-" + q;
  db.query("SELECT " + wrapped);
});
"""
        result = analyze_source(tmp_path, "flow.js", src)
        assert VulnerabilityClass.SQL_INJECTION in observation_classes(result)

    def test_constant_eval_is_quiet(self, tmp_path: Path) -> None:
        src = 'eval("1+1");\n'
        result = analyze_source(tmp_path, "ok.js", src)
        assert VulnerabilityClass.DYNAMIC_EXECUTION not in observation_classes(result)
