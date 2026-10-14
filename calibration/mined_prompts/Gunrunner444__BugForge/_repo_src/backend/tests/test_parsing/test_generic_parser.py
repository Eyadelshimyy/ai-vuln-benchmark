"""Critical tests for the generic profile parser."""

from __future__ import annotations

from pathlib import Path

from app.parsing.engine import parse_source
from tests.language_corpus import PARSER_CORPUS


def _call_names(graph) -> set[str]:
    return {c.name for c in graph.calls} | {c.qualified.split(".")[-1] for c in graph.calls}


def test_corpus_parser_expectations() -> None:
    for case in PARSER_CORPUS:
        graph = parse_source(case.language, Path(case.filename), case.source)
        names = _call_names(graph)
        for expected in case.expect_calls:
            assert expected in names, f"{case.name}: missing call {expected} in {names}"
        for unexpected in case.unexpected_calls:
            assert unexpected not in names, f"{case.name}: false call {unexpected} in {names}"
        binding_names = {b.name for b in graph.bindings}
        for expected in case.expect_bindings:
            assert expected in binding_names, f"{case.name}: missing binding {expected}"
        entity_names = {e.name for e in graph.entities}
        for expected in case.expect_entities:
            assert expected in entity_names, f"{case.name}: missing entity {expected}"
        import_mods = {imp.module for imp in graph.imports}
        for expected in case.expect_imports:
            assert any(expected in mod for mod in import_mods), (
                f"{case.name}: missing import {expected} in {import_mods}"
            )


def test_empty_source_does_not_crash() -> None:
    graph = parse_source("javascript", Path("empty.js"), "")
    assert graph.language == "javascript"
    assert graph.calls == ()
    assert graph.lines == ("",)


def test_malformed_unclosed_string_does_not_crash() -> None:
    graph = parse_source("javascript", Path("bad.js"), 'const x = "oops\nexec(y)\n')
    assert graph.language == "javascript"


def test_malformed_unclosed_paren_does_not_crash() -> None:
    graph = parse_source("javascript", Path("bad.js"), "run(foo(bar\n")
    names = {c.name for c in graph.calls}
    assert "run" in names or "foo" in names


def test_declarations_sharing_a_line() -> None:
    graph = parse_source("javascript", Path("m.js"), "const a = 1; const b = req.query; run(b);\n")
    names = {b.name for b in graph.bindings}
    assert "a" in names
    assert "b" in names


def test_unusual_whitespace_call() -> None:
    graph = parse_source("javascript", Path("w.js"), "obj  .\n  method  (\n  user  )\n")
    names = {c.name for c in graph.calls}
    assert "method" in names


def test_callbacks_and_function_expressions() -> None:
    src = "items.forEach(function handler(item) { save(item); });\n"
    graph = parse_source("javascript", Path("cb.js"), src)
    names = {c.name for c in graph.calls}
    assert "forEach" in names
    assert "save" in names
    assert "function" not in names


def test_generator_function_entity() -> None:
    graph = parse_source("javascript", Path("g.js"), "function* gen() { yield 1; }\n")
    assert any(e.name == "gen" for e in graph.entities)


def test_template_with_nested_call_and_string() -> None:
    src = "db.query(`select ${filter('a(b)')}`);\n"
    graph = parse_source("javascript", Path("t.js"), src)
    names = {c.name for c in graph.calls}
    assert "query" in names
    assert "filter" in names
