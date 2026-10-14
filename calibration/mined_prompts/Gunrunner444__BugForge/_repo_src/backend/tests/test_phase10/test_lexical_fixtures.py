"""Phase 10 string/comment/template lexical fixtures."""

from __future__ import annotations

from pathlib import Path

from app.parsing.engine import parse_source


def _calls(language: str, filename: str, source: str) -> set[str]:
    graph = parse_source(language, Path(filename), source)
    return {c.name for c in graph.calls} | {c.qualified for c in graph.calls}


def test_cpp_raw_string_is_data() -> None:
    src = 'const char* s = R"(system(q))";\n'
    names = _calls("cpp", "r.cpp", src)
    assert "system" not in names


def test_cpp_char_literal_is_data() -> None:
    src = "char c = 'x';\n"
    names = _calls("cpp", "c.cpp", src)
    assert "x" not in names


def test_java_text_block_is_data() -> None:
    src = 'class T { String s = """\neval(user)\n"""; }\n'
    names = _calls("java", "T.java", src)
    assert "eval" not in names


def test_swift_multiline_string_is_data() -> None:
    src = 'let s = """\neval(user)\n"""\n'
    names = _calls("swift", "s.swift", src)
    assert "eval" not in names


def test_ruby_heredoc_is_data() -> None:
    src = "q = <<~SQL\n  SELECT 1\nSQL\n"
    names = _calls("ruby", "h.rb", src)
    assert "SELECT" not in names


def test_php_backticks_are_code() -> None:
    src = "<?php echo `ls`;\n"
    names = _calls("php", "b.php", src)
    assert names


def test_js_template_interpolation_is_code() -> None:
    src = "const s = `hi ${eval(user)}`;\n"
    names = _calls("javascript", "t.js", src)
    assert "eval" in names


def test_ts_template_interpolation_is_code() -> None:
    src = "const s: string = `hi ${eval(user)}`;\n"
    names = _calls("typescript", "t.ts", src)
    assert "eval" in names
