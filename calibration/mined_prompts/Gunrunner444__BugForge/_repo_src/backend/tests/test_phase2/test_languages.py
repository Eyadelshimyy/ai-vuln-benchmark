"""Phase 2 language adapters, parsers, and framework detection."""

from __future__ import annotations

from pathlib import Path

from app.adapters.languages import JavaScriptAdapter, PythonAdapter, RubyAdapter, TypeScriptAdapter
from app.adapters.languages.known import CAdapter, CppAdapter
from app.analyzers.framework_detector import FrameworkDetector
from app.domain.language import LanguageCapability
from app.parsing.engine import parse_source
from app.plugins import get_plugin_catalog, reset_plugin_catalog


def setup_function() -> None:
    reset_plugin_catalog()


def teardown_function() -> None:
    reset_plugin_catalog()


def test_programming_languages_expose_security_capabilities() -> None:
    catalog = get_plugin_catalog()
    for language_id in (
        "python",
        "javascript",
        "typescript",
        "ruby",
        "c",
        "cpp",
        "go",
        "rust",
        "java",
        "php",
        "kotlin",
        "swift",
    ):
        adapter = catalog.languages.get(language_id)
        assert adapter.supports(LanguageCapability.DETECTION)
        assert adapter.supports(LanguageCapability.PARSE)
        assert adapter.supports(LanguageCapability.ENTITY_EXTRACTION)
        assert adapter.supports(LanguageCapability.IMPORT_EXTRACTION)
        assert adapter.supports(LanguageCapability.SECURITY_ANALYSIS)


def test_javascript_parse_entities_and_imports(tmp_path: Path) -> None:
    path = tmp_path / "app.js"
    path.write_text("const express = require('express');\nfunction hello() { return 1; }\n")
    result = JavaScriptAdapter().parse_file(path)
    assert result.language == "javascript"
    assert any("express" in (imp.module or "") for imp in result.imports)
    assert any(ent.name == "hello" for ent in result.entities)


def test_typescript_parse(tmp_path: Path) -> None:
    path = tmp_path / "app.ts"
    path.write_text("import axios from 'axios';\nexport function go() { return 1; }\n")
    result = TypeScriptAdapter().parse_file(path)
    assert result.language == "typescript"
    assert any(ent.name == "go" for ent in result.entities)


def test_ruby_parse(tmp_path: Path) -> None:
    path = tmp_path / "app.rb"
    path.write_text("require 'sinatra'\ndef hello\n  1\nend\n")
    result = RubyAdapter().parse_file(path)
    assert any(ent.name == "hello" for ent in result.entities)


def test_c_and_cpp_parse(tmp_path: Path) -> None:
    c_path = tmp_path / "a.c"
    c_path.write_text('#include <stdio.h>\nvoid run() {\n    puts("x");\n}\n')
    cpp_path = tmp_path / "a.cpp"
    cpp_path.write_text("#include <string>\nclass Box {};\n")
    c_result = CAdapter().parse_file(c_path)
    cpp_result = CppAdapter().parse_file(cpp_path)
    assert c_result.language == "c"
    assert any("stdio" in (imp.module or "") for imp in c_result.imports)
    assert cpp_result.language == "cpp"
    assert any(ent.name == "Box" for ent in cpp_result.entities)


def test_python_syntax_graph_from_ast(tmp_path: Path) -> None:
    path = tmp_path / "m.py"
    source = "import os\n\ndef hello(x):\n    y = os.getenv('A')\n    return y\n"
    path.write_text(source)
    graph = PythonAdapter().syntax_graph(path, source)
    assert graph.language == "python"
    assert any(b.name == "y" for b in graph.bindings)
    assert any(c.name in {"getenv", "hello"} or "getenv" in c.qualified for c in graph.calls)


def test_parse_source_dispatches_without_language_branches() -> None:
    graph = parse_source("javascript", Path("x.js"), "const x = 1;\n")
    assert graph.language == "javascript"


def test_framework_registry_detects_express_and_django(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        '{"dependencies": {"express": "4.18.0", "next": "14.0.0"}}'
    )
    (tmp_path / "manage.py").write_text("#!/usr/bin/env python\n")
    (tmp_path / "requirements.txt").write_text("django==5.0\n")
    names = {fw.name for fw in FrameworkDetector().detect(tmp_path, list(tmp_path.iterdir()))}
    assert "express" in names
    assert "next.js" in names
    assert "django" in names


def test_framework_registry_detects_rails_and_nestjs(tmp_path: Path) -> None:
    (tmp_path / "Gemfile").write_text('gem "rails"\n')
    (tmp_path / "package.json").write_text('{"dependencies": {"@nestjs/core": "10.0.0"}}')
    names = {fw.name for fw in FrameworkDetector().detect(tmp_path, list(tmp_path.iterdir()))}
    assert "rails" in names
    assert "nestjs" in names
