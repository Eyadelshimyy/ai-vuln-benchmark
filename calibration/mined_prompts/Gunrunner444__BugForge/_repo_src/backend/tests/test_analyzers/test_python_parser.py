from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.analyzers.python.parser import PythonParser


@pytest.fixture
def parser() -> PythonParser:
    return PythonParser()


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


class TestImportExtraction:
    def test_bare_import(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(tmp_path, "a.py", "import os\nimport sys\n")
        result = parser.parse_file(f)
        modules = [i.module for i in result.imports]
        assert "os" in modules
        assert "sys" in modules

    def test_from_import(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(tmp_path, "a.py", "from pathlib import Path\n")
        result = parser.parse_file(f)
        assert len(result.imports) == 1
        imp = result.imports[0]
        assert imp.module == "pathlib"
        assert imp.name == "Path"
        assert imp.is_from_import is True
        assert imp.import_type == "stdlib"

    def test_third_party_import(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(tmp_path, "a.py", "import fastapi\nfrom pydantic import BaseModel\n")
        result = parser.parse_file(f)
        types = {i.module: i.import_type for i in result.imports}
        assert types["fastapi"] == "third_party"
        assert types["pydantic"] == "third_party"

    def test_relative_import(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(tmp_path, "a.py", "from . import utils\nfrom ..core import config\n")
        result = parser.parse_file(f)
        assert all(i.import_type == "relative" for i in result.imports)

    def test_import_alias(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(tmp_path, "a.py", "import numpy as np\n")
        result = parser.parse_file(f)
        assert result.imports[0].alias == "np"


class TestEntityExtraction:
    def test_top_level_function(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(
            tmp_path,
            "a.py",
            """
            def add(a: int, b: int) -> int:
                return a + b
            """,
        )
        result = parser.parse_file(f)
        funcs = [e for e in result.entities if e.name == "add"]
        assert len(funcs) == 1
        fn = funcs[0]
        assert fn.entity_type == "function"
        assert fn.return_annotation == "int"
        assert len(fn.parameters) == 2
        assert fn.parameters[0].name == "a"
        assert fn.parameters[0].annotation == "int"

    def test_async_function(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(tmp_path, "a.py", "async def fetch(): pass\n")
        result = parser.parse_file(f)
        assert result.entities[0].entity_type == "async_function"

    def test_class_with_methods(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(
            tmp_path,
            "a.py",
            """
            class Calc:
                def add(self, x, y):
                    return x + y
                async def async_add(self, x, y):
                    return x + y
            """,
        )
        result = parser.parse_file(f)
        entity_types = {e.name: e.entity_type for e in result.entities}
        assert entity_types["Calc"] == "class"
        assert entity_types["add"] == "method"
        assert entity_types["async_add"] == "async_method"

    def test_method_qualified_name(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(
            tmp_path,
            "a.py",
            """
            class MyClass:
                def my_method(self): pass
            """,
        )
        result = parser.parse_file(f)
        method = next(e for e in result.entities if e.name == "my_method")
        assert method.qualified_name == "MyClass.my_method"
        assert method.parent == "MyClass"

    def test_docstring_extraction(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(
            tmp_path,
            "a.py",
            '''
            def greet(name: str) -> str:
                """Say hello."""
                return f"Hello {name}"
            ''',
        )
        result = parser.parse_file(f)
        fn = result.entities[0]
        assert fn.docstring == "Say hello."

    def test_decorator_extraction(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(
            tmp_path,
            "a.py",
            """
            @property
            @staticmethod
            def decorated(): pass
            """,
        )
        result = parser.parse_file(f)
        assert "property" in result.entities[0].decorators
        assert "staticmethod" in result.entities[0].decorators

    def test_default_parameter(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(tmp_path, "a.py", "def foo(x, y=10, *args, **kwargs): pass\n")
        result = parser.parse_file(f)
        params = {p.name: p for p in result.entities[0].parameters}
        assert params["y"].default == "10"
        assert params["args"].kind == "var_positional"
        assert params["kwargs"].kind == "var_keyword"

    def test_line_numbers(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(
            tmp_path,
            "a.py",
            "def first(): pass\n\ndef second(): pass\n",
        )
        result = parser.parse_file(f)
        names_to_lines = {e.name: e.start_line for e in result.entities}
        assert names_to_lines["first"] == 1
        assert names_to_lines["second"] == 3


class TestParserErrors:
    def test_syntax_error_recorded(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(tmp_path, "a.py", "def broken(\n")
        result = parser.parse_file(f)
        assert result.errors
        assert "SyntaxError" in result.errors[0]

    def test_missing_file(self, parser: PythonParser, tmp_path: Path) -> None:
        result = parser.parse_file(tmp_path / "nonexistent.py")
        assert result.errors
        assert "Cannot read file" in result.errors[0]

    def test_line_count(self, parser: PythonParser, tmp_path: Path) -> None:
        f = _write(tmp_path, "a.py", "x = 1\ny = 2\nz = 3\n")
        result = parser.parse_file(f)
        assert result.line_count == 3
