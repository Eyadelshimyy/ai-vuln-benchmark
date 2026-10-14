from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.analyzers.repo_analyzer import RepoAnalyzer


def _write(directory: Path, name: str, content: str) -> Path:
    (directory / name).parent.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_text(textwrap.dedent(content))
    return p


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Creates a minimal Python repository for testing."""
    src = tmp_path / "src"
    src.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()

    _write(
        tmp_path,
        "src/__init__.py",
        "",
    )
    _write(
        tmp_path,
        "src/calculator.py",
        """
        class Calculator:
            def add(self, a: int, b: int) -> int:
                return a + b

            def divide(self, a: float, b: float) -> float:
                return a / b
        """,
    )
    _write(
        tmp_path,
        "src/utils.py",
        """
        import os
        from pathlib import Path

        def get_cwd() -> Path:
            return Path(os.getcwd())
        """,
    )
    _write(
        tmp_path,
        "tests/__init__.py",
        "",
    )
    _write(
        tmp_path,
        "tests/test_calculator.py",
        """
        import pytest
        from src.calculator import Calculator

        def test_add():
            calc = Calculator()
            assert calc.add(2, 3) == 5
        """,
    )
    _write(tmp_path, "pyproject.toml", "[project]\nname = 'sample'\n")
    _write(tmp_path, "README.md", "# Sample\n")
    return tmp_path


class TestRepoAnalyzer:
    def test_analyze_returns_result(self, sample_repo: Path) -> None:
        result = RepoAnalyzer().analyze(sample_repo)
        assert result.repository_path == str(sample_repo)

    def test_detects_python_language(self, sample_repo: Path) -> None:
        result = RepoAnalyzer().analyze(sample_repo)
        langs = {s.language for s in result.language_stats}
        assert "python" in langs

    def test_counts_files(self, sample_repo: Path) -> None:
        result = RepoAnalyzer().analyze(sample_repo)
        assert result.total_files > 0
        assert result.source_file_count >= 2  # calculator.py, utils.py at minimum
        assert result.test_file_count >= 1

    def test_identifies_test_files(self, sample_repo: Path) -> None:
        result = RepoAnalyzer().analyze(sample_repo)
        test_paths = [f.relative_path for f in result.file_results if f.file_type == "test"]
        assert any("test_calculator" in p for p in test_paths)

    def test_identifies_config_files(self, sample_repo: Path) -> None:
        result = RepoAnalyzer().analyze(sample_repo)
        config_paths = [f.relative_path for f in result.file_results if f.file_type == "config"]
        assert any("pyproject.toml" in p for p in config_paths)

    def test_parses_python_entities(self, sample_repo: Path) -> None:
        result = RepoAnalyzer().analyze(sample_repo)
        all_entities = [
            e for fr in result.file_results if fr.parse_result for e in fr.parse_result.entities
        ]
        entity_names = {e.name for e in all_entities}
        assert "Calculator" in entity_names
        assert "add" in entity_names
        assert "divide" in entity_names

    def test_parses_imports(self, sample_repo: Path) -> None:
        result = RepoAnalyzer().analyze(sample_repo)
        all_imports = [
            i for fr in result.file_results if fr.parse_result for i in fr.parse_result.imports
        ]
        modules = {i.module for i in all_imports}
        assert "os" in modules
        assert "pathlib" in modules

    def test_import_classification(self, sample_repo: Path) -> None:
        result = RepoAnalyzer().analyze(sample_repo)
        all_imports = [
            i for fr in result.file_results if fr.parse_result for i in fr.parse_result.imports
        ]
        by_module = {i.module: i for i in all_imports}
        assert by_module["os"].import_type == "stdlib"
        assert by_module["pathlib"].import_type == "stdlib"

    def test_nonexistent_path_raises(self) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            RepoAnalyzer().analyze(Path("/nonexistent/path/12345"))

    def test_file_path_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="not a directory"):
            RepoAnalyzer().analyze(f)

    def test_skips_pycache(self, sample_repo: Path) -> None:
        cache_dir = sample_repo / "src" / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "calculator.cpython-312.pyc").write_bytes(b"")
        result = RepoAnalyzer().analyze(sample_repo)
        paths = [f.relative_path for f in result.file_results]
        assert not any("__pycache__" in p for p in paths)

    def test_line_counts_populated(self, sample_repo: Path) -> None:
        result = RepoAnalyzer().analyze(sample_repo)
        python_files = [
            f for f in result.file_results if f.language == "python" and f.file_type != "config"
        ]
        assert all(f.line_count > 0 for f in python_files)
