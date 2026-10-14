from __future__ import annotations

from pathlib import Path

from .parser import ParseResult, PythonParser


class PythonLanguageAnalyzer:
    """Parses Python files and enriches import classification with project-local packages."""

    def __init__(self) -> None:
        self._parser = PythonParser()

    def analyze_file(
        self, file_path: Path, local_packages: frozenset[str] | None = None
    ) -> ParseResult:
        result = self._parser.parse_file(file_path)
        if local_packages:
            for imp in result.imports:
                if imp.import_type == "third_party":
                    top = imp.module.split(".")[0]
                    if top in local_packages:
                        imp.import_type = "local"
        return result

    @staticmethod
    def discover_local_packages(repo_root: Path) -> frozenset[str]:
        """Return the set of top-level package names defined in this repository."""
        packages: set[str] = set()
        for init_file in repo_root.rglob("__init__.py"):
            package_dir = init_file.parent
            # Only count directories that are direct children of repo_root or a sub-package
            if package_dir != repo_root:
                packages.add(package_dir.relative_to(repo_root).parts[0])
        return frozenset(packages)
