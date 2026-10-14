from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.adapters.languages.registry import LanguageRegistry
from app.analyzers.framework_detector import FrameworkDetector, FrameworkInfo
from app.core.config import settings
from app.domain.language import LanguageCapability, LanguageStats
from app.domain.source import LanguageParseResult

logger = logging.getLogger(__name__)

# Directories to skip entirely during traversal
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".bzr",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        ".tox",
        "node_modules",
        ".npm",
        ".yarn",
        ".venv",
        "venv",
        "env",
        ".cache",
        "dist",
        "build",
        "target",
        "out",
        ".next",
        ".nuxt",
        "coverage",
        "htmlcov",
        ".idea",
        ".vscode",
    }
)

# File extensions that are never interesting to parse
_SKIP_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".pyd",
        ".class",
        ".jar",
        ".war",
        ".o",
        ".a",
        ".so",
        ".dll",
        ".exe",
        ".bin",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".webp",
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".whl",
        ".egg",
    }
)

# Name patterns that indicate test files
_TEST_FILENAME_PREFIXES: tuple[str, ...] = ("test_",)
_TEST_FILENAME_SUFFIXES: tuple[str, ...] = ("_test.py", "_spec.py")
_TEST_DIR_NAMES: frozenset[str] = frozenset({"tests", "test", "spec", "__tests__"})

# Names that indicate config / infrastructure files
_CONFIG_NAMES: frozenset[str] = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "Pipfile",
        "Pipfile.lock",
        "poetry.lock",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "tsconfig.json",
        "webpack.config.js",
        "babel.config.js",
        ".babelrc",
        ".eslintrc.js",
        ".eslintrc.json",
        "next.config.js",
        "tailwind.config.js",
        "Makefile",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".env.example",
        "alembic.ini",
        "pytest.ini",
        "conftest.py",
        "tox.ini",
        "mypy.ini",
        ".pre-commit-config.yaml",
        "MANIFEST.in",
    }
)


@dataclass
class FileAnalysisResult:
    relative_path: str
    absolute_path: Path
    language: str | None
    file_type: str  # source | test | config | other
    size_bytes: int
    line_count: int
    has_parse_errors: bool = False
    parse_result: LanguageParseResult | None = None


@dataclass
class AnalysisResult:
    repository_path: str
    file_results: list[FileAnalysisResult] = field(default_factory=list)
    language_stats: list[LanguageStats] = field(default_factory=list)
    framework_detections: list[FrameworkInfo] = field(default_factory=list)
    total_files: int = 0
    source_file_count: int = 0
    test_file_count: int = 0
    config_file_count: int = 0
    ignored_file_count: int = 0


class RepoAnalyzer:
    """Walks and analyses a local repository — synchronous, safe to run in a thread."""

    def __init__(self) -> None:
        self._framework_detector = FrameworkDetector()

    def analyze(self, repo_path: Path) -> AnalysisResult:
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        if not repo_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {repo_path}")

        logger.info("Analyzing repository at %s", repo_path)

        from app.plugins import get_plugin_catalog

        languages = get_plugin_catalog().languages

        # 1. Walk the filesystem
        raw_files = self._walk(repo_path)

        # 2. Classify each file
        file_results = [self._classify_file(repo_path, p, languages) for p in raw_files]

        # 3. Language statistics (over all discovered files)
        language_stats = languages.detect_languages([f.absolute_path for f in file_results])

        # 4. Per-language repository context (e.g. Python local packages)
        parse_contexts: dict[str, object | None] = {}
        for parser in languages.parsers():
            parse_contexts[parser.language_id] = parser.prepare_repository(repo_path)

        # 5. Parse files whose language adapter implements PARSE
        for fr in file_results:
            adapter = languages.for_path(fr.absolute_path)
            if adapter is None or not adapter.supports(LanguageCapability.PARSE):
                continue
            if fr.size_bytes > settings.max_file_size_bytes:
                logger.debug(
                    "Skipping oversized file (%d bytes): %s",
                    fr.size_bytes,
                    fr.relative_path,
                )
                fr.has_parse_errors = True
                continue
            try:
                pr = adapter.parse_file(
                    fr.absolute_path, context=parse_contexts.get(adapter.language_id)
                )
                fr.parse_result = pr
                fr.line_count = pr.line_count
                fr.has_parse_errors = bool(pr.errors)
                if pr.errors:
                    logger.debug("Parse errors in %s: %s", fr.relative_path, pr.errors)
            except (OSError, PermissionError) as exc:
                logger.warning("Cannot read %s: %s", fr.relative_path, exc)
                fr.has_parse_errors = True
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", fr.relative_path, exc)
                fr.has_parse_errors = True

        # 6. Framework detection
        framework_detections = self._framework_detector.detect(
            repo_path, [f.absolute_path for f in file_results]
        )

        result = AnalysisResult(
            repository_path=str(repo_path),
            file_results=file_results,
            language_stats=language_stats,
            framework_detections=framework_detections,
            total_files=len(file_results),
            source_file_count=sum(1 for f in file_results if f.file_type == "source"),
            test_file_count=sum(1 for f in file_results if f.file_type == "test"),
            config_file_count=sum(1 for f in file_results if f.file_type == "config"),
        )
        logger.info(
            "Analysis complete: %d files, %d source, %d test",
            result.total_files,
            result.source_file_count,
            result.test_file_count,
        )
        return result

    def _walk(self, repo_path: Path) -> list[Path]:
        files: list[Path] = []
        repo_root_resolved = repo_path.resolve()

        for item in repo_path.rglob("*"):
            if item.is_dir():
                continue

            # Guard against symlinks escaping the repository root
            try:
                resolved = item.resolve()
                if not resolved.is_relative_to(repo_root_resolved):
                    logger.debug("Skipping symlink outside repo root: %s", item)
                    continue
            except OSError:
                continue

            try:
                rel = item.relative_to(repo_path)
            except ValueError:
                continue

            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            if item.suffix.lower() in _SKIP_EXTENSIONS:
                continue

            # Enforce max file count early to avoid iterating millions of files
            if len(files) >= settings.max_repo_files:
                logger.warning(
                    "Repository exceeds max file count (%d); stopping walk",
                    settings.max_repo_files,
                )
                break

            files.append(item)
        return files

    def _classify_file(
        self, repo_root: Path, file_path: Path, languages: LanguageRegistry
    ) -> FileAnalysisResult:
        rel = file_path.relative_to(repo_root)
        rel_str = str(rel)
        size = 0
        try:
            size = file_path.stat().st_size
        except (OSError, PermissionError):
            pass

        adapter = languages.for_path(file_path)
        language = adapter.language_id if adapter is not None else None
        file_type = self._determine_file_type(rel, languages)

        return FileAnalysisResult(
            relative_path=rel_str,
            absolute_path=file_path,
            language=language,
            file_type=file_type,
            size_bytes=size,
            line_count=0,  # filled in after parsing
        )

    @staticmethod
    def _determine_file_type(rel: Path, languages: LanguageRegistry) -> str:
        name = rel.name
        parts = rel.parts

        # Config files by name
        if name in _CONFIG_NAMES:
            return "config"

        # Test detection: name pattern
        if name.startswith(_TEST_FILENAME_PREFIXES) or name.endswith(_TEST_FILENAME_SUFFIXES):
            return "test"

        # Test detection: parent directory
        for part in parts[:-1]:
            if part in _TEST_DIR_NAMES:
                return "test"

        adapter = languages.for_path(rel)
        if adapter is not None and adapter.supports(LanguageCapability.SOURCE):
            return "source"

        return "other"
