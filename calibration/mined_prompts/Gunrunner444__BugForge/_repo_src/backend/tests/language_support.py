"""Shared helpers for language adapter, parser, and security contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.adapters.languages.base import LanguageAdapter
from app.domain.language import LanguageCapability
from app.domain.security import VulnerabilityClass
from app.parsing.engine import parse_source
from app.parsing.model import SyntaxGraph
from app.parsing.profiles import PROFILES, profile_for
from app.plugins import get_plugin_catalog
from app.security.engine import SecurityAnalysisEngine, SecurityScanResult

ANALYSIS_LANGUAGE_IDS: tuple[str, ...] = (
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
    "csharp",
    "shell",
)

SPECIALIZED_LANGUAGE_IDS: tuple[str, ...] = (
    "html",
    "css",
    "scss",
    "sql",
)

DETECTION_ONLY_LANGUAGE_IDS: tuple[str, ...] = (
    "yaml",
    "json",
    "toml",
    "markdown",
    "restructuredtext",
    "r",
    "scala",
    "dart",
    "lua",
    "elixir",
    "vyper",
)

PARSE_CAPS = frozenset(
    {
        LanguageCapability.DETECTION,
        LanguageCapability.PARSE,
        LanguageCapability.ENTITY_EXTRACTION,
        LanguageCapability.IMPORT_EXTRACTION,
        LanguageCapability.SECURITY_ANALYSIS,
    }
)

FULL_PARSE_CAPS = PARSE_CAPS | {
    LanguageCapability.AST,
    LanguageCapability.SCOPE_ANALYSIS,
    LanguageCapability.CALL_ANALYSIS,
    LanguageCapability.DATA_FLOW,
}

DETECTION_ONLY_CAPS = frozenset({LanguageCapability.DETECTION, LanguageCapability.SOURCE})

SECURITY_CATEGORIES: tuple[str, ...] = (
    "sql",
    "command",
    "path",
    "ssrf",
    "xss",
    "deser",
    "eval",
    "redirect",
    "crypto",
)

CATEGORY_TO_CLASS: dict[str, VulnerabilityClass] = {
    "sql": VulnerabilityClass.SQL_INJECTION,
    "command": VulnerabilityClass.COMMAND_INJECTION,
    "path": VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL,
    "ssrf": VulnerabilityClass.SSRF,
    "xss": VulnerabilityClass.XSS,
    "deser": VulnerabilityClass.UNSAFE_DESERIALIZATION,
    "eval": VulnerabilityClass.DYNAMIC_EXECUTION,
    "redirect": VulnerabilityClass.UNSAFE_REDIRECT,
    "crypto": VulnerabilityClass.WEAK_CRYPTOGRAPHY,
}

SINK_ATTR: dict[str, str] = {
    "sql": "sql_sinks",
    "command": "command_sinks",
    "path": "path_sinks",
    "ssrf": "ssrf_sinks",
    "xss": "xss_sinks",
    "deser": "deser_sinks",
    "eval": "eval_sinks",
    "redirect": "redirect_sinks",
    "crypto": "crypto_patterns",
}


@dataclass(frozen=True)
class LanguageSample:
    language_id: str
    filename: str
    source: str


def graph_for(language_id: str, filename: str, source: str) -> SyntaxGraph:
    return parse_source(language_id, Path(filename), source)


def analyze_source(tmp_path: Path, filename: str, source: str) -> SecurityScanResult:
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return SecurityAnalysisEngine().analyze_repository(tmp_path, [path])


def observation_classes(result: SecurityScanResult) -> set[VulnerabilityClass]:
    return {obs.vulnerability_class for obs in result.observations}


def adapter(language_id: str) -> LanguageAdapter:
    return get_plugin_catalog().languages.get(language_id)


def profile_supports(language_id: str, category: str) -> bool:
    from app.security.language_vocab import vocab_for

    vocab = vocab_for(language_id)
    if vocab is not None:
        if category == "crypto":
            return bool(vocab.crypto_names)
        mapping = {
            "sql": VulnerabilityClass.SQL_INJECTION,
            "command": VulnerabilityClass.COMMAND_INJECTION,
            "path": VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL,
            "ssrf": VulnerabilityClass.SSRF,
            "xss": VulnerabilityClass.XSS,
            "deser": VulnerabilityClass.UNSAFE_DESERIALIZATION,
            "eval": VulnerabilityClass.DYNAMIC_EXECUTION,
            "redirect": VulnerabilityClass.UNSAFE_REDIRECT,
        }
        vuln = mapping[category]
        if any(s.vulnerability_class is vuln for s in vocab.sinks):
            return True
        if category == "deser" and any(
            s.vulnerability_class is VulnerabilityClass.POTENTIAL_UNSAFE_DESERIALIZATION
            for s in vocab.sinks
        ):
            return True
        if category == "path" and any(
            s.vulnerability_class is VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL
            for s in vocab.sinks
        ):
            return True
        return False
    profile = profile_for(language_id)
    if profile is None:
        return False
    if category == "crypto":
        return bool(profile.crypto_patterns)
    return bool(getattr(profile, SINK_ATTR[category]))


def registered_analysis_ids() -> tuple[str, ...]:
    return tuple(PROFILES)


EXTENSIONS: dict[str, tuple[str, ...]] = {
    "python": (".py", ".pyi"),
    "javascript": (".js", ".mjs", ".cjs", ".jsx"),
    "typescript": (".ts", ".tsx"),
    "ruby": (".rb",),
    "c": (".c", ".h"),
    "cpp": (".cpp", ".cc", ".cxx", ".hpp"),
    "go": (".go",),
    "rust": (".rs",),
    "java": (".java",),
    "php": (".php",),
    "kotlin": (".kt", ".kts"),
    "swift": (".swift",),
    "csharp": (".cs",),
    "shell": (".sh", ".bash", ".zsh"),
    "yaml": (".yaml", ".yml"),
    "json": (".json",),
    "toml": (".toml",),
    "markdown": (".md",),
    "restructuredtext": (".rst",),
    "html": (".html", ".htm"),
    "css": (".css",),
    "scss": (".scss", ".sass"),
    "sql": (".sql",),
    "r": (".r",),
    "scala": (".scala",),
    "dart": (".dart",),
    "lua": (".lua",),
    "elixir": (".ex", ".exs"),
    "vyper": (".vy",),
}
