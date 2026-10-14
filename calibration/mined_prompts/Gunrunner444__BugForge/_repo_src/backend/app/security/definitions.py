"""Semantic source, sink, sanitizer, and path-policy abstractions.

These types are language-neutral. Concrete catalogs live in language
vocabularies and are looked up by id — the taint engine never switches on
language names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.domain.security import VulnerabilityClass


class PathIssueKind(StrEnum):
    USER_CONTROLLED_PATH = "user_controlled_path"
    UNSAFE_PATH_CONSTRUCTION = "unsafe_path_construction"
    POSSIBLE_PATH_TRAVERSAL = "possible_path_traversal"
    DEMONSTRATED_DIRECTORY_ESCAPE = "demonstrated_directory_escape"


class SinkCertainty(StrEnum):
    SENSITIVE = "sensitive_sink"
    INDICATOR = "indicator"


class SanitizerKind(StrEnum):
    HTML_ENCODE = "html_encode"
    SQL_PARAMETERIZE = "sql_parameterize"
    PATH_CANONICALIZE = "path_canonicalize"
    URL_ALLOWLIST = "url_allowlist"
    SHELL_ESCAPE = "shell_escape"
    TYPE_VALIDATION = "type_validation"
    FRAMEWORK_AUTO_ESCAPE = "framework_auto_escape"


SINK_SANITIZER_KINDS: dict[VulnerabilityClass, frozenset[str]] = {
    VulnerabilityClass.XSS: frozenset(
        {SanitizerKind.HTML_ENCODE, SanitizerKind.FRAMEWORK_AUTO_ESCAPE}
    ),
    VulnerabilityClass.SQL_INJECTION: frozenset({SanitizerKind.SQL_PARAMETERIZE}),
    VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL: frozenset({SanitizerKind.PATH_CANONICALIZE}),
    VulnerabilityClass.SSRF: frozenset({SanitizerKind.URL_ALLOWLIST}),
    VulnerabilityClass.UNSAFE_REDIRECT: frozenset({SanitizerKind.URL_ALLOWLIST}),
    VulnerabilityClass.COMMAND_INJECTION: frozenset({SanitizerKind.SHELL_ESCAPE}),
}


@dataclass(frozen=True)
class SourceDefinition:
    source_id: str
    patterns: tuple[str, ...]
    kind: str  # http | cli | env | file | deserialized | browser | framework
    frameworks: tuple[str, ...] = ()


@dataclass(frozen=True)
class SinkDefinition:
    sink_id: str
    vulnerability_class: VulnerabilityClass
    api_names: tuple[str, ...]
    argument_index: int = 0
    requires_taint: bool = True
    certainty: SinkCertainty = SinkCertainty.SENSITIVE
    sanitizer_ids: tuple[str, ...] = ()
    safe_alternatives: tuple[str, ...] = ()
    frameworks: tuple[str, ...] = ()
    dangerous_condition: str = "attacker-controlled data reaches this API"
    notes: str = ""
    qualified_substrings: tuple[str, ...] = ()
    argument_indexes: tuple[int, ...] = ()
    sanitizer_kinds: tuple[str, ...] = ()
    required_context: str = ""
    receiver_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class SanitizerDefinition:
    sanitizer_id: str
    api_names: tuple[str, ...]
    kind: str  # html_encode | sql_parameterize | path_canonicalize | url_allowlist | shell_escape | type_validation | framework_auto_escape
    effective: bool = False
    notes: str = ""


@dataclass(frozen=True)
class LanguageSecurityVocab:
    language_id: str
    sources: tuple[SourceDefinition, ...] = ()
    sinks: tuple[SinkDefinition, ...] = ()
    sanitizers: tuple[SanitizerDefinition, ...] = ()
    crypto_names: tuple[str, ...] = ()
    extra_sources_by_framework: dict[str, tuple[SourceDefinition, ...]] = field(
        default_factory=dict
    )

    def sinks_for(self, vulnerability_class: VulnerabilityClass) -> tuple[SinkDefinition, ...]:
        return tuple(s for s in self.sinks if s.vulnerability_class is vulnerability_class)
