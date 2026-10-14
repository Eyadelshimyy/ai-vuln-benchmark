"""Language-neutral source parse contract.

Core orchestration (LanguageAdapter, RepoAnalyzer, AnalysisService) depends
on these types only. Language-specific parsers may keep richer native results
as long as they satisfy this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParsedParameter:
    name: str
    annotation: str | None = None
    default: str | None = None
    kind: str = "positional"  # positional | keyword | var_positional | var_keyword


@dataclass
class ParsedEntity:
    entity_type: str  # function | async_function | class | method | async_method | ...
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    parameters: list[ParsedParameter] = field(default_factory=list)
    return_annotation: str | None = None
    parent: str | None = None
    start_column: int = 1
    end_column: int = 1
    start_byte: int = 0
    end_byte: int = 0
    node_id: str = ""


@dataclass
class ParsedImport:
    module: str
    name: str | None = None
    alias: str | None = None
    line_number: int = 0
    is_from_import: bool = False
    import_type: str = "unknown"  # stdlib | third_party | relative | local | module | include | require | use | using
    column: int = 1
    start_byte: int = 0
    end_byte: int = 0
    syntax_kind: str = "import"
    relative_level: int = 0


@dataclass
class LanguageParseResult:
    """Neutral parse output returned by every LanguageAdapter.parse_file()."""

    file_path: str
    language: str
    line_count: int = 0
    errors: list[str] = field(default_factory=list)
    imports: list[ParsedImport] = field(default_factory=list)
    entities: list[ParsedEntity] = field(default_factory=list)
    has_errors: bool = False
    error_count: int = 0
    parser_backend: str = ""
    parser_tier: str = ""
