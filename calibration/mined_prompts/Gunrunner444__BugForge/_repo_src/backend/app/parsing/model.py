"""Shared syntax-graph types used by language parsers and security rules.

This is the language-neutral substrate. Python's CPython AST and Tree-sitter
backends both fill these types. The security engine never depends on a
specific parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.domain.language import ParserTier
from app.domain.source import LanguageParseResult, ParsedEntity, ParsedImport
from app.parsing.span import SourceSpan


class SemanticKind(StrEnum):
    MODULE = "module"
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    TYPE = "type"
    PARAMETER = "parameter"
    VARIABLE = "variable"
    ASSIGNMENT = "assignment"
    CALL = "call"
    MEMBER_ACCESS = "member_access"
    LITERAL = "literal"
    RETURN = "return"
    BRANCH = "branch"
    LOOP = "loop"
    EXCEPTION_HANDLER = "exception_handler"
    IMPORT = "import"
    DECORATOR = "decorator"
    PROPERTY = "field"
    COMMENT = "comment"
    INTERPOLATION = "interpolation"


class CallKind(StrEnum):
    DIRECT = "direct"
    MEMBER = "member"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    FUNCTION_POINTER = "function_pointer"
    OPERATOR = "operator"
    MACRO = "macro"
    MEMBER_WRITE = "member_write"
    BARE = "bare"


class SymbolKind(StrEnum):
    MODULE = "module"
    LOCAL = "local"
    PARAMETER = "parameter"
    FIELD = "field"
    PROPERTY = "property"


class ScopeKind(StrEnum):
    MODULE = "module"
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    BLOCK = "block"


class ParserStatus(StrEnum):
    """Runtime parser outcome. Never confuse this with installed capability."""

    NATIVE_AVAILABLE = "native_parser_available"
    NATIVE_UNAVAILABLE = "native_parser_unavailable"
    PROFILE_FALLBACK = "profile_fallback_used"
    PARSER_FAILURE = "parser_failure"


@dataclass(frozen=True)
class ParserDiagnostics:
    has_errors: bool = False
    error_count: int = 0
    error_spans: tuple[SourceSpan, ...] = ()
    recoverable: bool = True
    truncated: bool = False
    message: str = ""
    native_available: bool = False
    status: str = ParserStatus.NATIVE_UNAVAILABLE
    fallback_reason: str = ""


@dataclass(frozen=True)
class Scope:
    scope_id: str
    kind: ScopeKind
    name: str
    parent_id: str | None = None
    span: SourceSpan | None = None
    conditional: bool = False


@dataclass(frozen=True)
class Symbol:
    """Scope-aware identity. ``A.q`` and ``B.q`` are different symbols."""

    symbol_id: str
    name: str
    scope_id: str
    kind: SymbolKind
    span: SourceSpan | None = None
    node_id: str = ""

    @staticmethod
    def make_id(scope_id: str, name: str) -> str:
        return f"{scope_id}::{name}"


@dataclass(frozen=True)
class SemanticNode:
    node_id: str
    kind: SemanticKind
    name: str
    span: SourceSpan
    parent_id: str | None = None
    language_type: str = ""
    extra: str = ""


@dataclass(frozen=True)
class CallArgument:
    """One actual argument at a call site.

    ``index`` is the positional index. Keyword arguments keep ``index`` at
    ``-1`` and set ``keyword`` so they do not shift later positional indexes.
    """

    index: int
    text: str
    is_literal: bool = False
    idents: tuple[str, ...] = ()
    accesses: tuple[str, ...] = ()
    callees: tuple[str, ...] = ()
    dynamic: bool = False
    keyword: str = ""


@dataclass(frozen=True)
class CallSite:
    """One function/method call (or security-relevant member write) in source."""

    name: str
    qualified: str
    line: int
    argument_text: str
    dynamic: bool = False
    kind: CallKind = CallKind.DIRECT
    span: SourceSpan | None = None
    node_id: str = ""
    scope_id: str = "module"
    argument_is_literal: bool = False
    argument_idents: tuple[str, ...] = ()
    argument_accesses: tuple[str, ...] = ()
    arguments: tuple[CallArgument, ...] = ()
    callee_identity: str = ""

    def argument_at(self, index: int) -> CallArgument | None:
        positional = [argument for argument in self.arguments if not argument.keyword]
        for argument in positional:
            if argument.index == index:
                return argument
        if 0 <= index < len(positional):
            return positional[index]
        return None


@dataclass(frozen=True)
class Binding:
    """A name bound to an expression (assignment / declaration / parameter)."""

    name: str
    line: int
    rhs: str
    scope_id: str = "module"
    kind: SymbolKind = SymbolKind.LOCAL
    span: SourceSpan | None = None
    node_id: str = ""
    rhs_is_literal: bool = False
    rhs_callees: tuple[str, ...] = ()
    rhs_accesses: tuple[str, ...] = ()
    rhs_idents: tuple[str, ...] = ()
    definition_index: int = 0
    is_declaration: bool = True
    is_conditional: bool = False
    declarator: str = ""

    @property
    def symbol_id(self) -> str:
        return Symbol.make_id(self.scope_id, self.name)

    @property
    def version_id(self) -> str:
        return f"{self.symbol_id}#{self.definition_index}"


@dataclass(frozen=True)
class ReturnSite:
    scope_id: str
    line: int
    text: str
    idents: tuple[str, ...] = ()
    accesses: tuple[str, ...] = ()
    span: SourceSpan | None = None


@dataclass(frozen=True)
class RouteEndpoint:
    """A route declared by syntax. Decorators and calls are not executed."""

    method: str
    path: str
    function: str
    scope_id: str
    line: int
    parameter_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SyntaxEvent:
    """Language-neutral syntax event used by code-quality rules."""

    kind: str
    line: int
    text: str
    span: SourceSpan | None = None
    extra: str = ""


@dataclass
class SyntaxGraph:
    """Syntax-aware view of one source file."""

    language: str
    file_path: str
    source: str
    lines: tuple[str, ...]
    imports: tuple[ParsedImport, ...] = ()
    entities: tuple[ParsedEntity, ...] = ()
    calls: tuple[CallSite, ...] = ()
    bindings: tuple[Binding, ...] = ()
    errors: tuple[str, ...] = ()
    parser_backend: str = "profile"
    parser_tier: ParserTier = ParserTier.PROFILE_FALLBACK
    diagnostics: ParserDiagnostics = field(default_factory=ParserDiagnostics)
    scopes: tuple[Scope, ...] = ()
    symbols: tuple[Symbol, ...] = ()
    nodes: tuple[SemanticNode, ...] = ()
    returns: tuple[ReturnSite, ...] = ()
    events: tuple[SyntaxEvent, ...] = ()
    framework: str = ""
    file_context: str = ""  # library | test | generated | handler | cli | unknown
    semantic_context: tuple[str, ...] = ()
    default_export: str = ""
    routes: tuple[RouteEndpoint, ...] = ()

    def to_parse_result(self) -> LanguageParseResult:
        diag = self.diagnostics
        errors = list(self.errors)
        if diag.has_errors and not errors:
            errors.append(
                f"syntax errors: {diag.error_count}"
                + (f" ({diag.message})" if diag.message else "")
            )
        return LanguageParseResult(
            file_path=self.file_path,
            language=self.language,
            line_count=len(self.lines) or 1,
            errors=errors,
            imports=list(self.imports),
            entities=list(self.entities),
            has_errors=diag.has_errors or bool(errors),
            error_count=diag.error_count or len(errors),
            parser_backend=self.parser_backend,
            parser_tier=str(self.parser_tier),
        )


@dataclass
class LanguageProfile:
    """Declarative parser + taint profile for one language.

    Profiles live in the parsing/security adapters, not in core orchestration.
    Used as Tree-sitter fallback vocabulary and for the labeled profile parser.
    """

    language_id: str
    display_name: str
    extensions: frozenset[str]
    line_comment: str | tuple[str, ...] | None = "//"
    block_comment: tuple[str, str] | None = ("/*", "*/")
    import_patterns: tuple[str, ...] = ()
    function_patterns: tuple[str, ...] = ()
    class_patterns: tuple[str, ...] = ()
    assignment_patterns: tuple[str, ...] = ()
    bare_call_keywords: tuple[str, ...] = ()
    source_patterns: tuple[str, ...] = ()
    sql_sinks: tuple[str, ...] = ()
    command_sinks: tuple[str, ...] = ()
    path_sinks: tuple[str, ...] = ()
    ssrf_sinks: tuple[str, ...] = ()
    xss_sinks: tuple[str, ...] = ()
    deser_sinks: tuple[str, ...] = ()
    eval_sinks: tuple[str, ...] = ()
    redirect_sinks: tuple[str, ...] = ()
    crypto_patterns: tuple[str, ...] = ()
    extra_source_by_framework: dict[str, tuple[str, ...]] = field(default_factory=dict)
    tree_sitter_language: str | None = None
