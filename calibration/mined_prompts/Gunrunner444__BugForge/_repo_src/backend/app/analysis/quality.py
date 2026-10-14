"""Language-appropriate code-quality rules driven by SyntaxGraph events.

Security observations stay in the security catalog. These rules never emit
vulnerability classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.analysis.base import AnalyzerRule, CodeQualityRule
from app.analysis.catalog import FindingCatalog
from app.analysis.finding import Finding
from app.analysis.javascript_analyzer import ALL_JAVASCRIPT_RULES
from app.analysis.python_analyzer import ALL_PYTHON_RULES
from app.parsing.engine import parse_source
from app.parsing.model import SyntaxEvent, SyntaxGraph


def _finding(
    rule: CodeQualityRule,
    graph: SyntaxGraph,
    event: SyntaxEvent,
    message: str,
    explanation: str,
    fix: str,
) -> Finding:
    span = event.span
    return Finding(
        category=rule.RULE_ID,
        severity=rule.SEVERITY,
        confidence=rule.CONFIDENCE,
        file_path=graph.file_path,
        line=event.line,
        end_line=span.end_line if span is not None else event.line,
        message=message,
        explanation=explanation,
        analyzer=rule.RULE_ID,
        evidence=event.text[:200],
        suggested_fix=fix,
        catalog=FindingCatalog.CODE_QUALITY,
        language=graph.language,
        parser_backend=graph.parser_backend,
        start_column=span.start_column if span is not None else None,
        end_column=span.end_column if span is not None else None,
        start_byte=span.start_byte if span is not None else None,
        end_byte=span.end_byte if span is not None else None,
        node_id=f"{rule.RULE_ID}:{event.line}",
    )


class GraphEventRule(CodeQualityRule):
    EVENT_KIND: str = ""
    MESSAGE: str = ""
    EXPLANATION: str = ""
    FIX: str = ""

    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        language = _language_hint(file_path, self)
        return self.check_graph(parse_source(language, file_path, source))

    def check_graph(self, graph: SyntaxGraph) -> list[Finding]:
        findings: list[Finding] = []
        for event in graph.events:
            if event.kind != self.EVENT_KIND:
                continue
            findings.append(_finding(self, graph, event, self.MESSAGE, self.EXPLANATION, self.FIX))
        return findings


class EmptyCatchQualityRule(GraphEventRule):
    RULE_ID = "quality_empty_catch"
    SEVERITY = "medium"
    CONFIDENCE = "high"
    EVENT_KIND = "empty_catch"
    MESSAGE = "Empty catch/except/rescue swallows errors"
    EXPLANATION = (
        "An empty handler hides failures. Log, rethrow, or handle the error. "
        "A handler whose only content is an intentional-empty comment is allowed."
    )
    FIX = "Handle or rethrow the error"


class ShadowingQualityRule(GraphEventRule):
    RULE_ID = "quality_shadowing"
    SEVERITY = "low"
    CONFIDENCE = "medium"
    EVENT_KIND = "shadowing"
    MESSAGE = "Declaration shadows an outer binding of the same name"
    EXPLANATION = (
        "Inner declarations that reuse an outer name make data-flow harder to follow "
        "and hide the outer binding."
    )
    FIX = "Rename the inner binding"


class DeprecatedApiQualityRule(GraphEventRule):
    RULE_ID = "quality_deprecated_api"
    SEVERITY = "medium"
    CONFIDENCE = "high"
    EVENT_KIND = "deprecated_api"
    MESSAGE = "Deprecated or unsafe standard-library API"
    EXPLANATION = (
        "APIs such as gets/strcpy/sprintf are unbounded or withdrawn. Use bounded "
        "alternatives (fgets, snprintf) or safer wrappers."
    )
    FIX = "Replace with a bounded/safe API"


class UnwrapQualityRule(GraphEventRule):
    RULE_ID = "quality_unwrap"
    SEVERITY = "medium"
    CONFIDENCE = "high"
    EVENT_KIND = "unwrap"
    MESSAGE = "Forced unwrap/expect can panic on None/Err"
    EXPLANATION = (
        "unwrap/expect/!! convert absence into an abort. Prefer match/if-let/? or "
        "explicit error handling."
    )
    FIX = "Handle the None/Err case explicitly"


class PanicQualityRule(GraphEventRule):
    RULE_ID = "quality_panic"
    SEVERITY = "low"
    CONFIDENCE = "medium"
    EVENT_KIND = "panic"
    MESSAGE = "Unconditional panic/abort in application code"
    EXPLANATION = "panic!/todo!/unimplemented! abort the process. Reserve them for true invariants."
    FIX = "Return a Result/error instead of panicking"


class GotoQualityRule(GraphEventRule):
    RULE_ID = "quality_goto"
    SEVERITY = "low"
    CONFIDENCE = "high"
    EVENT_KIND = "goto"
    MESSAGE = "goto obscures control flow"
    EXPLANATION = "goto makes reachability and resource cleanup harder to reason about."
    FIX = "Restructure with structured control flow"


class DebuggerQualityRule(GraphEventRule):
    RULE_ID = "quality_debugger"
    SEVERITY = "low"
    CONFIDENCE = "high"
    EVENT_KIND = "debugger"
    MESSAGE = "debugger statement left in source"
    EXPLANATION = "debugger pauses execution and should not ship in production code."
    FIX = "Remove the debugger statement"


class ForceUnwrapQualityRule(GraphEventRule):
    RULE_ID = "quality_force_unwrap"
    SEVERITY = "medium"
    CONFIDENCE = "high"
    EVENT_KIND = "force_unwrap"
    MESSAGE = "Force-try / force-unwrap"
    EXPLANATION = "Forced unwraps crash when the value is missing. Use optional binding."
    FIX = "Use if-let / guard / try?"


class ForLoopQualityRule(GraphEventRule):
    RULE_ID = "quality_for_loop"
    SEVERITY = "low"
    CONFIDENCE = "medium"
    EVENT_KIND = "for_loop"
    MESSAGE = "for/in iterates with a separate scope; each/map is the usual Ruby style"
    EXPLANATION = (
        "Ruby `for` leaks the iterator into the enclosing scope in some versions and is "
        "rarely the intended iteration form. Prefer `each`."
    )
    FIX = "Replace `for x in xs` with `xs.each do |x|`"


class UnquotedExpansionQualityRule(GraphEventRule):
    RULE_ID = "quality_unquoted_expansion"
    SEVERITY = "medium"
    CONFIDENCE = "high"
    EVENT_KIND = "unquoted_expansion"
    MESSAGE = "Unquoted shell expansion is subject to word-splitting and globbing"
    EXPLANATION = (
        "Unquoted `$var` / `$(...)` expansions split on IFS and expand globs. Quote the "
        "expansion when the value is a single argument."
    )
    FIX = 'Use "$var" or "${var}" instead of $var'


class PosixTestQualityRule(GraphEventRule):
    RULE_ID = "quality_posix_test"
    SEVERITY = "low"
    CONFIDENCE = "medium"
    EVENT_KIND = "posix_test"
    MESSAGE = "POSIX `[` test is more error-prone than `[[` in bash"
    EXPLANATION = (
        "`[` is an ordinary command and requires quoting. Bash `[[` is parsed as syntax "
        "and avoids several word-splitting pitfalls."
    )
    FIX = "Prefer [[ ... ]] in bash scripts, with quoted expansions"


class BlankIdentQualityRule(GraphEventRule):
    RULE_ID = "quality_blank_ident"
    SEVERITY = "medium"
    CONFIDENCE = "medium"
    EVENT_KIND = "blank_ident"
    MESSAGE = "Blank identifier discards a result, often an error"
    EXPLANATION = (
        "Assigning to `_` ignores the value. In Go this commonly discards an error that "
        "should be checked."
    )
    FIX = "Handle the discarded result, especially error values"


class AnalyzerErrorRule(CodeQualityRule):
    """Emitted when a quality rule raises; never look like a clean file."""

    RULE_ID = "analyzer_error"
    SEVERITY = "medium"
    CONFIDENCE = "high"


def analyzer_error_finding(
    file_path: Path, language: str, parser_backend: str, rule_id: str, exc: BaseException
) -> Finding:
    return Finding(
        category=AnalyzerErrorRule.RULE_ID,
        severity="medium",
        confidence="high",
        file_path=str(file_path),
        line=1,
        end_line=1,
        message=f"Analyzer {rule_id} failed",
        explanation=f"{type(exc).__name__}: {exc}. The file was not fully analyzed.",
        analyzer=rule_id,
        evidence="",
        suggested_fix="Fix the analyzer failure or reduce the file to a parseable subset.",
        catalog=FindingCatalog.CODE_QUALITY,
        language=language,
        parser_backend=parser_backend,
    )


COMMON_FULL_ANALYSIS_RULES: list[CodeQualityRule] = [
    EmptyCatchQualityRule(),
    ShadowingQualityRule(),
]


def _solidity_quality_rules() -> list[AnalyzerRule]:
    from app.analysis.solidity_quality import SOLIDITY_QUALITY_RULES

    return list(SOLIDITY_QUALITY_RULES)


LANGUAGE_QUALITY_RULES: dict[str, Sequence[AnalyzerRule]] = {
    "python": list(ALL_PYTHON_RULES),
    "javascript": list(ALL_JAVASCRIPT_RULES) + [DebuggerQualityRule(), ShadowingQualityRule()],
    "typescript": list(ALL_JAVASCRIPT_RULES) + [DebuggerQualityRule(), ShadowingQualityRule()],
    "ruby": list(COMMON_FULL_ANALYSIS_RULES) + [ForLoopQualityRule()],
    "c": list(COMMON_FULL_ANALYSIS_RULES) + [DeprecatedApiQualityRule(), GotoQualityRule()],
    "cpp": list(COMMON_FULL_ANALYSIS_RULES) + [DeprecatedApiQualityRule(), GotoQualityRule()],
    "go": list(COMMON_FULL_ANALYSIS_RULES) + [PanicQualityRule(), BlankIdentQualityRule()],
    "rust": list(COMMON_FULL_ANALYSIS_RULES) + [UnwrapQualityRule(), PanicQualityRule()],
    "java": list(COMMON_FULL_ANALYSIS_RULES) + [DeprecatedApiQualityRule()],
    "php": list(COMMON_FULL_ANALYSIS_RULES) + [DeprecatedApiQualityRule()],
    "kotlin": list(COMMON_FULL_ANALYSIS_RULES) + [ForceUnwrapQualityRule()],
    "swift": list(COMMON_FULL_ANALYSIS_RULES) + [ForceUnwrapQualityRule()],
    "csharp": list(COMMON_FULL_ANALYSIS_RULES) + [GotoQualityRule()],
    "shell": [UnquotedExpansionQualityRule(), PosixTestQualityRule(), ShadowingQualityRule()],
    "solidity": list(_solidity_quality_rules()),
}


def quality_rules_for(language_id: str) -> list[AnalyzerRule]:
    return list(LANGUAGE_QUALITY_RULES.get(language_id, list(COMMON_FULL_ANALYSIS_RULES)))


def _language_hint(file_path: Path, rule: CodeQualityRule) -> str:
    suffix = file_path.suffix.lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".rb": "ruby",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".php": "php",
        ".kt": "kotlin",
        ".swift": "swift",
        ".cs": "csharp",
        ".sh": "shell",
        ".bash": "shell",
    }
    return mapping.get(suffix, "javascript")
