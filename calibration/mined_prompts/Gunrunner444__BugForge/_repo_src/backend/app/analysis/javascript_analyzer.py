"""JavaScript/TypeScript quality rules using syntax-graph events.

These are CODE QUALITY findings, not security observations.
"""

from __future__ import annotations

from pathlib import Path

from app.analysis.base import CodeQualityRule
from app.analysis.catalog import FindingCatalog
from app.analysis.finding import Finding
from app.parsing.engine import parse_source
from app.parsing.model import SyntaxGraph


def _language_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".ts", ".tsx"}:
        return "typescript"
    return "javascript"


def _graph(file_path: Path, source: str) -> SyntaxGraph:
    return parse_source(_language_for(file_path), file_path, source)


def _finding(
    rule: CodeQualityRule,
    file_path: Path,
    graph: SyntaxGraph,
    line: int,
    message: str,
    explanation: str,
    fix: str,
    evidence: str,
    *,
    extra_line: int | None = None,
) -> Finding:
    event_span = None
    return Finding(
        category=rule.RULE_ID,
        severity=rule.SEVERITY,
        confidence=rule.CONFIDENCE,
        file_path=str(file_path),
        line=line,
        end_line=extra_line or line,
        message=message,
        explanation=explanation,
        analyzer=rule.RULE_ID,
        evidence=evidence[:200],
        suggested_fix=fix,
        catalog=FindingCatalog.CODE_QUALITY,
        language=graph.language,
        parser_backend=graph.parser_backend,
        start_column=event_span,
    )


class LooseEqualityRule(CodeQualityRule):
    """Flag `==` / `!=` except the `== null` / `== undefined` idiom."""

    RULE_ID = "js_loose_equality"
    SEVERITY = "low"
    CONFIDENCE = "high"

    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        return self.check_graph(_graph(file_path, source))

    def check_graph(self, graph: SyntaxGraph) -> list[Finding]:
        findings: list[Finding] = []
        for event in graph.events:
            if event.kind != "loose_eq":
                continue
            compact = "".join(event.text.split()).lower()
            parts = event.extra.split("|")
            right = parts[2] if len(parts) >= 3 else ""
            if right in {"null", "undefined"}:
                continue
            if compact in {"x==null", "x!=null"}:
                continue
            # Only skip the null/undefined idiom when that operand is the comparison
            # itself, not because those words appear elsewhere in the expression.
            if len(parts) < 3 and (
                compact.endswith("==null")
                or compact.endswith("!=null")
                or compact.endswith("==undefined")
                or compact.endswith("!=undefined")
            ):
                continue
            findings.append(
                _finding(
                    self,
                    Path(graph.file_path),
                    graph,
                    event.line,
                    "Use `===` instead of `==`"
                    if "==" in event.text
                    else "Use `!==` instead of `!=`",
                    "Loose equality coerces types and hides bugs. Strict equality "
                    "(`===` / `!==`) compares without coercion. The `== null` idiom "
                    "is allowed because it matches both `null` and `undefined`.",
                    "Replace `==` with `===`",
                    event.text,
                )
            )
        return findings


class VarDeclarationRule(CodeQualityRule):
    """Flag `var` in favor of `let` / `const`."""

    RULE_ID = "js_var_declaration"
    SEVERITY = "low"
    CONFIDENCE = "high"

    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        return self.check_graph(_graph(file_path, source))

    def check_graph(self, graph: SyntaxGraph) -> list[Finding]:
        findings: list[Finding] = []
        for event in graph.events:
            if event.kind != "var_decl":
                continue
            findings.append(
                _finding(
                    self,
                    Path(graph.file_path),
                    graph,
                    event.line,
                    "`var` is function-scoped; prefer `let` or `const`",
                    "`var` is function-scoped and hoisted, which leads to accidental "
                    "sharing across blocks. Use `const` by default and `let` when the "
                    "binding must be reassigned.",
                    "Replace `var` with `const` or `let`",
                    event.text,
                )
            )
        return findings


class EmptyCatchRule(CodeQualityRule):
    """Flag empty `catch` blocks that swallow errors."""

    RULE_ID = "js_empty_catch"
    SEVERITY = "medium"
    CONFIDENCE = "high"

    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        return self.check_graph(_graph(file_path, source))

    def check_graph(self, graph: SyntaxGraph) -> list[Finding]:
        findings: list[Finding] = []
        for event in graph.events:
            if event.kind != "empty_catch":
                continue
            findings.append(
                _finding(
                    self,
                    Path(graph.file_path),
                    graph,
                    event.line,
                    "Empty `catch` block swallows errors",
                    "An empty catch hides failures and makes production issues undiagnosable. "
                    "Log, rethrow, or handle the error explicitly. A catch whose only content "
                    "is an 'intentionally empty' comment is allowed.",
                    "Handle or rethrow the error inside the catch block",
                    event.text,
                )
            )
        return findings


class WithStatementRule(CodeQualityRule):
    """Flag the `with` statement, which is forbidden in strict mode."""

    RULE_ID = "js_with_statement"
    SEVERITY = "medium"
    CONFIDENCE = "high"

    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        return self.check_graph(_graph(file_path, source))

    def check_graph(self, graph: SyntaxGraph) -> list[Finding]:
        findings: list[Finding] = []
        for event in graph.events:
            if event.kind != "with_stmt":
                continue
            extra = event.extra or "sloppy"
            if extra == "strict":
                message = "`with` statements are forbidden in strict mode"
                explanation = (
                    "`with` mutates identifier lookup and is a SyntaxError under "
                    "`'use strict'`, ES modules, and TypeScript. Reference properties "
                    "on the object explicitly."
                )
            else:
                message = "`with` is sloppy-mode only and is discouraged"
                explanation = (
                    "`with` is allowed only in non-strict scripts and is a SyntaxError "
                    "in modules and strict mode. Prefer explicit property access."
                )
            findings.append(
                _finding(
                    self,
                    Path(graph.file_path),
                    graph,
                    event.line,
                    message,
                    explanation,
                    "Replace `with (obj) { x }` with `obj.x`",
                    event.text,
                )
            )
        return findings


ALL_JAVASCRIPT_RULES: list[CodeQualityRule] = [
    LooseEqualityRule(),
    VarDeclarationRule(),
    EmptyCatchRule(),
    WithStatementRule(),
]
