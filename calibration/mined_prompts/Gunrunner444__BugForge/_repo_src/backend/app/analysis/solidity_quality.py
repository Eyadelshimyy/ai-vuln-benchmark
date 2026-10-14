"""Solidity code-quality rules. These are not security findings."""

from __future__ import annotations

from pathlib import Path

from app.analysis.base import CodeQualityRule
from app.analysis.finding import Finding
from app.parsing.engine import parse_source
from app.parsing.model import SyntaxEvent, SyntaxGraph


def _quality(
    rule: CodeQualityRule, graph: SyntaxGraph, event: SyntaxEvent, message: str
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
        explanation=message,
        analyzer=rule.RULE_ID,
        evidence=event.text[:200],
        suggested_fix="",
        language=graph.language,
        parser_backend=graph.parser_backend,
        node_id=f"{rule.RULE_ID}:{event.line}",
    )


class _SolidityQuality(CodeQualityRule):
    SEVERITY = "low"
    CONFIDENCE = "medium"
    KIND = ""
    MESSAGE = ""

    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        return self.check_graph(parse_source("solidity", file_path, source))

    def check_graph(self, graph: SyntaxGraph) -> list[Finding]:
        if graph.language != "solidity":
            return []
        return [
            _quality(self, graph, event, self.MESSAGE)
            for event in graph.events
            if event.kind == self.KIND
        ]


class EmptyHandlerQualityRule(_SolidityQuality):
    RULE_ID = "sol_quality_empty_handler"
    KIND = "sol_empty_handler"
    MESSAGE = "Empty fallback or receive does nothing with the received call"


class ShadowingQualityRuleSolidity(_SolidityQuality):
    RULE_ID = "sol_quality_shadowing"
    MESSAGE = "Function parameter shadows a state variable"

    def check_graph(self, graph: SyntaxGraph) -> list[Finding]:
        if graph.language != "solidity":
            return []
        state = set()
        for event in graph.events:
            if event.kind == "sol_state" and "name=" in event.extra:
                state.add(event.extra.split("name=", 1)[1].split("|", 1)[0])
        findings: list[Finding] = []
        for entity in graph.entities:
            if entity.entity_type != "function":
                continue
            for param in entity.parameters:
                if param.name in state:
                    findings.append(
                        Finding(
                            category=self.RULE_ID,
                            severity=self.SEVERITY,
                            confidence=self.CONFIDENCE,
                            file_path=graph.file_path,
                            line=entity.start_line,
                            end_line=entity.end_line,
                            message=self.MESSAGE,
                            explanation=f"{param.name} is also a state variable",
                            analyzer=self.RULE_ID,
                            evidence=param.name,
                            language="solidity",
                            parser_backend=graph.parser_backend,
                        )
                    )
        return findings


class UnsafePragmaQualityRule(_SolidityQuality):
    RULE_ID = "sol_quality_unsafe_pragma"
    KIND = "sol_pragma_unbounded"
    MESSAGE = "Pragma allows an unbounded compiler range"


class AssemblyQualityRule(_SolidityQuality):
    RULE_ID = "sol_quality_assembly"
    KIND = "sol_assembly"
    MESSAGE = "Inline assembly bypasses compiler checks"


class DeprecatedConstructQualityRule(_SolidityQuality):
    RULE_ID = "sol_quality_deprecated"
    KIND = "sol_deprecated"
    MESSAGE = "Deprecated Solidity construct"


class EmptyUncheckedQualityRule(_SolidityQuality):
    RULE_ID = "sol_quality_empty_unchecked"
    KIND = "sol_unchecked_empty"
    MESSAGE = "Unchecked block contains no arithmetic"


SOLIDITY_QUALITY_RULES: tuple[CodeQualityRule, ...] = (
    EmptyHandlerQualityRule(),
    ShadowingQualityRuleSolidity(),
    UnsafePragmaQualityRule(),
    AssemblyQualityRule(),
    DeprecatedConstructQualityRule(),
    EmptyUncheckedQualityRule(),
)
