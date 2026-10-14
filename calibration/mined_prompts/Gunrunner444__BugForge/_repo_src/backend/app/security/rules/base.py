"""Security-rule contract. Rules emit observations, never verified findings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.analyzers.framework_detector import FrameworkInfo
from app.domain.evidence import Evidence, EvidenceKind
from app.domain.security import VulnerabilityClass
from app.parsing.model import SyntaxGraph


@dataclass(frozen=True)
class RuleDocumentation:
    detects: str
    evidence: str
    limitations: str
    false_positives: str


@dataclass(frozen=True)
class SecurityObservation:
    """One static indicator. Not a verified vulnerability."""

    rule_id: str
    vulnerability_class: VulnerabilityClass
    title: str
    summary: str
    file_path: str
    line: int
    evidence_text: str
    confidence: str  # low | medium | high
    language: str
    documentation: RuleDocumentation
    metadata: dict[str, str] = field(default_factory=dict)
    end_line: int | None = None
    start_column: int | None = None
    end_column: int | None = None
    start_byte: int | None = None
    end_byte: int | None = None
    node_id: str = ""
    parser_backend: str = ""
    framework_context: str = ""
    taint_path: str = ""
    possible_false_positives: str = ""
    limitations: str = ""
    argument_index: int | None = None
    parser_tier: str = ""
    scope_id: str = ""
    sink_id: str = ""
    source_id: str = ""

    def to_evidence(self) -> Evidence:
        return Evidence(
            kind=EvidenceKind.STATIC_ANALYSIS,
            source=self.rule_id,
            summary=self.summary,
            details=self.evidence_text,
            artifact_path=self.file_path,
            metadata={
                "rule_id": self.rule_id,
                "vulnerability_class": self.vulnerability_class.value,
                "line": str(self.line),
                "language": self.language,
                "confidence": self.confidence,
                "detects": self.documentation.detects,
                "limitations": self.limitations or self.documentation.limitations,
                "false_positives": self.possible_false_positives
                or self.documentation.false_positives,
                "parser_backend": self.parser_backend,
                "node_id": self.node_id,
                "framework": self.framework_context,
                "taint_path": self.taint_path,
                "argument_index": str(self.argument_index or ""),
                "parser_tier": self.parser_tier,
                "scope_id": self.scope_id,
                "sink_id": self.sink_id,
                **self.metadata,
            },
        )


class SecurityRule(ABC):
    """Language-neutral security rule. Operates on a SyntaxGraph."""

    rule_id: str
    vulnerability_class: VulnerabilityClass
    documentation: RuleDocumentation

    @abstractmethod
    def check(
        self,
        graph: SyntaxGraph,
        *,
        frameworks: Sequence[FrameworkInfo] = (),
    ) -> list[SecurityObservation]: ...
