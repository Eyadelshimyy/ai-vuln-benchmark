"""Normalized dynamic-engine results. Tool text is untrusted evidence, not instructions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.discovery.capabilities import ResultStatus
from app.domain.evidence import Evidence, EvidenceKind
from app.security_testing.secrets import redact_text


@dataclass(frozen=True)
class DynamicFinding:
    """One tool observation. Status stays potential until BugForge corroborates it."""

    detector_id: str
    title: str
    severity: str = ""
    confidence: str = ""
    contract: str = ""
    function: str = ""
    file_path: str = ""
    line: int = 0
    description: str = ""
    status: str = "potential"


@dataclass
class DynamicResult:
    engine: str
    language: str
    target: str
    status: ResultStatus = ResultStatus.PLANNED
    executed: bool = False
    engine_version: str = ""
    campaign_id: str = ""
    seed_id: str = ""
    coverage: dict[str, str] = field(default_factory=dict)
    exit_code: int | None = None
    crash: str = ""
    assertion: str = ""
    sanitizer: str = ""
    stdout: str = ""
    stderr: str = ""
    minimized_input: str = ""
    reproduction_command: str = ""
    source_file: str = ""
    function: str = ""
    contract: str = ""
    provenance: str = ""
    timestamp: str = ""
    oracle_explanation: str = ""
    oracle_kind: str = ""
    findings: tuple[DynamicFinding, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
    economic_evidence: object | None = None
    protocol_evidence: object | None = None
    runtime_evidence: object | None = None

    def __post_init__(self) -> None:
        self.stdout = redact_text(self.stdout)
        self.stderr = redact_text(self.stderr)
        self.crash = redact_text(self.crash)
        self.minimized_input = redact_text(self.minimized_input)
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    def to_evidence(self) -> Evidence:
        kind = _evidence_kind(self)
        summary = f"{self.engine} {self.status.value} executed={self.executed}"
        if self.oracle_explanation:
            summary = f"{summary}: {self.oracle_explanation}"
        return Evidence(
            kind=kind,
            source=self.engine,
            summary=summary[:500],
            details=redact_text(self.stdout or self.stderr)[:2000],
            artifact_path=self.source_file or None,
            metadata={
                "engine": self.engine,
                "engine_version": self.engine_version,
                "language": self.language,
                "target": self.target,
                "campaign_id": self.campaign_id,
                "seed_id": self.seed_id,
                "status": self.status.value,
                "executed": str(self.executed).lower(),
                "contract": self.contract,
                "function": self.function,
                "provenance": self.provenance,
                "timestamp": self.timestamp,
                "oracle": self.oracle_explanation,
                "oracle_kind": self.oracle_kind,
                "economic_bound": str(self.metadata.get("bound", "false")),
                "observation_class": self.metadata.get("observation_class", ""),
                "crash": self.crash,
                "assertion": self.assertion,
                "sanitizer": self.sanitizer,
                **{key: str(value) for key, value in self.coverage.items()},
                **self.metadata,
                "verified": "false",
            },
        )


def _evidence_kind(result: DynamicResult) -> EvidenceKind:
    """Map a dynamic observation to an evidence kind without calling it verified.

    Assertion, property, symbolic, sanitizer, and crash observations stay
    distinguishable. A quiet log is only for an execution that has none of them.
    """
    if result.metadata.get("evidence_class") == "tool_status":
        return EvidenceKind.TOOL_STATUS
    oracle = result.oracle_kind
    symbolic = (
        oracle in {"symbolic", "counterexample"} or result.metadata.get("seed_source") == "symbolic"
    )
    if symbolic and (result.minimized_input or result.assertion):
        return EvidenceKind.FUZZING
    if result.crash or oracle == "crash":
        return EvidenceKind.TEST_FAILURE
    if result.sanitizer or oracle == "sanitizer":
        return EvidenceKind.FUZZING
    if result.assertion or oracle in {"assertion", "property", "invariant"}:
        return EvidenceKind.TEST_FAILURE
    if (
        result.metadata.get("evidence_class") == "fuzzing"
        and result.status is ResultStatus.INTERESTING
        and result.findings
    ):
        return EvidenceKind.FUZZING
    if result.oracle_kind == "economic" or result.metadata.get("evidence_class") == "economic":
        return EvidenceKind.ECONOMIC_OBSERVATION
    if result.oracle_kind == "protocol" or result.metadata.get("evidence_class") == "protocol":
        return EvidenceKind.PROTOCOL_OBSERVATION
    if result.oracle_kind in {"runtime", "differential"} or result.metadata.get(
        "evidence_class"
    ) in {
        "runtime",
        "differential",
    }:
        return EvidenceKind.RUNTIME_OBSERVATION
    if result.executed and result.status is ResultStatus.INGESTED:
        return EvidenceKind.SCANNER if result.findings else EvidenceKind.FUZZING
    if result.executed and _coverage_available(result.coverage):
        return EvidenceKind.FUZZING
    if result.executed:
        return EvidenceKind.LOG
    return EvidenceKind.TOOL_STATUS


def _coverage_available(coverage: dict[str, str]) -> bool:
    if coverage.get("coverage_available") == "true":
        return True
    return "percent" in coverage or "coverage_percent" in coverage


def unavailable_result(engine: str, language: str, target: str, *, reason: str) -> DynamicResult:
    return DynamicResult(
        engine=engine,
        language=language,
        target=target,
        status=ResultStatus.UNAVAILABLE,
        executed=False,
        provenance="tool_unavailable",
        oracle_explanation=reason,
        stderr=reason,
    )


def not_implemented_result(
    engine: str, language: str, target: str, operation: str
) -> DynamicResult:
    reason = f"{engine} does not implement {operation}"
    return DynamicResult(
        engine=engine,
        language=language,
        target=target,
        status=ResultStatus.NOT_IMPLEMENTED,
        executed=False,
        provenance="capability",
        oracle_explanation=reason,
    )


def as_dict(result: DynamicResult) -> dict[str, Any]:
    return {
        "engine": result.engine,
        "engine_version": result.engine_version,
        "language": result.language,
        "target": result.target,
        "campaign_id": result.campaign_id,
        "status": result.status.value,
        "executed": result.executed,
        "verified": False,
    }
