"""Collectors that wrap BugForge's existing static-analysis and test evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.adapters.evidence.attribution import ServerAttribution, attribution_metadata
from app.adapters.evidence.base import EvidenceCollector
from app.domain.evidence import Evidence, EvidenceKind, EvidenceSource

_FAILED_REPRODUCTION = frozenset(
    {
        "failed",
        "not_reproduced",
        "blocked",
        "inconclusive",
        "intermittent",
        "error",
        "no_evidence",
        "environment_error",
        "timeout",
    }
)
_POSITIVE_REPRODUCTION = frozenset(
    {"reproduced", "consistently_reproduced", "success", "exploited"}
)


class StaticAnalysisEvidenceCollector(EvidenceCollector):
    @property
    def collector_id(self) -> str:
        return "static_analysis"

    @property
    def kinds(self) -> frozenset[EvidenceKind]:
        return frozenset({EvidenceKind.STATIC_ANALYSIS})

    def collect(self, source: EvidenceSource) -> Sequence[Evidence]:
        items: list[Evidence] = []
        for finding in source.static_findings:
            items.append(_from_static_finding(finding))
        return items


class TestFailureEvidenceCollector(EvidenceCollector):
    @property
    def collector_id(self) -> str:
        return "test_failure"

    @property
    def kinds(self) -> frozenset[EvidenceKind]:
        return frozenset({EvidenceKind.TEST_FAILURE})

    def collect(self, source: EvidenceSource) -> Sequence[Evidence]:
        items: list[Evidence] = []
        for failure in source.test_failures:
            node_id = str(getattr(failure, "node_id", "") or getattr(failure, "test_name", "test"))
            traceback = getattr(failure, "traceback", None)
            metadata = {
                "test_file": getattr(failure, "test_file", None),
                "test_name": getattr(failure, "test_name", None),
                "outcome": "failed",
                "reproduced": "false",
            }
            metadata.update(_server_metadata(source))
            items.append(
                Evidence(
                    kind=EvidenceKind.TEST_FAILURE,
                    source="test_runner",
                    summary=f"Failing test {node_id}",
                    details=str(traceback or ""),
                    metadata={key: value for key, value in metadata.items() if value is not None},
                )
            )
        return items


class SourceCodeEvidenceCollector(EvidenceCollector):
    @property
    def collector_id(self) -> str:
        return "source_code"

    @property
    def kinds(self) -> frozenset[EvidenceKind]:
        return frozenset({EvidenceKind.SOURCE_CODE})

    def collect(self, source: EvidenceSource) -> Sequence[Evidence]:
        items: list[Evidence] = []
        for file_evidence in source.source_files:
            path = str(getattr(file_evidence, "file_path", "") or "")
            content = str(getattr(file_evidence, "content", "") or "")
            language = str(getattr(file_evidence, "language", "") or "")
            items.append(
                Evidence(
                    kind=EvidenceKind.SOURCE_CODE,
                    source="repository",
                    summary=f"Source file {path}" if path else "Source file",
                    details=content[:4_000],
                    artifact_path=path or None,
                    metadata={"language": language} if language else {},
                )
            )
        return items


class ReproductionEvidenceCollector(EvidenceCollector):
    @property
    def collector_id(self) -> str:
        return "reproduction"

    @property
    def kinds(self) -> frozenset[EvidenceKind]:
        return frozenset({EvidenceKind.REPRODUCTION, EvidenceKind.LOG, EvidenceKind.TEST_FAILURE})

    def collect(self, source: EvidenceSource) -> Sequence[Evidence]:
        items: list[Evidence] = []
        for reproduction in source.reproductions:
            summary = str(
                getattr(reproduction, "summary", None)
                or getattr(reproduction, "classification", None)
                or "Reproduction attempt"
            )
            details = str(getattr(reproduction, "details", "") or "")
            outcome = str(
                getattr(reproduction, "outcome", None)
                or getattr(reproduction, "classification", None)
                or ""
            ).lower()
            reproduced = getattr(reproduction, "reproduced", None)
            failed = outcome in _FAILED_REPRODUCTION or reproduced is False
            success = (not failed) and (reproduced is True or outcome in _POSITIVE_REPRODUCTION)
            kind = EvidenceKind.LOG
            if success:
                kind = EvidenceKind.REPRODUCTION
            elif failed:
                kind = EvidenceKind.TEST_FAILURE
            recorded_outcome = outcome
            if success and recorded_outcome not in _POSITIVE_REPRODUCTION:
                recorded_outcome = "reproduced"
            elif not success and not failed:
                recorded_outcome = recorded_outcome or "attempt"
            metadata: dict[str, object] = {"outcome": recorded_outcome}
            metadata["reproduced"] = "true" if success else "false"
            file_path = getattr(reproduction, "file_path", None)
            line = getattr(reproduction, "line", None)
            if file_path:
                metadata["file_path"] = file_path
            if line is not None:
                metadata["line"] = line
            metadata.update(_server_metadata(source))
            items.append(
                Evidence(
                    kind=kind,
                    source="reproduction_engine",
                    summary=summary,
                    details=details,
                    artifact_path=str(file_path) if file_path else None,
                    metadata=metadata,
                )
            )
        return items


def _server_metadata(source: EvidenceSource) -> dict[str, str]:
    raw = source.extra.get("attribution")
    if not isinstance(raw, ServerAttribution):
        return {}
    return attribution_metadata(raw)


def _from_static_finding(finding: Any) -> Evidence:
    category = str(getattr(finding, "category", "static"))
    file_path = str(getattr(finding, "file_path", ""))
    line = getattr(finding, "line", None)
    message = str(getattr(finding, "message", "") or category)
    location = f"{file_path}:{line}" if file_path and line is not None else file_path or category
    return Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source=str(getattr(finding, "analyzer", "static_analysis")),
        summary=f"{category} at {location}: {message}",
        details=str(getattr(finding, "explanation", "") or getattr(finding, "evidence", "") or ""),
        artifact_path=file_path or None,
        metadata={
            "category": category,
            "severity": getattr(finding, "severity", None),
            "confidence": getattr(finding, "confidence", None),
            "line": line,
            "vulnerability_class": getattr(finding, "vulnerability_class", None),
            "sink": getattr(finding, "flow_sink", None) or getattr(finding, "sink", None),
            "taint_source": getattr(finding, "flow_source", None),
        },
    )
