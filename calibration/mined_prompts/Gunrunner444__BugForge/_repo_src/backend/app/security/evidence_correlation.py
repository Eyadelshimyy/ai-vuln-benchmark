"""Correlate static findings with tests, reproductions, and hypotheses.

This uses the existing finding and evidence types. It does not create a
second evidence store and it never marks a finding verified. A test result
is attached evidence, not a bypass of ``SecurityFinding.verify``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.domain.evidence import (
    Evidence,
    EvidenceKind,
    EvidenceProvenance,
    evidence_contradicts,
    observation_identity,
)
from app.domain.findings import SecurityFinding

_RUNTIME_KINDS = frozenset(
    {
        EvidenceKind.TEST_FAILURE,
        EvidenceKind.REPRODUCTION,
        EvidenceKind.API_TEST,
        EvidenceKind.REPLAY,
        EvidenceKind.HTTP_RESPONSE,
        EvidenceKind.HTTP_REQUEST,
        EvidenceKind.BROWSER,
        EvidenceKind.SCANNER,
        EvidenceKind.FUZZING,
        EvidenceKind.PROXY,
        EvidenceKind.LOG,
    }
)


@dataclass(frozen=True)
class ConfidenceExplanation:
    """Why a finding looks the way it does. Not a verification decision."""

    summary: str
    detected_because: str
    source: str
    sink: str
    relationships: tuple[str, ...]
    parser: str
    cross_file: bool
    field_path: str
    analysis_incomplete: str
    runtime_support: str
    evidence_kinds: tuple[str, ...]
    static_status: str


def explain_confidence(finding: SecurityFinding) -> ConfidenceExplanation:
    """Read provenance already stored on the finding. Do not invent steps."""
    static = [
        item
        for item in finding.evidence.items
        if item.provenance is EvidenceProvenance.STATIC_ANALYSIS
        or item.kind is EvidenceKind.STATIC_ANALYSIS
    ]
    meta = static[0].metadata if static else {}
    source = str(meta.get("taint_source") or meta.get("taint") or "")
    sink = str(meta.get("sink") or "")
    field_path = str(meta.get("field_path") or "")
    parser = str(meta.get("parser_backend") or meta.get("parser_tier") or "")
    incomplete = str(meta.get("analysis_incomplete") or "")
    relationships: list[str] = []
    for key in ("relationship", "re_export", "class_method", "route", "route_method"):
        value = meta.get(key)
        if value and value not in {"false", ""}:
            relationships.append(f"{key}:{value}")
    if field_path:
        relationships.append(f"field:{field_path}")
    cross_file = str(meta.get("taint_scope") or "") == "cross_file" or "cross_file:" in str(
        meta.get("taint") or ""
    )
    kinds = tuple(sorted({item.kind.value for item in finding.evidence.items}))
    runtime = _runtime_label(finding)
    because = _because(source, sink, relationships, parser, cross_file, field_path, runtime)
    return ConfidenceExplanation(
        summary=because,
        detected_because=because,
        source=source,
        sink=sink,
        relationships=tuple(relationships),
        parser=parser,
        cross_file=cross_file,
        field_path=field_path,
        analysis_incomplete=incomplete,
        runtime_support=runtime,
        evidence_kinds=kinds,
        static_status=finding.status.value,
    )


def correlate_finding(
    finding: SecurityFinding,
    evidence: Sequence[Evidence],
    *,
    peers: Sequence[SecurityFinding] = (),
) -> SecurityFinding:
    """Attach same-issue evidence. Status is unchanged.

    Trusted server attribution may match without a source file. Otherwise the
    evidence needs one normalized location and vulnerability identity, and it
    must not be claimed by a peer. Duplicate items are not repeated.
    """
    accepted: list[Evidence] = []
    seen = {evidence_identity(item) for item in finding.evidence.items}
    for item in sorted(evidence, key=_evidence_sort):
        if not _same_issue(finding, item, peers):
            continue
        key = evidence_identity(item)
        if key in seen:
            continue
        seen.add(key)
        accepted.append(item)
    if not accepted:
        return finding
    merged = finding.evidence.extend(accepted)
    # Status is copied, not promoted. Verification stays on SecurityFinding.verify.
    return replace(finding, evidence=merged, status=finding.status)


def _same_issue(
    finding: SecurityFinding,
    evidence: Evidence,
    peers: Sequence[SecurityFinding] = (),
) -> bool:
    loc = finding.source_location
    path = evidence.artifact_path or _meta_str(evidence, "file_path")
    if path and loc is not None and loc.file_path and not _same_path(loc.file_path, path):
        return False
    parsed = _coerce_line(evidence.metadata.get("line"))
    if parsed is not None and loc is not None and loc.line is not None and parsed != loc.line:
        return False
    sink = _meta_str(evidence, "sink")
    recorded_sink = _finding_sink(finding)
    if sink and recorded_sink and sink != recorded_sink:
        return False
    if _server_attributed(evidence):
        # Unstamped client keys are ignored above. Only a server stamp can
        # name the finding, and a stamp for a different finding does not
        # fall through to location matching.
        return _trusted_identity(finding, evidence)
    if not path or loc is None or not loc.file_path or not _same_path(loc.file_path, path):
        return False
    vuln = _meta_str(evidence, "vulnerability_class")
    if vuln and finding.vulnerability_class and vuln != finding.vulnerability_class:
        return False
    source = _meta_str(evidence, "taint_source") or _meta_str(evidence, "flow_source")
    if source:
        recorded_source = finding.flow_source or _finding_source(finding)
        if recorded_source and source != recorded_source:
            return False
    if parsed is None or loc.line is None or parsed != loc.line:
        return False
    return not any(_competes_for_line(finding, peer, parsed) for peer in peers)


def _server_attributed(evidence: Evidence) -> bool:
    return _meta_str(evidence, "attribution") == "server"


def _trusted_identity(finding: SecurityFinding, evidence: Evidence) -> bool:
    """Exact server finding key, or a server execution already stored on the finding."""
    identity = _meta_str(evidence, "finding_key")
    if identity and finding.finding_key and identity == finding.finding_key:
        return True
    execution = _meta_str(evidence, "execution_id")
    if not execution:
        return False
    return any(_meta_str(item, "execution_id") == execution for item in finding.evidence.items)


def _coerce_line(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
    return None


def _finding_sink(finding: SecurityFinding) -> str:
    if finding.flow_sink:
        return finding.flow_sink
    for item in finding.evidence.items:
        sink = _meta_str(item, "sink")
        if sink:
            return sink
    return ""


def _finding_source(finding: SecurityFinding) -> str:
    if finding.flow_source:
        return finding.flow_source
    for item in finding.evidence.items:
        source = _meta_str(item, "taint_source") or _meta_str(item, "flow_source")
        if source:
            return source
    return ""


def _competes_for_line(finding: SecurityFinding, peer: SecurityFinding, line: int) -> bool:
    loc = peer.source_location
    own = finding.source_location
    if loc is None or own is None or peer is finding:
        return False
    if peer.vulnerability_class != finding.vulnerability_class:
        return False
    if not _same_path(own.file_path, loc.file_path):
        return False
    return loc.line == line


def _same_path(left: str, right: str) -> bool:
    norm_left = left.replace("\\", "/").lstrip("./")
    norm_right = right.replace("\\", "/").lstrip("./")
    return (
        norm_left == norm_right
        or norm_left.endswith("/" + norm_right)
        or norm_right.endswith("/" + norm_left)
    )


def evidence_identity(item: Evidence) -> tuple[str, ...]:
    """Stable evidence identity. Object ids are excluded because reload mints new ones."""
    return observation_identity(item)


def _evidence_sort(item: Evidence) -> tuple[str, str, str, str]:
    return (item.kind.value, item.source, item.summary, item.artifact_path or "")


def _meta_str(item: Evidence, key: str) -> str:
    value = item.metadata.get(key)
    return "" if value is None else str(value)


def _contradicts(item: Evidence) -> bool:
    return evidence_contradicts(item)


def _runtime_label(finding: SecurityFinding) -> str:
    runtime = [item for item in finding.evidence.items if item.kind in _RUNTIME_KINDS]
    if any(_contradicts(item) for item in runtime):
        return "contradicts"
    if runtime:
        return "supports"
    if any(item.kind is EvidenceKind.AI_ANALYSIS for item in finding.evidence.items):
        return "ai_only"
    return "none"


def _because(
    source: str,
    sink: str,
    relationships: list[str],
    parser: str,
    cross_file: bool,
    field_path: str,
    runtime: str,
) -> str:
    parts = ["static analysis"]
    if source:
        parts.append(f"source {source}")
    if sink:
        parts.append(f"sink {sink}")
    if field_path:
        parts.append(f"field {field_path}")
    if cross_file:
        parts.append("cross-file flow")
    if relationships:
        parts.append("relationships " + ", ".join(relationships))
    if parser:
        parts.append(f"parser {parser}")
    if runtime == "supports":
        parts.append("a test or reproduction names the same location")
    elif runtime == "contradicts":
        parts.append(
            "the static result is kept; stored lifecycle state stays; "
            "a later observation says this path was not reached"
        )
    elif runtime == "ai_only":
        parts.append("AI text is attached and is not verification")
    else:
        parts.append("no independent runtime evidence")
    return "; ".join(parts)
