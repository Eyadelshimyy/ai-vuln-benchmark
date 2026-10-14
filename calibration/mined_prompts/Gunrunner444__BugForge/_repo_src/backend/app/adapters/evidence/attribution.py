"""Server-owned attribution for evidence collected for one finding.

A client-supplied ``finding_key`` is not trusted. BugForge stamps identity
only when it has already loaded the persisted finding for the operation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.domain.evidence import Evidence
from app.domain.findings import SecurityFinding

# Client metadata must not carry lifecycle identity. These fields are removed
# from every untrusted record, including a forged ``attribution=server`` stamp.
_CLIENT_IDENTITY_KEYS = frozenset(
    {
        "attribution",
        "execution_id",
        "finding_id",
        "finding_key",
        "observation_signature",
        "observed_target",
        "project_id",
        "server_observation_id",
    }
)


@dataclass(frozen=True)
class ServerAttribution:
    """Identity copied from a persisted finding by the lifecycle service."""

    finding_key: str
    finding_id: str
    file_path: str
    line: int | None
    vulnerability_class: str
    sink: str
    flow_source: str
    execution_id: str

    @classmethod
    def from_finding(cls, finding: SecurityFinding, execution_id: str) -> ServerAttribution:
        loc = finding.source_location
        return cls(
            finding_key=finding.finding_key,
            finding_id=str(finding.id),
            file_path=loc.file_path if loc else "",
            line=loc.line if loc else None,
            vulnerability_class=finding.vulnerability_class or "",
            sink=finding.flow_sink,
            flow_source=finding.flow_source,
            execution_id=execution_id,
        )


def strip_client_attribution(item: Evidence) -> Evidence:
    """Remove client-supplied lifecycle identity.

    A finding key, finding id, execution id, or ``attribution=server`` marker
    is not trusted because the caller knew the words. The lifecycle service
    stamps those fields only after it has loaded the persisted finding.
    """
    metadata = {
        key: value for key, value in item.metadata.items() if key not in _CLIENT_IDENTITY_KEYS
    }
    return replace(item, metadata=metadata)


def stamp_server_attribution(item: Evidence, attribution: ServerAttribution) -> Evidence | None:
    """Return evidence stamped for this finding, or None when location conflicts.

    ``None`` means the collector observed a different file or line. The caller
    must not attach that item through this finding's execution.
    """
    if _location_conflicts(item, attribution):
        return None
    metadata = {
        key: value
        for key, value in item.metadata.items()
        if key not in _CLIENT_IDENTITY_KEYS and key != "finding_key"
    }
    metadata["attribution"] = "server"
    metadata["finding_key"] = attribution.finding_key
    metadata["finding_id"] = attribution.finding_id
    metadata["execution_id"] = attribution.execution_id
    if attribution.vulnerability_class:
        metadata["vulnerability_class"] = attribution.vulnerability_class
    if attribution.sink:
        metadata["sink"] = attribution.sink
    if attribution.flow_source:
        metadata["taint_source"] = attribution.flow_source
    if attribution.file_path and "file_path" not in metadata:
        metadata["file_path"] = attribution.file_path
    if attribution.line is not None and metadata.get("line") in {None, ""}:
        metadata["line"] = str(attribution.line)
    artifact = item.artifact_path
    return replace(item, metadata=metadata, artifact_path=artifact)


def attribution_metadata(attribution: ServerAttribution | None) -> dict[str, str]:
    """Metadata collectors may copy when the service passed a server stamp."""
    if attribution is None:
        return {}
    data = {
        "attribution": "server",
        "finding_key": attribution.finding_key,
        "finding_id": attribution.finding_id,
        "execution_id": attribution.execution_id,
        "vulnerability_class": attribution.vulnerability_class,
        "sink": attribution.sink,
        "taint_source": attribution.flow_source,
    }
    if attribution.file_path:
        data["file_path"] = attribution.file_path
    if attribution.line is not None:
        data["line"] = str(attribution.line)
    return {key: value for key, value in data.items() if value}


def _location_conflicts(item: Evidence, attribution: ServerAttribution) -> bool:
    path = item.artifact_path or _text(item.metadata.get("file_path"))
    if path and attribution.file_path and not _same_path(path, attribution.file_path):
        return True
    parsed = _line(item.metadata.get("line"))
    return parsed is not None and attribution.line is not None and parsed != attribution.line


def _same_path(left: str, right: str) -> bool:
    norm_left = left.replace("\\", "/").lstrip("./")
    norm_right = right.replace("\\", "/").lstrip("./")
    return (
        norm_left == norm_right
        or norm_left.endswith("/" + norm_right)
        or norm_right.endswith("/" + norm_left)
    )


def _line(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _text(value: object) -> str:
    return "" if value is None else str(value)
