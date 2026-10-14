"""Server-issued observations.

``Evidence(...)`` is never trusted. ``issue_server_observation`` generates
the observation id and HMAC with the server secret. A caller cannot opt in
by setting ``attribution=server`` or any other metadata flag.

The signature covers the canonical event and the finding binding. Human
summary text is included only in its whitespace-normalized form. Extra
metadata such as operator notes is not signed and cannot create trust.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from uuid import uuid4

from app.core.config import settings
from app.domain.evidence import Evidence, EvidenceKind
from app.domain.target_identity import semantic_target_identity

_DROPPED = frozenset(
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

# Fields that decide identity or lifecycle. Arbitrary notes are omitted.
_SIGNED_META = (
    "argument_index",
    "check_id",
    "contradicts",
    "event",
    "execution_id",
    "field_path",
    "finding_id",
    "finding_key",
    "line",
    "method",
    "observed_target",
    "outcome",
    "project_id",
    "reached",
    "reproduced",
    "result_id",
    "route",
    "rule",
    "rule_id",
    "scope_id",
    "session_id",
    "sink",
    "sink_occurrence",
    "status",
    "status_code",
    "target",
    "url",
)


def issue_server_observation(
    item: Evidence,
    *,
    execution_id: str,
    observed_target: str,
    finding_id: str,
    finding_key: str = "",
    project_id: str = "",
) -> ServerObservation:
    """Issue one observation. Identity fields come from the arguments, not the item."""
    if not execution_id.strip():
        raise ValueError("A server observation requires a server-generated execution id")
    if not observed_target.strip():
        raise ValueError("A server observation must be bound to a semantic target")
    if not finding_id.strip():
        raise ValueError("A server observation must be bound to a finding")
    metadata = {key: value for key, value in item.metadata.items() if key not in _DROPPED}
    metadata["attribution"] = "server"
    metadata["execution_id"] = execution_id
    metadata["observed_target"] = observed_target
    metadata["finding_id"] = finding_id
    metadata["finding_key"] = finding_key
    metadata["project_id"] = project_id
    observation_id = uuid4().hex
    metadata["server_observation_id"] = observation_id
    signature = _sign(
        item.kind.value,
        item.source,
        item.summary,
        item.details,
        item.artifact_path,
        metadata,
        observation_id,
    )
    metadata["observation_signature"] = signature
    return ServerObservation(
        kind=item.kind,
        source=item.source,
        summary=item.summary,
        details=item.details,
        artifact_path=item.artifact_path,
        metadata=metadata,
        collected_at=item.collected_at,
        server_observation_id=observation_id,
        observation_signature=signature,
    )


def issue_for_finding(
    finding: object,
    item: Evidence,
    execution_id: str,
) -> ServerObservation:
    """Issue an observation bound to this finding, project, and semantic target."""
    project_id = str(getattr(finding, "project_id", "") or "")
    return issue_server_observation(
        item,
        execution_id=execution_id,
        observed_target=semantic_target_identity(finding),
        finding_id=str(getattr(finding, "id", "") or ""),
        finding_key=str(getattr(finding, "finding_key", "") or ""),
        project_id=project_id,
    )


def is_trusted_observation(item: Evidence) -> bool:
    """True only for a signature that matches this process's server secret."""
    if not isinstance(item, ServerObservation):
        return False
    if not item.server_observation_id:
        return False
    actual = str(item.metadata.get("observation_signature") or "")
    expected = item.observation_signature
    if not actual or not expected or len(actual) != len(expected):
        return False
    if not hmac.compare_digest(actual, expected):
        return False
    recomputed = _sign(
        item.kind.value,
        item.source,
        item.summary,
        item.details,
        item.artifact_path,
        item.metadata,
        item.server_observation_id,
    )
    return hmac.compare_digest(actual, recomputed)


def restore_server_observation(item: Evidence) -> Evidence:
    """Rebuild a trusted observation from storage, or leave the record untrusted."""
    observation_id = str(item.metadata.get("server_observation_id") or "")
    signature = str(item.metadata.get("observation_signature") or "")
    if not observation_id or not signature:
        return _without_trust_markers(item)
    try:
        restored = ServerObservation(
            kind=item.kind,
            source=item.source,
            summary=item.summary,
            details=item.details,
            artifact_path=item.artifact_path,
            metadata=dict(item.metadata),
            collected_at=item.collected_at,
            server_observation_id=observation_id,
            observation_signature=signature,
        )
    except ValueError:
        return _without_trust_markers(item)
    if not is_trusted_observation(restored):
        return _without_trust_markers(item)
    return restored


@dataclass(frozen=True)
class ServerObservation(Evidence):
    """Observation issued by BugForge. Construction checks the server signature.

    ``metadata`` is a read-only mapping. Changing a signed field requires a
    new issuance; the previous signature no longer validates.
    """

    server_observation_id: str = ""
    observation_signature: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.server_observation_id:
            raise ValueError("server observation id is required")
        metadata = dict(self.metadata)
        expected = _sign(
            self.kind.value,
            self.source,
            self.summary,
            self.details,
            self.artifact_path,
            metadata,
            self.server_observation_id,
        )
        actual = self.observation_signature or str(metadata.get("observation_signature") or "")
        if len(actual) != len(expected) or not hmac.compare_digest(actual, expected):
            raise ValueError("observation signature was not issued by BugForge")
        metadata["observation_signature"] = expected
        metadata["server_observation_id"] = self.server_observation_id
        object.__setattr__(self, "observation_signature", expected)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


def _sign(
    kind: str,
    source: str,
    summary: str,
    details: str,
    artifact: str | None,
    metadata: Mapping[str, object],
    observation_id: str,
) -> str:
    artifact_path = (artifact or "").replace("\\", "/")
    body = {
        "artifact": artifact_path,
        "details": hashlib.sha256(details.encode("utf-8")).hexdigest(),
        "id": observation_id,
        "kind": kind,
        "source": source.strip(),
        "summary": " ".join(summary.split()),
    }
    for key in _SIGNED_META:
        body[key] = _canon(metadata.get(key))
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(settings.secret_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def _canon(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _without_trust_markers(item: Evidence) -> Evidence:
    """Drop a forged runtime stamp. Keep execution metadata so dedup stays stable.

    Static, source, and reproduction records are not server signatures. Their
    ``observed_target`` is the lifecycle binding and must survive reload.
    Unsigned HTTP and other runtime records lose that field so they cannot
    corroborate or verify.
    """
    from dataclasses import replace

    dropped = {
        "attribution",
        "observation_signature",
        "server_observation_id",
    }
    if item.kind not in {
        EvidenceKind.STATIC_ANALYSIS,
        EvidenceKind.SOURCE_CODE,
        EvidenceKind.REPRODUCTION,
    }:
        dropped.add("observed_target")
    metadata = {key: value for key, value in item.metadata.items() if key not in dropped}
    return replace(item, metadata=metadata)
