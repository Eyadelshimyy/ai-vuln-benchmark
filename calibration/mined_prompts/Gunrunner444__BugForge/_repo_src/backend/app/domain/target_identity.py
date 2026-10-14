"""Semantic target identity for lifecycle verification.

Line number, formatting whitespace, and parser byte offsets are not part of
the identity. Project scope is. Two equivalent sink calls in one function
stay distinct through sink occurrence and call shape, not through the line.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.domain.evidence import EvidenceKind


def semantic_target_identity(finding: Any, *, project_id: str | None = None) -> str:
    """Stable identity of the code a verification observation was bound to.

    ``project_id`` overrides the finding field so a rescan finding that has
    not yet copied the row's project still hashes the same scope.
    """
    location = getattr(finding, "source_location", None)
    path = ""
    if location is not None and getattr(location, "file_path", None):
        path = str(location.file_path).replace("\\", "/").lstrip("./")
    project = project_id if project_id is not None else getattr(finding, "project_id", "")
    scope = ""
    argument = ""
    occurrence = ""
    sink_id = ""
    call_identity = ""
    evidence = getattr(finding, "evidence", None)
    items = getattr(evidence, "items", ()) if evidence is not None else ()
    for item in items:
        if getattr(item, "kind", None) is not EvidenceKind.STATIC_ANALYSIS:
            continue
        meta = getattr(item, "metadata", {}) or {}
        scope = scope or _norm(meta.get("scope_id"))
        argument = argument or _norm(meta.get("argument_index"))
        occurrence = occurrence or _norm(meta.get("sink_occurrence"))
        sink_id = sink_id or _norm(meta.get("sink_id"))
        call_identity = call_identity or _norm(meta.get("call_identity"))
    parts = (
        _norm(project),
        path,
        _norm(getattr(finding, "vulnerability_class", "")),
        _norm(getattr(finding, "flow_sink", "")),
        _norm(getattr(finding, "flow_source", "")),
        _norm(getattr(finding, "field_path", "")),
        scope,
        argument,
        occurrence,
        sink_id,
        call_identity,
    )
    payload = json.dumps(parts, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _norm(value: object) -> str:
    return " ".join(str(value or "").split())
