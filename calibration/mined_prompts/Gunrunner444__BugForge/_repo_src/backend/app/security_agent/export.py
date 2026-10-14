"""Deterministic sanitized evidence export packages.

Hash semantics
--------------
``hashes.sha256`` is the SHA-256 digest of the canonical JSON encoding of the
package *without* the ``hashes`` field. Canonical encoding means:

* dictionaries serialized with ``sort_keys=True``
* sets emitted as sorted lists
* graph nodes ordered by id
* graph edges ordered by (from, to, relation)
* evidence, hypotheses, tool results, and timeline ordered by id/timestamp
* list order is preserved only where it is semantically recorded (timeline,
  tool history). Sets and mappings are always sorted.

Nondeterministic runtime values are included only when they were part of the
recorded artifact (for example an evidence node's ``created_at``). Secrets are
redacted before hashing. Replay-generated evidence is labelled ``provenance=replay``
and is not live verification evidence.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from app.domain.findings import SecurityFinding
from app.security_agent.agent import ResearchSession
from app.security_testing.secrets import redact_text


class FindingNotFoundError(LookupError):
    """Requested finding does not belong to the session."""


def export_package(
    session: ResearchSession,
    finding: SecurityFinding | None = None,
    *,
    finding_id: str | None = None,
) -> dict[str, Any]:
    selected = finding
    if finding_id:
        selected = next(
            (item for item in session.findings if str(item.id) == str(finding_id)),
            None,
        )
        if selected is None:
            raise FindingNotFoundError(str(finding_id))
    graph = _canonical(session.graph.snapshot())
    hypotheses = [_canonical(item.snapshot()) for item in _sorted_hypotheses(session)]
    timeline = [_canonical(item.snapshot()) for item in _sorted_timeline(session)]
    tool_results = [
        _canonical(item.snapshot() if hasattr(item, "snapshot") else {"tool": item})
        for item in _sorted_tools(session)
    ]
    nodes = list((graph.get("nodes") or []) if isinstance(graph, dict) else [])
    reproductions = [node for node in nodes if node.get("kind") == "reproduction"]
    sources = [node for node in nodes if node.get("kind") == "source"]
    requests = [node for node in nodes if node.get("kind") in {"request", "tool_request"}]
    responses = [node for node in nodes if node.get("kind") == "response"]
    package = {
        "finding": None if selected is None else _finding_payload(selected),
        "findings": [_finding_payload(item) for item in _sorted_findings(session)],
        "hypotheses": hypotheses,
        "evidence": nodes,
        "evidence_graph": graph,
        "tool_executions": tool_results,
        "http_exchanges": _canonical(_redact_exchanges(session)),
        "requests": requests,
        "responses": responses,
        "reproductions": reproductions,
        "source_references": sources,
        "tool_results": tool_results,
        "timeline": timeline,
        "timestamps": {
            "session_created": session.created_at.isoformat(),
        },
        "scope": _canonical(
            {
                "program": session.program_handle,
                "mode": session.mode.value,
                "target": session.target,
                "project_id": session.project_id,
                "lab_mode": session.engine.session.scope.lab_mode,
                "includes": [
                    {
                        "identifier": rule.identifier,
                        "asset_type": getattr(rule.asset_type, "value", str(rule.asset_type)),
                        "structured_scope_id": rule.structured_scope_id,
                    }
                    for rule in _sorted_rules(session.engine.session.scope.includes)
                ],
            }
        ),
        "scope_snapshot": _canonical(
            {
                "program_id": session.engine.session.scope.program_id,
                "program_name": session.engine.session.scope.program_name,
                "includes": [rule.identifier for rule in session.engine.session.scope.includes],
                "excludes": [rule.identifier for rule in session.engine.session.scope.excludes],
            }
        ),
    }
    package = _canonical(package)
    if not isinstance(package, dict):
        raise TypeError("export package must be a mapping")
    raw = json.dumps(package, sort_keys=True, separators=(",", ":"), default=str)
    package["hashes"] = {
        "sha256": sha256(raw.encode("utf-8")).hexdigest(),
        "canonicalization": (
            "SHA-256 of canonical JSON (sorted keys, sorted graph nodes/edges, "
            "redacted secrets) excluding this hashes object."
        ),
    }
    return package


def _finding_payload(finding: SecurityFinding) -> dict[str, Any]:
    evidence_items = [
        {
            "id": str(item.id),
            "kind": item.kind.value,
            "provenance": item.provenance.value if item.provenance is not None else "",
            "summary": redact_text(item.summary),
            "source": item.source,
        }
        for item in sorted(finding.evidence.items, key=lambda item: str(item.id))
    ]
    payload = _canonical(
        {
            "id": str(finding.id),
            "title": finding.title,
            "status": finding.status.value,
            "target": finding.target,
            "vulnerability_class": finding.vulnerability_class,
            "hypothesis": finding.hypothesis,
            "confidence": finding.confidence,
            "impact": finding.impact,
            "reproduction": finding.reproduction,
            "observation_refs": list(finding.observation_refs),
            "evidence": evidence_items,
        }
    )
    if not isinstance(payload, dict):
        return {}
    return payload


def _redact_exchanges(session: ResearchSession) -> list[dict[str, Any]]:
    items = []
    for key in sorted(session.exchanges):
        payload = session.exchanges[key]
        items.append(json.loads(redact_text(json.dumps(payload, default=str, sort_keys=True))))
    return items


def _sorted_hypotheses(session: ResearchSession) -> list[Any]:
    return sorted(session.hypotheses, key=lambda item: item.id)


def _sorted_findings(session: ResearchSession) -> list[SecurityFinding]:
    return sorted(session.findings, key=lambda item: str(item.id))


def _sorted_timeline(session: ResearchSession) -> list[Any]:
    return sorted(session.timeline, key=lambda item: (item.created_at.isoformat(), item.event_type))


def _sorted_tools(session: ResearchSession) -> list[Any]:
    records = getattr(session, "tool_call_records", session.tool_history)
    if records and hasattr(records[0], "id"):
        return sorted(records, key=lambda item: getattr(item, "id", ""))
    return list(records)


def _sorted_rules(rules: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(
        sorted(rules, key=lambda rule: (rule.identifier, str(rule.structured_scope_id or "")))
    )


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical(value[key]) for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, set):
        return [_canonical(item) for item in sorted(value, key=lambda item: str(item))]
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, list):
        if value and all(isinstance(item, dict) and "id" in item for item in value):
            return [
                _canonical(item) for item in sorted(value, key=lambda item: str(item.get("id")))
            ]
        if value and all(
            isinstance(item, dict) and {"from", "to", "relation"} <= set(item) for item in value
        ):
            return [
                _canonical(item)
                for item in sorted(
                    value,
                    key=lambda item: (
                        str(item.get("from")),
                        str(item.get("to")),
                        str(item.get("relation")),
                    ),
                )
            ]
        return [_canonical(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value
