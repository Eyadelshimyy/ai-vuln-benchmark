"""Evidence graph: finding → hypothesis → evidence → tool → request/source/reproduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.security_testing.errors import RestrictedActivityError
from app.security_testing.secrets import redact_text

ALLOWED_RELATIONS = frozenset(
    {
        "supports",
        "contradicted_by",
        "executes",
        "observes",
        "produced",
        "responds_to",
        "validates",
        "derived_from",
        "same_issue_as",
        "identity_diff",
        "follows",
        "motivates",
        "produces",
    }
)


class GraphIntegrityError(RestrictedActivityError):
    """Malformed evidence-graph mutation."""


@dataclass
class EvidenceNode:
    id: str
    kind: str
    provenance: str
    summary: str
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    project_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "provenance": self.provenance,
            "summary": self.summary,
            "source": self.source,
            "extra": dict(self.extra),
            "session_id": self.session_id,
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class EvidenceGraph:
    nodes: dict[str, EvidenceNode] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    session_id: str = ""
    project_id: str = ""

    def add(
        self,
        *,
        kind: str,
        provenance: str,
        summary: str,
        source: str = "",
        extra: dict[str, Any] | None = None,
        parent_id: str | None = None,
        relation: str = "supports",
        node_id: str | None = None,
        created_at: datetime | None = None,
    ) -> EvidenceNode:
        ident = node_id or uuid4().hex
        if ident in self.nodes:
            raise GraphIntegrityError(f"duplicate_node:{ident}")
        if relation not in ALLOWED_RELATIONS:
            raise GraphIntegrityError(f"unknown_relation:{relation}")
        node = EvidenceNode(
            id=ident,
            kind=kind,
            provenance=provenance,
            summary=redact_text(summary)[:4000],
            source=source,
            extra=_sanitize_extra(extra or {}),
            session_id=self.session_id,
            project_id=self.project_id,
            created_at=created_at or datetime.now(UTC),
        )
        self.nodes[node.id] = node
        if parent_id:
            self.link(parent_id, node.id, relation)
        return node

    def link(self, source_id: str, dest_id: str, relation: str = "supports") -> None:
        if relation not in ALLOWED_RELATIONS:
            raise GraphIntegrityError(f"unknown_relation:{relation}")
        if source_id == dest_id:
            raise GraphIntegrityError("self_edge_forbidden")
        if source_id not in self.nodes or dest_id not in self.nodes:
            raise GraphIntegrityError("invalid_node_reference")
        edge = (source_id, dest_id, relation)
        if edge in self.edges:
            raise GraphIntegrityError("duplicate_edge")
        self.edges.append(edge)

    def get(
        self, node_id: str, *, session_id: str | None = None, project_id: str | None = None
    ) -> EvidenceNode | None:
        node = self.nodes.get(node_id)
        if node is None:
            return None
        if session_id and node.session_id and node.session_id != session_id:
            return None
        if project_id and node.project_id and node.project_id != project_id:
            return None
        return node

    def children(self, node_id: str) -> list[EvidenceNode]:
        out: list[EvidenceNode] = []
        for src, dst, _rel in self.edges:
            if src == node_id and dst in self.nodes:
                out.append(self.nodes[dst])
        return out

    def parents(self, node_id: str) -> list[EvidenceNode]:
        out: list[EvidenceNode] = []
        for src, dst, _rel in self.edges:
            if dst == node_id and src in self.nodes:
                out.append(self.nodes[src])
        return out

    def inspect(self, node_id: str, *, session_id: str, project_id: str) -> dict[str, Any] | None:
        node = self.get(node_id, session_id=session_id, project_id=project_id)
        if node is None:
            return None
        return {
            "id": node.id,
            "kind": node.kind,
            "provenance": node.provenance,
            "summary": node.summary,
            "source": node.source,
            "extra": dict(node.extra),
            "created_at": node.created_at.isoformat(),
            "session_id": node.session_id,
            "project_id": node.project_id,
            "parents": [item.snapshot() for item in self.parents(node.id)],
            "children": [item.snapshot() for item in self.children(node.id)],
            "relations": [
                {"from": src, "to": dst, "relation": rel}
                for src, dst, rel in self.edges
                if src == node.id or dst == node.id
            ],
        }

    def why(self, node_id: str) -> list[dict[str, Any]]:
        """Explain why BugForge believes a node, following inbound edges."""
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        current = [node_id]
        while current:
            nxt: list[str] = []
            for ident in current:
                if ident in seen:
                    continue
                seen.add(ident)
                node = self.nodes.get(ident)
                if node is None:
                    continue
                chain.append(
                    {
                        "id": node.id,
                        "kind": node.kind,
                        "provenance": node.provenance,
                        "summary": node.summary,
                        "source": node.source,
                    }
                )
                for src, dst, _rel in self.edges:
                    if dst == ident and src != ident:
                        nxt.append(src)
            current = nxt
        return chain

    def relevant(self, *, limit: int = 12, kinds: tuple[str, ...] = ()) -> list[EvidenceNode]:
        items = list(self.nodes.values())
        if kinds:
            items = [node for node in items if node.kind in kinds]
        items.sort(key=lambda node: node.created_at, reverse=True)
        return items[:limit]

    def snapshot(self) -> dict[str, Any]:
        nodes = [node.snapshot() for node in sorted(self.nodes.values(), key=lambda item: item.id)]
        edges = [{"from": src, "to": dst, "relation": rel} for src, dst, rel in sorted(self.edges)]
        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_snapshot(
        cls,
        payload: dict[str, Any] | None,
        *,
        session_id: str = "",
        project_id: str = "",
    ) -> EvidenceGraph:
        graph = cls(session_id=session_id, project_id=project_id)
        if not payload:
            return graph
        for raw in payload.get("nodes") or []:
            if not isinstance(raw, dict):
                raise GraphIntegrityError("malformed_node")
            created = raw.get("created_at")
            created_at = None
            if isinstance(created, str) and created:
                try:
                    created_at = datetime.fromisoformat(created)
                except ValueError:
                    created_at = None
            ident = str(raw.get("id") or "")
            if not ident:
                raise GraphIntegrityError("missing_node_id")
            graph.add(
                kind=str(raw.get("kind") or "observation"),
                provenance=str(raw.get("provenance") or ""),
                summary=str(raw.get("summary") or ""),
                source=str(raw.get("source") or ""),
                extra=dict(raw.get("extra") or {}),
                node_id=ident,
                created_at=created_at,
            )
            node = graph.nodes[ident]
            node.session_id = str(raw.get("session_id") or session_id)
            node.project_id = str(raw.get("project_id") or project_id)
        for raw in payload.get("edges") or []:
            if not isinstance(raw, dict):
                raise GraphIntegrityError("malformed_edge")
            src = str(raw.get("from") or raw.get("source") or "")
            dst = str(raw.get("to") or raw.get("destination") or "")
            rel = str(raw.get("relation") or "supports")
            graph.link(src, dst, rel)
        return graph


def _sanitize_extra(extra: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in extra.items():
        if key.lower() in {"authorization", "cookie", "token", "secret", "password"}:
            continue
        if isinstance(value, str):
            clean[key] = redact_text(value)[:2000]
        elif isinstance(value, (int, float, bool)) or value is None:
            clean[key] = value
        elif isinstance(value, list):
            clean[key] = [redact_text(str(item))[:500] for item in value[:40]]
        elif isinstance(value, dict):
            clean[key] = _sanitize_extra(value)
        else:
            clean[key] = redact_text(str(value))[:500]
    return clean
