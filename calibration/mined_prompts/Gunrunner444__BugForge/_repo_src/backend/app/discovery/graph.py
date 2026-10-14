"""Record a discovery chain on the existing evidence graph."""

from __future__ import annotations

from app.discovery.results import DynamicResult
from app.security_agent.evidence_graph import EvidenceGraph
from app.security_agent.states import EvidenceGraphKind


def record_discovery_chain(
    graph: EvidenceGraph,
    *,
    summary: str,
    result: DynamicResult,
    hypothesis_id: str = "",
) -> str:
    hypothesis = hypothesis_id
    if not hypothesis:
        hypothesis = graph.add(
            kind=EvidenceGraphKind.HYPOTHESIS.value,
            provenance="static_finding",
            summary=summary,
            source="bugforge-static",
        ).id
    target = graph.add(
        kind=EvidenceGraphKind.FUZZ_TARGET.value,
        provenance="static_finding",
        summary=result.target or summary,
        source=result.engine,
        parent_id=hypothesis,
        relation="motivates",
    )
    campaign = graph.add(
        kind=EvidenceGraphKind.FUZZ_CAMPAIGN.value,
        provenance=result.provenance or result.engine,
        summary=f"{result.engine}:{result.status.value}",
        source=result.engine,
        parent_id=target.id,
        relation="executes",
        extra={"executed": result.executed, "verified": False},
    )
    execution = graph.add(
        kind=EvidenceGraphKind.EXECUTION.value,
        provenance=result.provenance or result.engine,
        summary=f"{result.status.value} exit={result.exit_code}",
        source=result.engine,
        parent_id=campaign.id,
        relation="executes",
        extra={
            "verified": False,
            "engine_version": result.engine_version,
            "contract": result.contract,
            "function": result.function,
            "status": result.status.value,
            "campaign_id": result.campaign_id,
        },
    )
    if result.seed_id:
        graph.add(
            kind=EvidenceGraphKind.SEED.value,
            provenance=result.provenance or result.engine,
            summary=result.seed_id,
            source=result.engine,
            parent_id=execution.id,
            relation="derived_from",
            extra={"seed_id": result.seed_id, "verified": False},
        )
    if result.minimized_input:
        graph.add(
            kind=EvidenceGraphKind.COUNTEREXAMPLE.value,
            provenance=result.provenance or result.engine,
            summary=result.minimized_input[:180],
            source=result.engine,
            parent_id=execution.id,
            relation="produces",
            extra={"verified": False},
        )
    if result.oracle_explanation:
        graph.add(
            kind=EvidenceGraphKind.ORACLE.value,
            provenance="oracle",
            summary=result.oracle_explanation[:240],
            source=result.engine,
            parent_id=execution.id,
            relation="validates",
            extra={"verified": False, "oracle_kind": result.oracle_kind},
        )
    if result.reproduction_command:
        graph.add(
            kind=EvidenceGraphKind.REPRODUCTION.value,
            provenance=result.provenance or result.engine,
            summary=result.reproduction_command[:240],
            source=result.engine,
            parent_id=execution.id,
            relation="validates",
            extra={"verified": False},
        )
    return campaign.id
