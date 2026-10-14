"""In-process protocol discovery.

The engine builds a bounded graph from parser-backed semantic programs. It does
not call a model, open a network connection, or mark a finding verified.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.results import DynamicResult
from app.parsing.solidity_ir import SemanticProgram, build_semantic_program
from app.parsing.solidity_protocol import (
    MAX_PROTOCOL_CONTRACTS,
    build_protocol_graph,
    expand_paths_report,
    protocol_evidence,
)

_VERSION = "phase47"


class ProtocolEngine(DiscoveryEngine):
    @property
    def engine_id(self) -> str:
        return "bugforge-protocol"

    @property
    def display_name(self) -> str:
        return "BugForge protocol analysis"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset(
            {EngineCapability.CROSS_CONTRACT_ANALYSIS, EngineCapability.RESULTS_INGESTION}
        )

    def availability(self) -> EngineAvailability:
        return EngineAvailability.AVAILABLE

    def version(self) -> str:
        return _VERSION

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        return _analyze(self, request)

    def _analyze_target(self, request: AnalysisRequest) -> DynamicResult:
        return _analyze(self, request)


def _analyze(engine: ProtocolEngine, request: AnalysisRequest) -> DynamicResult:
    loaded = _load_programs(request)
    if loaded is None:
        return DynamicResult(
            engine=engine.engine_id,
            engine_version=engine.version(),
            language=request.language,
            target=request.target,
            status=ResultStatus.UNSUPPORTED,
            executed=False,
            provenance="protocol-analysis",
            oracle_explanation="no Solidity sources were available to analyze",
            oracle_kind="",
            metadata=_meta(request, status="unsupported", contracts="0", paths="0", observed=False),
        )
    programs, sources = loaded
    graph = build_protocol_graph(
        programs,
        project=request.extra.get("project_id", "") or str(request.repo_root),
        source_snapshot=request.extra.get("source_snapshot", ""),
        compiler_configuration=request.extra.get("compiler_configuration", ""),
        sources=sources,
    )
    expansion = expand_paths_report(graph)
    paths = expansion.paths
    evidence = protocol_evidence(
        graph,
        sequence_id="",
        engines=(engine.engine_id,),
        environment="local",
        uncertainty="candidate",
        paths=paths,
        paths_truncated=expansion.truncated,
    )
    metadata = _meta(
        request,
        status="candidate",
        contracts=str(len(graph.nodes)),
        paths=str(len(paths)),
        observed=True,
    )
    metadata["path_ids"] = json.dumps(list(evidence.path_ids))
    metadata["paths_truncated"] = str(expansion.truncated).lower()
    return DynamicResult(
        engine=engine.engine_id,
        engine_version=engine.version(),
        language=request.language,
        target=request.target,
        campaign_id=request.campaign_id,
        source_file=request.source_file,
        contract=request.contract,
        function=request.function,
        status=ResultStatus.INGESTED,
        executed=False,
        provenance="protocol-analysis",
        oracle_explanation="protocol candidates are not verification",
        oracle_kind="protocol",
        metadata=metadata,
        protocol_evidence=evidence,
    )


def _meta(
    request: AnalysisRequest, *, status: str, contracts: str, paths: str, observed: bool
) -> dict[str, str]:
    return {
        "verified": "false",
        "status": status,
        "capability": "cross_contract_analysis",
        "evidence_class": "protocol" if observed else "",
        "project": request.extra.get("project_id", ""),
        "source_snapshot": request.extra.get("source_snapshot", ""),
        "compiler_configuration": request.extra.get("compiler_configuration", ""),
        "contracts": contracts,
        "paths": paths,
        "llm_invoked": "false",
    }


def _load_programs(
    request: AnalysisRequest,
) -> tuple[tuple[SemanticProgram, ...], dict[str, str]] | None:
    from app.parsing.engine import parse_source

    names = request.files or ((request.source_file,) if request.source_file else ())
    programs = []
    sources: dict[str, str] = {}
    root = request.repo_root.resolve()
    for name in names[:MAX_PROTOCOL_CONTRACTS]:
        if not name or name.endswith("/"):
            continue
        path = Path(name)
        if not path.is_absolute():
            path = request.repo_root / name
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.suffix != ".sol" or not resolved.is_file():
            continue
        try:
            text = resolved.read_text(encoding="utf-8")
            program = build_semantic_program(parse_source("solidity", resolved, text))
        except (OSError, UnicodeError, ValueError, RuntimeError):
            return None
        programs.append(program)
        sources[program.file] = text
    if not programs:
        return None
    return tuple(programs), sources
