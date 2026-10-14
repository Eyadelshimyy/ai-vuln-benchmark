"""Scripted runtime, economic, and protocol engines for Phase 49 scenarios."""

from __future__ import annotations

import json

from app.discovery.capabilities import EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest
from app.discovery.results import DynamicResult
from tests.test_phase49.phase49_support import FakeEngine, ok

C = EngineCapability

FORK_EXTRA = {
    "chain_id": "1",
    "fork_block": "19000000",
    "state_snapshot": "state-1",
}
RUNTIME_EXTRA = {
    "runtime": "true",
    "sequence_id": "seq-1",
    "runtime_configuration": "pinned-local",
}


def runtime_result(
    request: AnalysisRequest,
    *,
    mode: str,
    classification: str = "",
    outcome: str = "succeeded",
    observation: str = "candidate",
    executions: int = 1,
) -> DynamicResult:
    metadata = {
        "evidence_class": "differential" if mode == "differential" else "runtime",
        "observation_status": observation,
        "runtime_mode": mode,
        "transaction_outcome": outcome,
        "sequence_id": request.extra.get("sequence_id", ""),
        "state_snapshot": "state-1",
        "transaction_index": "0",
        "executions": str(executions),
        "source_snapshot": request.extra.get("source_snapshot", ""),
        "compiler_configuration": request.extra.get("compiler_configuration", ""),
        "verified": "false",
    }
    if classification:
        metadata["classification"] = classification
    return ok(
        "bugforge-runtime",
        request,
        status=ResultStatus.INGESTED,
        provenance="sandbox",
        oracle_kind="differential" if mode == "differential" else "runtime",
        metadata=metadata,
    )


def runtime_engine(script=None, *, available: bool = True) -> FakeEngine:
    def default(request: AnalysisRequest) -> DynamicResult:
        mode = request.extra.get("mode", "local")
        classification = {
            "differential": "deterministic same result",
            "fork": "deterministic same result",
        }.get(mode, "")
        return runtime_result(request, mode=mode, classification=classification)

    return FakeEngine(
        "bugforge-runtime",
        {
            C.RUNTIME_VALIDATION,
            C.FORK_VALIDATION,
            C.DIFFERENTIAL_VALIDATION,
            C.RESULTS_INGESTION,
        },
        script or default,
        available=available,
    )


def economic_engine(positive: bool = True) -> FakeEngine:
    def script(request: AnalysisRequest) -> DynamicResult:
        return ok(
            "bugforge-economic",
            request,
            status=ResultStatus.INGESTED,
            oracle_kind="economic",
            provenance="economic",
            metadata={
                "bound": "true" if positive else "false",
                "evidence_class": "economic",
                "source_snapshot": request.extra.get("source_snapshot", ""),
                "compiler_configuration": request.extra.get("compiler_configuration", ""),
            },
        )

    return FakeEngine("bugforge-economic", {C.ECONOMIC_SIMULATION}, script)


def protocol_engine(paths: int = 2) -> FakeEngine:
    def script(request: AnalysisRequest) -> DynamicResult:
        ids = [f"path-{index}" for index in range(paths)]
        return ok(
            "bugforge-protocol",
            request,
            status=ResultStatus.INGESTED,
            oracle_kind="protocol",
            provenance="protocol",
            metadata={
                "evidence_class": "protocol",
                "path_ids": json.dumps(ids),
                "source_snapshot": request.extra.get("source_snapshot", ""),
                "compiler_configuration": request.extra.get("compiler_configuration", ""),
            },
        )

    return FakeEngine("bugforge-protocol", {C.CROSS_CONTRACT_ANALYSIS}, script)
