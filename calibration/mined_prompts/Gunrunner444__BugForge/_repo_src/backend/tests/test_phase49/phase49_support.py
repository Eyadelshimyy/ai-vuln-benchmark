"""Scripted engines for Phase 49 tests. They return canned results; nothing here runs a tool."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.corpus import DiscoveryCorpus, SeedSource
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.orchestration import MemoryStore, Orchestrator
from app.discovery.orchestration.gate import DefaultGate
from app.discovery.results import DynamicFinding, DynamicResult
from app.discovery.scheduler import DiscoveryScheduler

C = EngineCapability
Script = Callable[[AnalysisRequest], DynamicResult]

IDENTITY: dict[str, str] = {
    "project_id": "proj",
    "source_snapshot": "snap1",
    "compiler_configuration": "solc-0.8.24",
}


class FakeEngine(DiscoveryEngine):
    def __init__(
        self,
        engine_id: str,
        capabilities: set[EngineCapability],
        script: Script | None = None,
        *,
        available: bool = True,
        languages: frozenset[str] = frozenset({"solidity"}),
    ) -> None:
        self._id = engine_id
        self._caps = frozenset(capabilities)
        self._script = script
        self.available = available
        self._languages = languages
        self.calls: list[AnalysisRequest] = []

    @property
    def engine_id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._id

    @property
    def supported_languages(self) -> frozenset[str]:
        return self._languages

    def capabilities(self) -> frozenset[EngineCapability]:
        return self._caps

    def availability(self) -> EngineAvailability:
        return EngineAvailability.AVAILABLE if self.available else EngineAvailability.UNAVAILABLE

    def _run(self, request: AnalysisRequest) -> DynamicResult:
        self.calls.append(request)
        if self._script is not None:
            return self._script(request)
        return ok(self._id, request)

    def _analyze_target(self, request: AnalysisRequest) -> DynamicResult:
        return self._run(request)

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        return self._run(request)


def ok(engine: str, request: AnalysisRequest, **overrides: object) -> DynamicResult:
    values: dict[str, object] = {
        "engine": engine,
        "language": request.language,
        "target": request.target,
        "status": ResultStatus.EXECUTED,
        "executed": True,
        "contract": request.contract,
        "function": request.function,
    }
    values.update(overrides)
    return DynamicResult(**values)  # type: ignore[arg-type]


def finding(
    detector: str = "reentrancy-eth", contract: str = "Vault", function: str = "withdraw"
) -> DynamicFinding:
    return DynamicFinding(
        detector_id=detector,
        title=detector,
        contract=contract,
        function=function,
        file_path="Vault.sol",
        line=10,
    )


def static_with_finding(request: AnalysisRequest) -> DynamicResult:
    return ok(
        "bugforge-static",
        request,
        status=ResultStatus.INGESTED,
        findings=(finding(),),
        provenance="bugforge_static",
    )


def static_clean(request: AnalysisRequest) -> DynamicResult:
    return ok(
        "bugforge-static", request, status=ResultStatus.INGESTED, provenance="bugforge_static"
    )


def make_request(tmp_path: Path, **extra: str) -> AnalysisRequest:
    return AnalysisRequest(
        repo_root=tmp_path,
        language="solidity",
        target="Vault.withdraw",
        contract="Vault",
        function="withdraw",
        source_file="Vault.sol",
        extra={**IDENTITY, **extra},
    )


def make_scheduler(
    engines: list[DiscoveryEngine], *, max_engines: int = 6, max_rounds: int = 6
) -> DiscoveryScheduler:
    return DiscoveryScheduler(
        engines=tuple(engines),
        max_engines=max_engines,
        max_rounds=max_rounds,
        corpus=DiscoveryCorpus(),
    )


def make_orchestrator(
    engines: list[DiscoveryEngine],
    request: AnalysisRequest,
    *,
    store: MemoryStore | None = None,
    gate: DefaultGate | None = None,
    max_engines: int = 6,
    max_rounds: int = 6,
    **kwargs: object,
) -> Orchestrator:
    scheduler = make_scheduler(engines, max_engines=max_engines, max_rounds=max_rounds)
    return Orchestrator(
        scheduler,
        request,
        store=store,
        gate=gate,
        clock=lambda: "2026-01-01T00:00:00+00:00",
        **kwargs,  # type: ignore[arg-type]
    )


def static_engine(
    script: Script = static_with_finding, engine_id: str = "bugforge-static"
) -> FakeEngine:
    return FakeEngine(engine_id, {C.STATIC_ANALYSIS, C.RESULTS_INGESTION}, script)


def fuzz_engine(script: Script | None = None, engine_id: str = "ityfuzz") -> FakeEngine:
    return FakeEngine(engine_id, {C.FUZZING}, script)


def symbolic_engine(script: Script | None = None) -> FakeEngine:
    return FakeEngine("halmos", {C.SYMBOLIC_EXECUTION}, script)


def add_symbolic_seed(scheduler: DiscoveryScheduler, request: AnalysisRequest) -> None:
    scheduler.corpus.add(
        "counterexample",
        source=SeedSource.SYMBOLIC_EXECUTION,
        reason="counterexample",
        language=request.language,
        target=request.target,
        source_snapshot=request.extra.get("source_snapshot", ""),
        compiler_configuration=request.extra.get("compiler_configuration", ""),
    )
