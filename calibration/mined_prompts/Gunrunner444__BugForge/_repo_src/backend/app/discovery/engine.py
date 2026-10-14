"""Discovery engine contract. Unsupported operations report that they did not run."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from app.discovery.capabilities import CAPABILITY_ORDER, EngineAvailability, EngineCapability
from app.discovery.corpus import DiscoveryCorpus
from app.discovery.results import DynamicResult, not_implemented_result, unavailable_result


@dataclass(frozen=True)
class AnalysisRequest:
    repo_root: Path
    language: str
    target: str = ""
    files: tuple[str, ...] = ()
    contract: str = ""
    function: str = ""
    source_file: str = ""
    difficult: bool = False
    framework: str = ""
    has_harness: bool = False
    campaign_id: str = ""
    match_test: str = ""
    corpus: DiscoveryCorpus | None = None
    extra: dict[str, str] = field(default_factory=dict)


class DiscoveryEngine(ABC):
    @property
    @abstractmethod
    def engine_id(self) -> str: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset()

    @property
    def supported_targets(self) -> frozenset[str]:
        return frozenset({"repository", "function", "contract"})

    @abstractmethod
    def capabilities(self) -> frozenset[EngineCapability]: ...

    @abstractmethod
    def availability(self) -> EngineAvailability: ...

    def version(self) -> str:
        return ""

    def supports(self, capability: EngineCapability) -> bool:
        return capability in self.capabilities()

    def _guard(
        self, request: AnalysisRequest, capability: EngineCapability, operation: str
    ) -> DynamicResult | None:
        if not self.supports(capability):
            return not_implemented_result(
                self.engine_id, request.language, request.target, operation
            )
        if self.availability() is not EngineAvailability.AVAILABLE:
            return unavailable_result(
                self.engine_id,
                request.language,
                request.target,
                reason=f"{self.display_name} is not installed",
            )
        if (
            request.language
            and self.supported_languages
            and request.language not in self.supported_languages
        ):
            return not_implemented_result(
                self.engine_id,
                request.language,
                request.target,
                f"{operation} for {request.language}",
            )
        return None

    def analyze_target(self, request: AnalysisRequest) -> DynamicResult:
        blocked = self._guard(request, EngineCapability.STATIC_ANALYSIS, "analyze_target")
        return blocked if blocked is not None else self._analyze_target(request)

    def start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        needed = self._campaign_capability(request)
        blocked = self._guard(request, needed, "start_campaign")
        return blocked if blocked is not None else self._start_campaign(request)

    def _campaign_capability(self, request: AnalysisRequest) -> EngineCapability:
        caps = self.capabilities()
        mode = request.extra.get("mode", "")
        if mode == "build" and EngineCapability.BUILD in caps:
            return EngineCapability.BUILD
        if mode == "invariant" and EngineCapability.INVARIANT_TESTING in caps:
            return EngineCapability.INVARIANT_TESTING
        if mode == "fuzz" and EngineCapability.FUZZING in caps:
            return EngineCapability.FUZZING
        if mode == "coverage" and EngineCapability.COVERAGE_FEEDBACK in caps:
            return EngineCapability.COVERAGE_FEEDBACK
        if EngineCapability.SYMBOLIC_EXECUTION in caps and (
            request.difficult or mode == "symbolic"
        ):
            return EngineCapability.SYMBOLIC_EXECUTION
        if EngineCapability.TEST_EXECUTION in caps:
            return EngineCapability.TEST_EXECUTION
        if EngineCapability.PROPERTY_TESTING in caps:
            return EngineCapability.PROPERTY_TESTING
        if EngineCapability.INVARIANT_TESTING in caps:
            return EngineCapability.INVARIANT_TESTING
        if EngineCapability.FUZZING in caps:
            return EngineCapability.FUZZING
        if EngineCapability.SYMBOLIC_EXECUTION in caps:
            return EngineCapability.SYMBOLIC_EXECUTION
        if EngineCapability.ECONOMIC_SIMULATION in caps:
            return EngineCapability.ECONOMIC_SIMULATION
        if EngineCapability.CROSS_CONTRACT_ANALYSIS in caps:
            return EngineCapability.CROSS_CONTRACT_ANALYSIS
        if mode == "fork" and EngineCapability.FORK_VALIDATION in caps:
            return EngineCapability.FORK_VALIDATION
        if mode in {"differential", "replay"} and EngineCapability.DIFFERENTIAL_VALIDATION in caps:
            return EngineCapability.DIFFERENTIAL_VALIDATION
        if EngineCapability.RUNTIME_VALIDATION in caps:
            return EngineCapability.RUNTIME_VALIDATION
        return EngineCapability.FUZZING

    def selected_capability(self, request: AnalysisRequest) -> EngineCapability:
        """The capability the scheduler will actually exercise for this request.

        It matches the operation `analyze_target` or `start_campaign` runs and it
        never depends on set iteration order.
        """
        caps = self.capabilities()
        if EngineCapability.STATIC_ANALYSIS in caps:
            return EngineCapability.STATIC_ANALYSIS
        chosen = self._campaign_capability(request)
        if chosen in caps:
            return chosen
        for item in CAPABILITY_ORDER:
            if item in caps:
                return item
        return EngineCapability.PLANNING_ONLY

    def collect_results(self, request: AnalysisRequest) -> DynamicResult:
        blocked = self._guard(request, EngineCapability.RESULTS_INGESTION, "collect_results")
        return blocked if blocked is not None else self._collect_results(request)

    def minimize(self, request: AnalysisRequest) -> DynamicResult:
        blocked = self._guard(request, EngineCapability.RESULTS_INGESTION, "minimize")
        return blocked if blocked is not None else self._minimize(request)

    def export_corpus(self, request: AnalysisRequest) -> DynamicResult:
        blocked = self._guard(request, EngineCapability.RESULTS_INGESTION, "export_corpus")
        return blocked if blocked is not None else self._export_corpus(request)

    def stop(self, request: AnalysisRequest) -> DynamicResult:
        if self.availability() is not EngineAvailability.AVAILABLE:
            return unavailable_result(
                self.engine_id,
                request.language,
                request.target,
                reason=f"{self.display_name} is not installed",
            )
        return self._stop(request)

    def _analyze_target(self, request: AnalysisRequest) -> DynamicResult:
        return not_implemented_result(
            self.engine_id, request.language, request.target, "analyze_target"
        )

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        return not_implemented_result(
            self.engine_id, request.language, request.target, "start_campaign"
        )

    def _collect_results(self, request: AnalysisRequest) -> DynamicResult:
        return not_implemented_result(
            self.engine_id, request.language, request.target, "collect_results"
        )

    def _minimize(self, request: AnalysisRequest) -> DynamicResult:
        return not_implemented_result(self.engine_id, request.language, request.target, "minimize")

    def _export_corpus(self, request: AnalysisRequest) -> DynamicResult:
        return not_implemented_result(
            self.engine_id, request.language, request.target, "export_corpus"
        )

    def _stop(self, request: AnalysisRequest) -> DynamicResult:
        return not_implemented_result(self.engine_id, request.language, request.target, "stop")
