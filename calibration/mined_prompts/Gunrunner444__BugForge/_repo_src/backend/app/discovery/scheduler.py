"""Bounded discovery scheduler. It selects complementary engines; it does not verify."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.discovery.capabilities import (
    CAPABILITY_ORDER,
    EngineAvailability,
    EngineCapability,
    ResultStatus,
)
from app.discovery.corpus import DiscoveryCorpus, SeedSource
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.results import DynamicResult


@dataclass(frozen=True)
class ScheduleDecision:
    engine_id: str
    action: str
    reason: str
    capability: str = ""
    code: str = ""


@dataclass
class SchedulerFeedback:
    stagnating: bool = False
    difficult: bool = False
    new_coverage: bool = False
    crashes: int = 0
    assertion_failures: int = 0
    rounds: int = 0
    transitions: list[str] = field(default_factory=list)
    exercised: list[str] = field(default_factory=list)


# Preferred order. Later engines run only when budget and preconditions allow.
_SOLIDITY_ORDER = (
    "bugforge-static",
    "slither",
    "foundry",
    "echidna",
    "medusa",
    "ityfuzz",
    "halmos",
    "wake",
)
_STATIC_ENGINES = frozenset({"bugforge-static", "slither", "wake"})
_RUNTIME_FRONT = ("ityfuzz", "echidna", "medusa", "foundry", "halmos")
_LANGUAGE_ORDER: dict[str, tuple[str, ...]] = {
    "solidity": _SOLIDITY_ORDER,
    "c": ("bugforge-static", "native-fuzz", "sanitizer"),
    "cpp": ("bugforge-static", "native-fuzz", "sanitizer"),
    "go": ("bugforge-static", "go-test"),
    "rust": ("bugforge-static", "cargo-test"),
    "java": ("bugforge-static", "jazzer", "property"),
    "python": ("bugforge-static", "property", "pytest"),
}


@dataclass
class DiscoveryScheduler:
    engines: tuple[DiscoveryEngine, ...]
    max_engines: int = 3
    max_rounds: int = 2
    corpus: DiscoveryCorpus = field(default_factory=DiscoveryCorpus)
    feedback: SchedulerFeedback = field(default_factory=SchedulerFeedback)
    _coverage_seen: dict[str, float] = field(default_factory=dict)
    engines_started: int = 0

    def select(self, request: AnalysisRequest) -> list[ScheduleDecision]:
        order = _engine_order_for(request, self.feedback)
        by_id = {engine.engine_id: engine for engine in self.engines}
        decisions: list[ScheduleDecision] = []
        selected = 0
        for engine_id in order:
            engine = by_id.get(engine_id)
            if engine is None:
                decisions.append(
                    ScheduleDecision(
                        engine_id, "skip", "engine is not registered", "", "unregistered"
                    )
                )
                continue
            decision = self._decide(engine, request)
            if decision.action == "run" and selected + self.engines_started >= self.max_engines:
                decision = ScheduleDecision(
                    engine.engine_id,
                    "skip",
                    "campaign budget exhausted",
                    decision.capability,
                    "budget",
                )
            decisions.append(decision)
            if decision.action == "run":
                selected += 1
        return decisions

    def run_selected(self, request: AnalysisRequest) -> list[DynamicResult]:
        results: list[DynamicResult] = []
        by_id = {engine.engine_id: engine for engine in self.engines}
        for decision in self.select(request):
            if decision.action != "run":
                continue
            if self.engines_started >= self.max_engines:
                continue
            self.engines_started += 1
            engine = by_id[decision.engine_id]
            if EngineCapability.STATIC_ANALYSIS in engine.capabilities():
                result = engine.analyze_target(request)
            else:
                result = engine.start_campaign(request)
            results.append(result)
            self.note_result(result, request)
        return results

    def note_result(self, result: DynamicResult, request: AnalysisRequest) -> None:
        self.feedback.rounds += 1
        label = _exercised_label(result)
        if label and label not in self.feedback.exercised:
            self.feedback.exercised.append(label)
        increased = self._coverage_increased(result, request)
        if increased is True:
            self.feedback.new_coverage = True
            self.feedback.stagnating = False
            if result.minimized_input:
                self.corpus.add(
                    result.minimized_input,
                    source=SeedSource.COVERAGE,
                    reason="coverage-increasing input",
                    language=request.language,
                    target=request.target,
                    project=str(request.repo_root),
                    source_snapshot=request.extra.get("source_snapshot", ""),
                    compiler_configuration=request.extra.get("compiler_configuration", ""),
                    engine=result.engine,
                    engine_version=result.engine_version,
                    campaign=request.campaign_id,
                )
        elif increased is False and result.executed:
            self.feedback.stagnating = True
        if result.crash:
            self.feedback.crashes += 1
            if result.minimized_input:
                self.corpus.add(
                    result.minimized_input,
                    source=SeedSource.CRASH,
                    reason="crashing input",
                    language=request.language,
                    target=request.target,
                    project=str(request.repo_root),
                    source_snapshot=request.extra.get("source_snapshot", ""),
                    compiler_configuration=request.extra.get("compiler_configuration", ""),
                    engine=result.engine,
                    engine_version=result.engine_version,
                    campaign=request.campaign_id,
                )
        if result.assertion:
            self.feedback.assertion_failures += 1
        if (
            result.engine == "ityfuzz"
            and result.minimized_input
            and result.metadata.get("executable_input") == "verified"
        ):
            self.corpus.add(
                result.minimized_input,
                source=SeedSource.ITYFUZZ,
                reason="verified executable ityfuzz input",
                language=request.language,
                target=request.target,
                project=str(request.repo_root),
                source_snapshot=request.extra.get("source_snapshot", ""),
                compiler_configuration=request.extra.get("compiler_configuration", ""),
                engine=result.engine,
                engine_version=result.engine_version,
                campaign=request.campaign_id,
            )
        if result.minimized_input and result.metadata.get("seed_source") == "symbolic":
            self.corpus.add(
                result.minimized_input,
                source=SeedSource.SYMBOLIC_EXECUTION,
                reason="symbolic counterexample",
                language=request.language,
                target=request.target,
                project=str(request.repo_root),
                source_snapshot=request.extra.get("source_snapshot", ""),
                compiler_configuration=request.extra.get("compiler_configuration", ""),
                engine=result.engine,
                engine_version=result.engine_version,
                campaign=request.campaign_id,
            )
        if (
            self.feedback.rounds >= self.max_rounds
            and self.feedback.stagnating
            and not self.feedback.new_coverage
        ):
            self.feedback.difficult = True

    def _coverage_increased(self, result: DynamicResult, request: AnalysisRequest) -> bool | None:
        """True, false, or unknown. Unknown never counts as new coverage.

        An explicit tool comparison wins. Otherwise a percent is compared only
        with a percent this scheduler already observed for the same engine.
        """
        explicit = result.coverage.get("new_coverage", result.coverage.get("new", ""))
        if explicit in {"true", "false"}:
            return explicit == "true"
        raw = result.coverage.get("coverage_percent") or result.coverage.get("percent")
        if not raw or result.coverage.get("coverage_available") == "false":
            return None
        try:
            current = float(raw)
        except ValueError:
            return None
        previous = self._coverage_seen.get(coverage_key(result, request))
        self._coverage_seen[coverage_key(result, request)] = current
        if previous is None:
            return None
        return current > previous

    def target_from_static(
        self,
        *,
        repo_root: AnalysisRequest,
        file_path: str,
        function: str,
        contract: str = "",
        rule_id: str = "",
    ) -> AnalysisRequest:
        """Turn a static location into the next research request."""
        difficult = self.feedback.stagnating or self.feedback.difficult or repo_root.difficult
        extra = dict(repo_root.extra)
        extra["rule_id"] = rule_id
        extra["source"] = "static_finding"
        return AnalysisRequest(
            repo_root=repo_root.repo_root,
            language=repo_root.language,
            target=function or contract or repo_root.target or file_path,
            files=repo_root.files,
            contract=contract or repo_root.contract,
            function=function or repo_root.function,
            source_file=file_path or repo_root.source_file,
            difficult=difficult,
            framework=repo_root.framework,
            has_harness=repo_root.has_harness,
            campaign_id=repo_root.campaign_id or rule_id,
            match_test=repo_root.match_test,
            corpus=self.corpus,
            extra=extra,
        )

    def _decide(self, engine: DiscoveryEngine, request: AnalysisRequest) -> ScheduleDecision:
        if request.language not in engine.supported_languages and engine.supported_languages:
            return ScheduleDecision(
                engine.engine_id,
                "skip",
                f"does not support {request.language}",
                "",
                "language",
            )
        if engine.engine_id == "wake" and request.extra.get("include_wake") != "true":
            return ScheduleDecision(
                engine.engine_id,
                "skip",
                "Wake stays optional unless this round asks for it",
                EngineCapability.STATIC_ANALYSIS.value,
                "optional",
            )
        if engine.engine_id in {"echidna", "medusa"} and not (
            request.function or request.contract or request.has_harness
        ):
            return ScheduleDecision(
                engine.engine_id,
                "skip",
                "property and coverage fuzzing wait for a contract or harness target",
                EngineCapability.PROPERTY_TESTING.value,
                "target",
            )
        if engine.engine_id == "halmos" and not (request.difficult or self.feedback.difficult):
            return ScheduleDecision(
                engine.engine_id,
                "skip",
                "symbolic testing waits until a target is hard to reach",
                EngineCapability.SYMBOLIC_EXECUTION.value,
                "symbolic_hint",
            )
        if (
            engine.engine_id == "foundry"
            and request.framework != "foundry"
            and not request.has_harness
        ):
            return ScheduleDecision(
                engine.engine_id,
                "skip",
                "no Foundry project or harness was detected",
                EngineCapability.TEST_EXECUTION.value,
                "framework",
            )
        if engine.availability() is not EngineAvailability.AVAILABLE:
            return ScheduleDecision(
                engine.engine_id,
                "skip",
                "not installed",
                "",
                "unavailable",
            )
        capability = capability_for(engine, request)
        return ScheduleDecision(engine.engine_id, "run", "selected", capability.value, "selected")

    def decide(self, engine_id: str, request: AnalysisRequest) -> ScheduleDecision:
        """Apply the scheduler's own preconditions to one named engine. Nothing runs."""
        engine = self._engine(engine_id)
        if engine is None:
            return ScheduleDecision(
                engine_id, "skip", "engine is not registered", "", "unregistered"
            )
        return self._decide(engine, request)

    def _engine(self, engine_id: str) -> DiscoveryEngine | None:
        for engine in self.engines:
            if engine.engine_id == engine_id:
                return engine
        return None

    def run_engine(self, engine_id: str, request: AnalysisRequest) -> DynamicResult:
        """Run exactly one named engine for an orchestrated round.

        The caller has already chosen the capability. A run counts toward
        `max_engines` only when the engine executed, and a spent campaign budget
        refuses here too, so a second caller cannot exceed it.
        """
        engine = self._engine(engine_id)
        if engine is None:
            return DynamicResult(
                engine=engine_id,
                language=request.language,
                target=request.target,
                status=ResultStatus.UNSUPPORTED,
                executed=False,
                provenance="unregistered",
                oracle_explanation="engine is not registered",
            )
        if self.engines_started >= self.max_engines:
            return DynamicResult(
                engine=engine_id,
                language=request.language,
                target=request.target,
                status=ResultStatus.UNSUPPORTED,
                executed=False,
                provenance="budget_exhausted",
                oracle_explanation="campaign budget exhausted",
            )
        if EngineCapability.STATIC_ANALYSIS in engine.capabilities():
            result = engine.analyze_target(request)
        else:
            result = engine.start_campaign(request)
        if result.executed:
            self.engines_started += 1
        self.note_result(result, request)
        return result

    def plan_followup(self, request: AnalysisRequest) -> list[ScheduleDecision]:
        """Choose the next complementary engine from what the last round learned."""
        by_id = {engine.engine_id: engine for engine in self.engines}
        decisions: list[ScheduleDecision] = []
        halmos_already = any(item.startswith("halmos:run") for item in self.feedback.transitions)
        if self.feedback.difficult and "halmos" in by_id and not halmos_already:
            decision = self._decide(by_id["halmos"], request)
            if decision.action == "skip" and self.feedback.difficult:
                decision = ScheduleDecision(
                    "halmos",
                    "run"
                    if by_id["halmos"].availability() is EngineAvailability.AVAILABLE
                    else "skip",
                    "coverage stalled, so symbolic execution is justified",
                    EngineCapability.SYMBOLIC_EXECUTION.value,
                )
            decisions.append(decision)
        if self.corpus.by_source(SeedSource.SYMBOLIC_EXECUTION) and "foundry" in by_id:
            decisions.append(
                ScheduleDecision(
                    "foundry",
                    "run"
                    if by_id["foundry"].availability() is EngineAvailability.AVAILABLE
                    else "skip",
                    "symbolic counterexample becomes a fuzz seed",
                    EngineCapability.FUZZING.value,
                )
            )
        elif self.feedback.stagnating and request.function and "medusa" in by_id:
            decisions.append(
                self._decide(
                    by_id["medusa"],
                    AnalysisRequest(
                        repo_root=request.repo_root,
                        language=request.language,
                        target=request.target,
                        files=request.files,
                        contract=request.contract,
                        function=request.function,
                        source_file=request.source_file,
                        difficult=request.difficult,
                        framework=request.framework,
                        has_harness=True,
                        corpus=self.corpus,
                        extra={**request.extra, "mode": "fuzz"},
                    ),
                )
            )
        for decision in decisions:
            self.feedback.transitions.append(
                f"{decision.engine_id}:{decision.action}:{decision.reason}"
            )
        return decisions

    def run_followup(self, request: AnalysisRequest) -> list[DynamicResult]:
        results: list[DynamicResult] = []
        by_id = {engine.engine_id: engine for engine in self.engines}
        for decision in self.plan_followup(request):
            if decision.action != "run":
                continue
            if self.engines_started >= self.max_engines:
                continue
            self.engines_started += 1
            engine = by_id[decision.engine_id]
            extra = dict(request.extra)
            if decision.engine_id == "foundry" and self.corpus.by_source(
                SeedSource.SYMBOLIC_EXECUTION
            ):
                extra["mode"] = "fuzz"
            elif decision.engine_id == "medusa":
                extra["mode"] = "fuzz"
            targeted = AnalysisRequest(
                repo_root=request.repo_root,
                language=request.language,
                target=request.target,
                files=request.files,
                contract=request.contract,
                function=request.function,
                source_file=request.source_file,
                difficult=True if decision.engine_id == "halmos" else request.difficult,
                framework=request.framework,
                has_harness=request.has_harness,
                match_test=request.match_test or request.function,
                corpus=self.corpus,
                extra=extra,
            )
            if (
                EngineCapability.STATIC_ANALYSIS in engine.capabilities()
                and decision.engine_id != "halmos"
            ):
                result = engine.analyze_target(targeted)
            else:
                result = engine.start_campaign(targeted)
            results.append(result)
            self.note_result(result, targeted)
        return results


def engine_rank(engine_id: str, language: str) -> int:
    """Stable preference index for one engine. Unknown engines sort last."""
    order = _LANGUAGE_ORDER.get(language, ())
    return order.index(engine_id) if engine_id in order else len(order) + 1


def capability_for(engine: DiscoveryEngine, request: AnalysisRequest) -> EngineCapability:
    """Name one capability deterministically, even for a duck-typed engine."""
    chooser = getattr(engine, "selected_capability", None)
    if callable(chooser):
        chosen = chooser(request)
        if isinstance(chosen, EngineCapability):
            return chosen
    declared = engine.capabilities()
    for item in CAPABILITY_ORDER:
        if item in declared:
            return item
    return EngineCapability.PLANNING_ONLY


def coverage_key(result: DynamicResult, request: AnalysisRequest) -> str:
    """Isolate coverage by project, source snapshot, compiler, target, and engine."""
    project = request.extra.get("project_id", "") or str(request.repo_root)
    return "\n".join(
        (
            project,
            request.extra.get("source_snapshot", ""),
            request.extra.get("compiler_configuration", ""),
            str(request.repo_root),
            request.target,
            request.source_file,
            request.function,
            request.contract,
            request.campaign_id,
            result.engine,
            request.extra.get("mode", ""),
        )
    )


def missing_capability(request: AnalysisRequest, feedback: SchedulerFeedback) -> str:
    """Name the capability the current evidence still lacks.

    Selection uses this gap. An engine is not chosen only because it appears
    earlier in the historical list, and this function cannot raise a budget.
    """
    exercised = set(feedback.exercised)
    stalled = feedback.stagnating or feedback.difficult or request.difficult
    if _economic_requested(request) and "economic_simulation" not in exercised:
        return "economic_simulation"
    if _protocol_requested(request) and "cross_contract_analysis" not in exercised:
        return "cross_contract_analysis"
    if _runtime_requested(request) and "runtime_validation" not in exercised:
        return "runtime_validation"
    if _fork_requested(request) and "fork_validation" not in exercised:
        return "fork_validation"
    if _differential_requested(request) and "differential_validation" not in exercised:
        return "differential_validation"
    if stalled and "symbolic_execution" not in exercised:
        return "symbolic_execution"
    static_known = request.extra.get("source") == "static_finding" or bool(request.campaign_id)
    if static_known and "fuzzing" not in exercised:
        return "fuzzing"
    if request.extra.get("property") == "encoded" and "test_execution" not in exercised:
        return "test_execution"
    if "static_analysis" not in exercised:
        return "static_analysis"
    if "fuzzing" not in exercised:
        return "fuzzing"
    return "static_analysis"


def _economic_requested(request: AnalysisRequest) -> bool:
    return request.extra.get("economic") == "true" or request.extra.get("source") == "economic"


def _protocol_requested(request: AnalysisRequest) -> bool:
    return request.extra.get("protocol") == "true" or request.extra.get("source") == "protocol"


def _runtime_requested(request: AnalysisRequest) -> bool:
    return request.extra.get("runtime") == "true" or request.extra.get("source") == "runtime"


def _fork_requested(request: AnalysisRequest) -> bool:
    return request.extra.get("fork") == "true" or request.extra.get("source") == "fork"


def _differential_requested(request: AnalysisRequest) -> bool:
    return (
        request.extra.get("differential") == "true" or request.extra.get("source") == "differential"
    )


_CAPABILITY_LABELS = frozenset(
    {
        "economic_simulation",
        "cross_contract_analysis",
        "runtime_validation",
        "fork_validation",
        "differential_validation",
        "bounty_context_analysis",
        "caller_context_analysis",
        "oracle_quality_analysis",
        "proof_binding_analysis",
        "account_abstraction_analysis",
        "accounting_analysis",
        "compiler_advisory_analysis",
        "compiler_differential_validation",
        "vfcs_generation",
        "static_analysis",
        "fuzzing",
        "symbolic_execution",
        "test_execution",
    }
)


def _exercised_label(result: DynamicResult) -> str:
    """Unavailable, unsupported, and unexecuted failures are not evidence."""
    ran = result.status in {
        ResultStatus.INGESTED,
        ResultStatus.INTERESTING,
        ResultStatus.EXECUTED,
    } or (
        result.executed
        and result.status in {ResultStatus.FAILED, ResultStatus.TIMEOUT, ResultStatus.TOOL_FAILURE}
    )
    if not ran:
        return ""
    named = result.metadata.get("capability", "")
    if named in _CAPABILITY_LABELS:
        return named
    return _capability_label(result.engine)


def _capability_label(engine_id: str) -> str:
    if engine_id == "bugforge-economic":
        return "economic_simulation"
    if engine_id == "bugforge-protocol":
        return "cross_contract_analysis"
    if engine_id == "bugforge-runtime":
        return "runtime_validation"
    if engine_id in {"ityfuzz", "echidna", "medusa"}:
        return "fuzzing"
    if engine_id == "halmos":
        return "symbolic_execution"
    if engine_id == "foundry":
        return "test_execution"
    if engine_id in _STATIC_ENGINES:
        return "static_analysis"
    return ""


def _engine_order_for(request: AnalysisRequest, feedback: SchedulerFeedback) -> tuple[str, ...]:
    base = _LANGUAGE_ORDER.get(request.language, ("bugforge-static",))
    if request.language != "solidity":
        return base
    needed = missing_capability(request, feedback)
    specialized = {
        "economic_simulation": "bugforge-economic",
        "cross_contract_analysis": "bugforge-protocol",
        "runtime_validation": "bugforge-runtime",
        "fork_validation": "bugforge-runtime",
        "differential_validation": "bugforge-runtime",
    }.get(needed, "")
    if specialized:
        return (specialized, *[item for item in base if item != specialized])
    if needed == "test_execution":
        return ("foundry", *[item for item in base if item != "foundry"])
    if needed not in {"fuzzing", "symbolic_execution"}:
        return base
    runtime = [item for item in base if item not in _STATIC_ENGINES]
    front = ("halmos", *_RUNTIME_FRONT) if needed == "symbolic_execution" else _RUNTIME_FRONT
    ordered: list[str] = []
    for engine_id in (*front, *runtime):
        if engine_id not in ordered and engine_id in runtime:
            ordered.append(engine_id)
    ordered.extend(item for item in base if item in _STATIC_ENGINES)
    return tuple(ordered)


def campaign_stopped(results: list[DynamicResult], *, limit: int) -> bool:
    executed = [item for item in results if item.executed]
    return len(executed) >= limit or any(
        item.status is ResultStatus.FAILED for item in executed[-1:]
    )
