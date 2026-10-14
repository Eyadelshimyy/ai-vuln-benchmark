"""Bounty research engines for the Phase 49 orchestrator.

``BugforgeResearchEngine`` runs the Phase 50 semantic analyzers, the compiler advisory
match, and VFCS generation in process. It makes no network call, runs no compiler, and
calls no model. ``BugforgeCompilerDifferentialEngine`` is separate so that its
availability is the availability of an installed compiler: when none exists the
orchestrator sees an unavailable engine and records it, rather than a fabricated result.

Every result is static or bounded local analysis. ``verified`` is always false, and
an empty result is reported as not-found and never as safe.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from app.discovery.bounty.advisories import match_advisories
from app.discovery.bounty.campaign import UNKNOWN, BountyManifest
from app.discovery.bounty.compiler_diff import (
    COMPLETED,
    CompilerBackend,
    HostSolcBackend,
    run_differential,
)
from app.discovery.bounty.priority import prioritize
from app.discovery.bounty.vfcs import MAX_VFCS_CANDIDATES, SequenceIdentity, generate
from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.results import DynamicFinding, DynamicResult, unavailable_result
from app.parsing.solidity_research import (
    MAX_FILE_BYTES,
    MAX_FILES,
    SemanticCandidate,
    build_research_model,
)
from app.parsing.solidity_research_suite import run_suite

C = EngineCapability

RESEARCH_FAMILIES: dict[str, tuple[str, ...]] = {
    C.CALLER_CONTEXT_ANALYSIS.value: ("caller_context",),
    C.ORACLE_QUALITY_ANALYSIS.value: ("oracle_quality",),
    C.PROOF_BINDING_ANALYSIS.value: ("message_binding",),
    C.ACCOUNT_ABSTRACTION_ANALYSIS.value: ("account_abstraction",),
    C.ACCOUNTING_ANALYSIS.value: ("balance_delta", "arithmetic"),
}
RESEARCH_CAPABILITIES = frozenset(
    {
        C.BOUNTY_CONTEXT_ANALYSIS,
        C.COMPILER_ADVISORY_ANALYSIS,
        C.VFCS_GENERATION,
        *(EngineCapability(name) for name in RESEARCH_FAMILIES),
    }
)
MAX_DIRECTORY_FILES = 400
ENGINE_VERSION = "phase51.1"


def load_sources(request: AnalysisRequest) -> dict[str, str]:
    """Read the requested Solidity files under the repository root. Nothing leaves it.

    When the request names no file, the files come from the truncation-aware
    :func:`select_sources` ranking (scope, deployed, proxy, imported, sensitive
    first) instead of the first files in alphabetical order. Truncation is
    reported by the selection; a file that was not read is not implied safe.
    """
    root = request.repo_root.resolve()
    chosen: list[Path] = []
    named = [*request.files, *([request.source_file] if request.source_file else [])]
    for name in named:
        path = (root / name).resolve()
        if path.is_file() and root in path.parents and path.suffix == ".sol":
            chosen.append(path)
    if not chosen and root.is_dir():
        from app.discovery.bounty.source_selection import select_sources

        selection = select_sources(
            root,
            focus_contract=request.contract,
            focus_file=request.source_file,
            limit=min(MAX_FILES, MAX_DIRECTORY_FILES),
        )
        chosen.extend((root / rel).resolve() for rel in selection.selected)
    sources: dict[str, str] = {}
    for path in list(dict.fromkeys(chosen))[:MAX_FILES]:
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            sources[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            continue
    return sources


def _finding(candidate: SemanticCandidate) -> DynamicFinding:
    return DynamicFinding(
        detector_id=candidate.detector,
        title=candidate.title,
        confidence=candidate.confidence,
        contract=candidate.contract,
        function=candidate.function,
        file_path=candidate.file,
        line=candidate.line,
        description=candidate.summary[:300],
        status="potential",
    )


class BugforgeResearchEngine(DiscoveryEngine):
    def __init__(self, manifest: BountyManifest) -> None:
        self.manifest = manifest

    @property
    def engine_id(self) -> str:
        return "bugforge-research"

    @property
    def display_name(self) -> str:
        return "BugForge bounty research analysis"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return RESEARCH_CAPABILITIES

    def availability(self) -> EngineAvailability:
        return EngineAvailability.AVAILABLE

    def version(self) -> str:
        # New results carry the current phase. Evidence recorded by earlier versions
        # keeps the version it was recorded with; it is never rewritten.
        return ENGINE_VERSION

    def _campaign_capability(self, request: AnalysisRequest) -> EngineCapability:
        try:
            named = EngineCapability(request.extra.get("mode", ""))
        except ValueError:
            return C.BOUNTY_CONTEXT_ANALYSIS
        return named if named in RESEARCH_CAPABILITIES else C.BOUNTY_CONTEXT_ANALYSIS

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        capability = self._campaign_capability(request)
        if request.extra.get("program_context", "") != self.manifest.identity_digest():
            return self._refused(request, "the request does not carry this program's context")
        sources = load_sources(request)
        if not sources and capability is not C.BOUNTY_CONTEXT_ANALYSIS:
            return self._refused(request, "no Solidity source was readable for the request")
        if capability is C.BOUNTY_CONTEXT_ANALYSIS:
            return self._context(request, sources)
        if capability is C.COMPILER_ADVISORY_ANALYSIS:
            return self._advisories(request, sources)
        if capability is C.VFCS_GENERATION:
            return self._vfcs(request, sources)
        return self._semantic(request, sources, capability)

    # ---- capabilities -------------------------------------------------------------------

    def _context(self, request: AnalysisRequest, sources: Mapping[str, str]) -> DynamicResult:
        scope = self.manifest.scope_of(contract=request.contract, file=request.source_file)
        ranked = ""
        if sources:
            model = build_research_model(sources)
            result = prioritize(
                model, manifest=self.manifest, candidates=run_suite(model).candidates
            )
            ranked = ",".join(entry.identity for entry in result.ranked[:5])
        return self._result(
            request,
            C.BOUNTY_CONTEXT_ANALYSIS,
            metadata={
                "scope_status": scope.status.value,
                "manifest_gaps": ",".join(self.manifest.gaps()),
                "priority_top": ranked,
                "priority_is": "triage_only",
            },
            explanation="program context and triage priority; not an exploitability statement",
        )

    def _semantic(
        self, request: AnalysisRequest, sources: Mapping[str, str], capability: EngineCapability
    ) -> DynamicResult:
        families = RESEARCH_FAMILIES[capability.value]
        suite = run_suite(build_research_model(sources), families)
        findings = tuple(_finding(c) for c in suite.candidates)
        return self._result(
            request,
            capability,
            findings=findings,
            metadata={
                "families_run": ",".join(suite.families_run),
                "families_skipped": ",".join(suite.families_skipped),
                "applicable": "true" if suite.families_run else "false",
                "truncated": str(suite.truncated).lower(),
            },
            explanation=f"{capability.value}: static candidates only",
        )

    def _advisories(self, request: AnalysisRequest, sources: Mapping[str, str]) -> DynamicResult:
        report = match_advisories(self.manifest.compiler, sources)
        findings = tuple(
            DynamicFinding(
                detector_id=f"compiler_advisory.{m.uid}",
                title=f"Compiler advisory {m.uid}: {m.name}",
                confidence="low",
                contract="",
                function="",
                file_path=(m.trigger_locations[0].split(":")[0] if m.trigger_locations else ""),
                line=0,
                description="; ".join(m.reasons)[:300],
                status="potential",
            )
            for m in report.matches
            if m.status == "applicable_candidate"
        )
        return self._result(
            request,
            C.COMPILER_ADVISORY_ANALYSIS,
            findings=findings,
            metadata={
                "advisories_unknown": str(report.unknown),
                "advisories_considered": str(report.considered),
                "corpus_source": dict(report.corpus_provenance).get("source", ""),
                "compiler_known": str(self.manifest.compiler.known()).lower(),
            },
            explanation="advisory match needs the exact version and pipeline evidence",
        )

    def _vfcs(self, request: AnalysisRequest, sources: Mapping[str, str]) -> DynamicResult:
        model = build_research_model(sources)
        suite = run_suite(model)
        identity = SequenceIdentity(
            campaign_id=request.campaign_id,
            source_snapshot=self.manifest.source_commit,
            compiler_configuration=self.manifest.compiler.fingerprint(),
            fork_reference=self.manifest.fork_reference(),
            program_context=self.manifest.identity_digest(),
        )
        ranked = prioritize(model, manifest=self.manifest, candidates=suite.candidates)
        order = {entry.identity: index for index, entry in enumerate(ranked.ranked)}
        candidates = sorted(
            suite.candidates,
            key=lambda c: (order.get(f"{c.contract}.{c.function}", len(order)), c.detector, c.line),
        )
        built = generate(model, candidates, identity, limit=MAX_VFCS_CANDIDATES, sources=sources)
        ids = [item.sequence_id for item in built.sequences]
        metadata = {
            "vfcs_count": str(len(ids)),
            "vfcs_ids": json.dumps(ids),
            "vfcs_skipped": str(len(built.skipped)),
            "vfcs_truncated": str(built.truncated).lower(),
        }
        if ids:
            metadata["sequence_id"] = ids[0]
        return self._result(
            request,
            C.VFCS_GENERATION,
            metadata=metadata,
            explanation="call-sequence plans from static candidates; plans are not executions",
        )

    # ---- helpers ------------------------------------------------------------------------

    def _result(
        self,
        request: AnalysisRequest,
        capability: EngineCapability,
        *,
        findings: tuple[DynamicFinding, ...] = (),
        metadata: dict[str, str] | None = None,
        explanation: str = "",
    ) -> DynamicResult:
        base = {
            "capability": capability.value,
            "program_context": self.manifest.identity_digest(),
            "source_snapshot": self.manifest.source_commit,
            "compiler_configuration": self.manifest.compiler.fingerprint(),
            "analysis": "static",
            "verified": "false",
        }
        for key, value in request.extra.items():
            if key.startswith("source_selection_") or key.startswith("deployment_"):
                base[key] = value
        return DynamicResult(
            engine=self.engine_id,
            engine_version=self.version(),
            language=request.language,
            target=request.target,
            status=ResultStatus.INGESTED,
            executed=True,
            findings=findings,
            source_file=request.source_file,
            function=request.function,
            contract=request.contract,
            provenance="bugforge_research",
            oracle_explanation=explanation,
            metadata={**base, **(metadata or {})},
        )

    def _refused(self, request: AnalysisRequest, reason: str) -> DynamicResult:
        return DynamicResult(
            engine=self.engine_id,
            language=request.language,
            target=request.target,
            status=ResultStatus.UNSUPPORTED,
            executed=False,
            provenance="refused",
            oracle_explanation=reason,
        )


class BugforgeCompilerDifferentialEngine(DiscoveryEngine):
    def __init__(self, manifest: BountyManifest, backend: CompilerBackend | None = None) -> None:
        self.manifest = manifest
        self.backend: CompilerBackend = backend or HostSolcBackend()

    @property
    def engine_id(self) -> str:
        return "bugforge-compiler-diff"

    @property
    def display_name(self) -> str:
        return "BugForge compiler differential"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset({C.COMPILER_DIFFERENTIAL_VALIDATION})

    def availability(self) -> EngineAvailability:
        return (
            EngineAvailability.AVAILABLE
            if self.backend.available()
            else EngineAvailability.UNAVAILABLE
        )

    def version(self) -> str:
        return self.backend.version() or "unavailable"

    def _campaign_capability(self, request: AnalysisRequest) -> EngineCapability:
        return C.COMPILER_DIFFERENTIAL_VALIDATION

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        if request.extra.get("program_context", "") != self.manifest.identity_digest():
            return DynamicResult(
                engine=self.engine_id,
                language=request.language,
                target=request.target,
                status=ResultStatus.UNSUPPORTED,
                provenance="refused",
                oracle_explanation="the request does not carry this program's context",
            )
        sources = load_sources(request)
        outcome = run_differential(sources, backend=self.backend, expected=self.manifest.compiler)
        if not outcome.executed:
            return unavailable_result(
                self.engine_id, request.language, request.target, reason=outcome.reason
            )
        status = ResultStatus.INGESTED if outcome.status == COMPLETED else ResultStatus.FAILED
        return DynamicResult(
            engine=self.engine_id,
            engine_version=outcome.compiler_version,
            language=request.language,
            target=request.target,
            status=status,
            executed=True,
            source_file=request.source_file,
            function=request.function,
            contract=request.contract,
            provenance="bugforge_compiler_diff",
            oracle_explanation=outcome.reason,
            metadata={
                "capability": C.COMPILER_DIFFERENTIAL_VALIDATION.value,
                "program_context": self.manifest.identity_digest(),
                "source_snapshot": self.manifest.source_commit,
                "compiler_configuration": self.manifest.compiler.fingerprint(),
                "compiler_version": outcome.compiler_version or UNKNOWN,
                "differential_status": outcome.status,
                "bytecode_differences": str(len(outcome.differences)),
                "classification": "bytecode_difference_candidate"
                if outcome.differences
                else "no_bytecode_difference",
                "verified": "false",
            },
        )


def build_bounty_engines(
    manifest: BountyManifest, backend: CompilerBackend | None = None
) -> tuple[DiscoveryEngine, ...]:
    return (BugforgeResearchEngine(manifest), BugforgeCompilerDifferentialEngine(manifest, backend))
