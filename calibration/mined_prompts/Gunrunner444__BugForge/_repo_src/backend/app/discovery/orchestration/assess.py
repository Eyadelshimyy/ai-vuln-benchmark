"""Capability need assessment.

Needs come from evidence the campaign already holds and from what the caller
requested. A weaker engine never raises a need on its own, and an exercised
capability stops being a need unless new evidence justifies a follow-up.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from app.discovery.orchestration.model import (
    ACTIONABLE,
    CapabilityNeed,
    EvidenceItem,
    EvidenceQuality,
    NeedStatus,
    ResearchState,
    Uncertainty,
)

W_CONTRADICTION = 80
W_FUZZ_CANDIDATE = 70
W_SEED_FOLLOWUP = 65
W_SYMBOLIC = 60
W_TEST = 60
W_FUZZ_TARGET = 55
W_RUNTIME_FOLLOWUP = 55
W_DIVERGENCE = 58
W_PROTOCOL = 50
W_ECONOMIC = 50
W_FORK = 45
W_DIFFERENTIAL = 45
W_BASELINE = 100
W_SUGGESTION = 10

_CROSS_CONTRACT_FAMILIES = frozenset(
    {"external_call", "reentrancy", "authorization", "caller_context"}
)
_RESEARCH_FAMILIES = frozenset(
    {
        "caller_context",
        "oracle_quality",
        "message_binding",
        "account_abstraction",
        "balance_delta",
        "arithmetic",
    }
)

# Bounty research needs. Each runs once; none is raised by a weaker engine.
W_BOUNTY_CONTEXT = 95
W_CALLER_CONTEXT = 78
W_ORACLE = 62
W_PROOF = 61
W_ACCOUNTING = 59
W_VFCS = 57
W_COMPILER_ADVISORY = 52
W_COMPILER_DIFF = 44
W_ACCOUNT_ABSTRACTION = 40


@dataclass(frozen=True)
class AssessContext:
    """Read-only facts about the request. The assessor cannot change any of them."""

    language: str = ""
    extra: Mapping[str, str] = field(default_factory=dict)
    difficult: bool = False
    has_harness: bool = False
    framework: str = ""
    targeted: bool = False

    def flag(self, name: str) -> bool:
        return self.extra.get(name) == "true" or self.extra.get("source") == name


def stalled(state: ResearchState) -> bool:
    """The last exploratory round added nothing. Unknown history is not stalled."""
    for record in reversed(state.decisions):
        if record.selected_capability in {"fuzzing", "test_execution"} and (
            record.result_status and record.phase.value == "completed"
        ):
            return not record.evidence_delta.informative
    return False


class _Builder:
    def __init__(self, exercised: set[str]) -> None:
        self.exercised = exercised
        self.weight: dict[str, int] = {}
        self.status: dict[str, NeedStatus] = {}
        self.reasons: dict[str, set[str]] = {}
        self.uncertainties: dict[str, set[str]] = {}

    def add(
        self,
        capability: str,
        status: NeedStatus,
        weight: int,
        reason: str,
        uncertainty: Uncertainty | None = None,
    ) -> None:
        if capability not in self.weight or weight >= self.weight[capability]:
            self.weight[capability] = weight
            self.status[capability] = status
        self.reasons.setdefault(capability, set()).add(reason)
        unc = self.uncertainties.setdefault(capability, set())
        if uncertainty is not None:
            unc.add(uncertainty.value)

    def need(
        self,
        capability: str,
        weight: int,
        uncertainty: Uncertainty,
        reason: str,
        *,
        followup: bool = False,
    ) -> None:
        """An exercised capability stays satisfied unless new evidence makes this a follow-up."""
        if capability in self.exercised and not followup:
            self.add(capability, NeedStatus.SATISFIED, 0, f"{capability} was exercised")
        elif capability in self.exercised:
            self.add(capability, NeedStatus.HIGH_VALUE_FOLLOWUP, weight, reason, uncertainty)
        else:
            self.add(capability, NeedStatus.MISSING, weight, reason, uncertainty)

    def build(self) -> tuple[CapabilityNeed, ...]:
        built = [
            CapabilityNeed(
                capability=name,
                status=self.status[name],
                weight=self.weight[name],
                reasons=tuple(sorted(self.reasons[name])),
                uncertainties=tuple(sorted(self.uncertainties[name])),
            )
            for name in self.weight
        ]
        return tuple(sorted(built, key=lambda need: (-need.weight, need.capability)))


def assess(state: ResearchState, ctx: AssessContext) -> tuple[CapabilityNeed, ...]:
    exercised = set(state.exercised_capabilities)
    builder = _Builder(exercised)
    evidence = [item for item in state.evidence if item.attrs.get("identity") != "mismatch"]
    candidates = [
        item
        for item in evidence
        if item.polarity == "positive"
        and item.quality in {EvidenceQuality.CANDIDATE, EvidenceQuality.CORROBORATED}
    ]
    families = {item.attrs.get("family", "") for item in candidates}

    if "static_analysis" in exercised:
        builder.add("static_analysis", NeedStatus.SATISFIED, 0, "baseline static evidence exists")
    else:
        builder.add(
            "static_analysis",
            NeedStatus.MISSING,
            W_BASELINE,
            "baseline static evidence is missing",
            Uncertainty.BASELINE,
        )

    if ctx.extra.get("program_context"):
        _bounty_needs(builder, candidates, families)

    if ctx.flag("protocol") or (families & _CROSS_CONTRACT_FAMILIES):
        builder.need(
            "cross_contract_analysis",
            W_PROTOCOL,
            Uncertainty.CROSS_CONTRACT,
            "cross-contract flow evidence is missing",
        )
    if ctx.flag("economic") or ctx.extra.get("case"):
        builder.need(
            "economic_simulation",
            W_ECONOMIC,
            Uncertainty.ECONOMIC,
            "no asset-delta evidence exists for the economic candidate",
        )
    if candidates or ctx.targeted:
        builder.need(
            "fuzzing",
            W_FUZZ_CANDIDATE if candidates else W_FUZZ_TARGET,
            Uncertainty.UNEXERCISED,
            "a candidate has not been explored statefully"
            if candidates
            else "the target has not been explored statefully",
        )
    if ctx.extra.get("property") == "encoded" or ctx.has_harness:
        builder.need(
            "test_execution",
            W_TEST,
            Uncertainty.PROPERTY,
            "an encoded property can be replayed",
        )
    if "fuzzing" in exercised and any(
        ref.startswith("symbolic_execution:") for ref in state.corpus_refs
    ):
        builder.add(
            "fuzzing",
            NeedStatus.HIGH_VALUE_FOLLOWUP,
            W_SEED_FOLLOWUP,
            "a symbolic counterexample is a new fuzz seed",
            Uncertainty.SEED_FOLLOWUP,
        )
    if ctx.difficult or (
        ("fuzzing" in exercised or "test_execution" in exercised) and stalled(state)
    ):
        builder.need(
            "symbolic_execution",
            W_SYMBOLIC,
            Uncertainty.REACHABILITY,
            "reachability stalled, so symbolic execution is the next capability",
        )

    if ctx.flag("runtime") or ctx.extra.get("sequence_id"):
        builder.need(
            "runtime_validation",
            W_RUNTIME_FOLLOWUP,
            Uncertainty.RUNTIME,
            "runtime evidence is missing",
        )
    if any(
        item.attrs.get("category") == "economic"
        and item.quality in {EvidenceQuality.OBSERVATION, EvidenceQuality.CANDIDATE}
        for item in evidence
    ):
        builder.need(
            "runtime_validation",
            W_RUNTIME_FOLLOWUP,
            Uncertainty.ECONOMIC_RUNTIME,
            "an economic observation has not been exercised at runtime",
            followup=True,
        )
    if ctx.flag("fork"):
        builder.need("fork_validation", W_FORK, Uncertainty.FORK, "fork-state evidence is missing")
    if ctx.flag("differential"):
        builder.need(
            "differential_validation",
            W_DIFFERENTIAL,
            Uncertainty.REPEATABILITY,
            "differential evidence is missing",
        )
    if any(item.polarity == "divergent" for item in evidence):
        builder.need(
            "fork_validation",
            W_DIVERGENCE,
            Uncertainty.DIVERGENCE,
            "local runs diverged and a pinned fork can discriminate",
        )

    for contradiction in state.contradictions:
        if contradiction.status != "open":
            continue
        for discriminator in contradiction.discriminators:
            if discriminator in exercised:
                continue
            builder.add(
                discriminator,
                NeedStatus.HIGH_VALUE_FOLLOWUP,
                W_CONTRADICTION,
                f"contradiction {contradiction.contradiction_id} needs a discriminating capability",
                Uncertainty.CONTRADICTION,
            )
            break
    return builder.build()


def _bounty_needs(builder: _Builder, candidates: list[EvidenceItem], families: set[str]) -> None:
    """Needs of a bounty campaign. They exist only when the request names a program."""
    for capability, weight, reason in (
        (
            "bounty_context_analysis",
            W_BOUNTY_CONTEXT,
            "program scope and triage priority are missing",
        ),
        (
            "caller_context_analysis",
            W_CALLER_CONTEXT,
            "caller-context flows have not been analyzed",
        ),
        ("oracle_quality_analysis", W_ORACLE, "oracle source quality has not been analyzed"),
        ("proof_binding_analysis", W_PROOF, "proof and message binding has not been analyzed"),
        (
            "accounting_analysis",
            W_ACCOUNTING,
            "token accounting and arithmetic have not been analyzed",
        ),
        (
            "account_abstraction_analysis",
            W_ACCOUNT_ABSTRACTION,
            "account abstraction has not been analyzed",
        ),
    ):
        builder.need(capability, weight, Uncertainty.SEMANTIC_RESEARCH, reason)
    builder.need(
        "compiler_advisory_analysis",
        W_COMPILER_ADVISORY,
        Uncertainty.COMPILER,
        "compiler advisories have not been matched against the build configuration",
    )
    builder.need(
        "compiler_differential_validation",
        W_COMPILER_DIFF,
        Uncertainty.COMPILER,
        "pipeline differences have not been compared",
    )
    if candidates and families & _RESEARCH_FAMILIES:
        builder.need(
            "vfcs_generation",
            W_VFCS,
            Uncertainty.UNEXERCISED,
            "research candidates have no call-sequence plan",
        )


def actionable(needs: tuple[CapabilityNeed, ...]) -> tuple[CapabilityNeed, ...]:
    return tuple(need for need in needs if need.status in ACTIONABLE)


def uncertainties_of(needs: tuple[CapabilityNeed, ...]) -> tuple[str, ...]:
    found: set[str] = set()
    for need in actionable(needs):
        found.update(need.uncertainties)
    return tuple(sorted(found))
