"""One entry point for the Phase 50 semantic analyzers.

Every analyzer reads the same bounded ``ResearchModel`` and returns static
``SemanticCandidate`` values. A family that does not apply to the target returns
nothing, and the suite reports which families ran so an empty result is never
mistaken for a clean bill of health.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from app.parsing.solidity_account_abstraction import (
    account_abstraction_present,
    analyze_account_abstraction,
)
from app.parsing.solidity_accounting import analyze_balance_delta
from app.parsing.solidity_arithmetic import analyze_arithmetic
from app.parsing.solidity_caller_context import analyze_caller_context
from app.parsing.solidity_high_value import analyze_high_value
from app.parsing.solidity_message_binding import analyze_message_binding
from app.parsing.solidity_modules import analyze_protocol_modules, modules_present
from app.parsing.solidity_oracle_quality import analyze_oracle_quality
from app.parsing.solidity_research import ResearchModel, SemanticCandidate, build_research_model

Analyzer = Callable[[ResearchModel], list[SemanticCandidate]]

FAMILIES: dict[str, Analyzer] = {
    "caller_context": analyze_caller_context,
    "oracle_quality": analyze_oracle_quality,
    "message_binding": analyze_message_binding,
    "account_abstraction": analyze_account_abstraction,
    "balance_delta": analyze_balance_delta,
    "arithmetic": analyze_arithmetic,
    "high_value": analyze_high_value,
    "protocol_modules": analyze_protocol_modules,
}
MAX_TOTAL_CANDIDATES = 256


@dataclass(frozen=True)
class SuiteResult:
    model: ResearchModel
    candidates: tuple[SemanticCandidate, ...]
    families_run: tuple[str, ...]
    families_skipped: tuple[str, ...]
    truncated: bool

    def by_family(self, family: str) -> tuple[SemanticCandidate, ...]:
        return tuple(item for item in self.candidates if item.family == family)


def run_suite(model: ResearchModel, families: tuple[str, ...] | None = None) -> SuiteResult:
    selected = tuple(sorted(FAMILIES)) if families is None else tuple(sorted(set(families)))
    run: list[str] = []
    skipped: list[str] = []
    found: list[SemanticCandidate] = []
    for name in selected:
        analyzer = FAMILIES.get(name)
        if analyzer is None:
            skipped.append(name)
            continue
        if name == "account_abstraction" and not account_abstraction_present(model):
            skipped.append(name)
            continue
        if name == "protocol_modules" and not modules_present(model):
            skipped.append(name)  # evidence-triggered: no hook/bridge evidence
            continue
        run.append(name)
        found.extend(analyzer(model))
    ordered = sorted(
        found, key=lambda c: (c.file, c.line, c.family, c.detector, c.contract, c.function)
    )
    return SuiteResult(
        model=model,
        candidates=tuple(ordered[:MAX_TOTAL_CANDIDATES]),
        families_run=tuple(run),
        families_skipped=tuple(skipped),
        truncated=model.truncated or len(ordered) > MAX_TOTAL_CANDIDATES,
    )


def analyze_sources(
    sources: Mapping[str, str], families: tuple[str, ...] | None = None
) -> SuiteResult:
    return run_suite(build_research_model(sources), families)
