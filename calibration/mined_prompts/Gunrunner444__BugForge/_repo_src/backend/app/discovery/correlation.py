"""Relate independent engine observations without verifying a finding."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.discovery.results import DynamicFinding, DynamicResult


@dataclass
class RelatedObservation:
    key: str
    engines: tuple[str, ...]
    detectors: tuple[str, ...]
    independent_engines: int
    verified: bool = False
    findings: tuple[DynamicFinding, ...] = field(default_factory=tuple)


def correlate_results(results: list[DynamicResult]) -> list[RelatedObservation]:
    """Group observations that name the same contract function and detector family.

    Two hits from the same engine and detector are one observation. A second
    engine can support the same hypothesis. Nothing here is verified.
    """
    groups: dict[str, list[tuple[str, DynamicFinding]]] = {}
    for result in results:
        if not result.executed and result.provenance != "slither_json":
            continue
        for finding in result.findings:
            family = detector_family(finding.detector_id)
            key = "|".join(
                part.lower()
                for part in (
                    finding.contract or result.contract,
                    finding.function or result.function,
                    family,
                )
            )
            groups.setdefault(key, []).append((result.engine, finding))
    related: list[RelatedObservation] = []
    for key, items in groups.items():
        unique: list[tuple[str, DynamicFinding]] = []
        seen: set[tuple[str, str]] = set()
        for engine, finding in items:
            ident = (engine, finding.detector_id)
            if ident in seen:
                continue
            seen.add(ident)
            unique.append((engine, finding))
        engines = tuple(dict.fromkeys(engine for engine, _finding in unique))
        related.append(
            RelatedObservation(
                key=key,
                engines=engines,
                detectors=tuple(finding.detector_id for _engine, finding in unique),
                independent_engines=len(engines),
                verified=False,
                findings=tuple(finding for _engine, finding in unique),
            )
        )
    return related


_RESEARCH_PREFIXES = {
    "caller_context": "caller_context",
    "oracle": "oracle_quality",
    "message": "message_binding",
    "signature": "message_binding",
    "aa": "account_abstraction",
    "accounting": "balance_delta",
    "arithmetic": "arithmetic",
    "compiler_advisory": "compiler_advisory",
}


def detector_family(detector_id: str) -> str:
    text = detector_id.lower()
    research = _RESEARCH_PREFIXES.get(text.split(".", 1)[0]) if "." in text else None
    if research:
        return research
    if "reentr" in text:
        return "reentrancy"
    if "tx.origin" in text or "auth" in text:
        return "authorization"
    if "delegate" in text or "unchecked_call" in text or "low-level" in text:
        return "external_call"
    return text
