from app.adapters.evidence.base import EvidenceCollector
from app.adapters.evidence.collectors import (
    ReproductionEvidenceCollector,
    SourceCodeEvidenceCollector,
    StaticAnalysisEvidenceCollector,
    TestFailureEvidenceCollector,
)
from app.adapters.evidence.composite import CompositeEvidenceCollector
from app.domain.evidence import EvidenceBundle, EvidenceSource


def default_debugging_collectors() -> CompositeEvidenceCollector:
    """Collectors that wrap BugForge's existing debugging evidence sources."""
    return CompositeEvidenceCollector(
        [
            StaticAnalysisEvidenceCollector(),
            TestFailureEvidenceCollector(),
            SourceCodeEvidenceCollector(),
            ReproductionEvidenceCollector(),
        ]
    )


def bundle_from_debugging_artifacts(
    *,
    static_findings: list[object] | tuple[object, ...] = (),
    test_failures: list[object] | tuple[object, ...] = (),
    source_files: list[object] | tuple[object, ...] = (),
    reproductions: list[object] | tuple[object, ...] = (),
) -> EvidenceBundle:
    collector = default_debugging_collectors()
    return collector.collect_bundle(
        EvidenceSource(
            static_findings=static_findings,
            test_failures=test_failures,
            source_files=source_files,
            reproductions=reproductions,
        )
    )


__all__ = [
    "CompositeEvidenceCollector",
    "EvidenceCollector",
    "ReproductionEvidenceCollector",
    "SourceCodeEvidenceCollector",
    "StaticAnalysisEvidenceCollector",
    "TestFailureEvidenceCollector",
    "bundle_from_debugging_artifacts",
    "default_debugging_collectors",
]
