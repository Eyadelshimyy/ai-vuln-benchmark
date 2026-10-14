"""Regression contracts for the canonical documentation fitness matrix."""

from pathlib import Path


_MATRIX = Path(__file__).parents[1] / "docs" / "documentation_coverage.md"


def _matrix() -> str:
    """Return the canonical documentation-fitness matrix source."""
    return _MATRIX.read_text(encoding="utf-8")


def _row(source: str, capability: str) -> str:
    """Return one capability row from the protected-main maturity table."""
    return next(line for line in source.splitlines() if f"| {capability} |" in line)


def test_document_and_capability_states_are_not_conflated() -> None:
    """The matrix must expose separate finite vocabularies for docs and runtime maturity."""
    source = _matrix()
    for state in (
        "PRESENT_CURRENT",
        "PRESENT_STALE",
        "PARTIAL",
        "MISSING",
        "NOT_APPLICABLE",
        "SUPERSEDED",
        "OWNED_BY_ACTIVE_PR",
    ):
        assert f"**{state}**" in source
    for state in (
        "IMPLEMENTED_ON_PROTECTED_MAIN",
        "IMPLEMENTED_ON_ACTIVE_PR",
        "ACCEPTED_ARCHITECTURE",
        "PLANNED",
        "RESEARCH_ONLY",
        "DOWNSTREAM",
        "REJECTED",
        "OUT_OF_SCOPE",
    ):
        assert f"**{state}**" in source


def test_protected_main_docs_are_not_described_as_open_pr_only() -> None:
    """Integrated architecture docs must not regress to the former open-PR-only narrative."""
    source = _matrix()
    stale_claim = "canonical documentation baseline or its complete contract test is still only on an open PR"
    assert stale_claim not in source
    assert "Documentation contract CI" in source
    assert "PRESENT_CURRENT" in source


def test_recently_integrated_capabilities_are_not_left_as_active_pr_only() -> None:
    """Recent protected-main contracts and numerical migrations remain shipped truth."""
    source = _matrix()
    for capability in (
        "Reference-free RAG request/provenance boundary",
        "Governed post-pilot item-bank lifecycle",
        "Multilevel / cross-classified / multiple-membership contracts",
        "Factor retention evidence contract",
        "Fixed-anchor parameter linking arithmetic",
        "Observed-information Hessian and second-order diagnostics",
        "JMLE Adam/L-BFGS optimizer arithmetic",
        "Hourly review-repair caller",
        "LLM-judge raw JSON depth hardening",
    ):
        assert "IMPLEMENTED_ON_PROTECTED_MAIN" in _row(source, capability)


def test_parallel_hardening_is_promoted_on_protected_main() -> None:
    """Parallel-analysis control bounds are ancestral after #767."""
    source = _matrix()
    row = _row(source, "Parallel-analysis public control/resource hardening")
    assert "IMPLEMENTED_ON_PROTECTED_MAIN" in row


def test_dark_mode_report_change_is_promoted_on_protected_main() -> None:
    """Native dark-mode status accents are ancestral after #765."""
    source = _matrix()
    row = _row(source, "Essay-report native dark-mode status accents")
    assert "IMPLEMENTED_ON_PROTECTED_MAIN" in row
