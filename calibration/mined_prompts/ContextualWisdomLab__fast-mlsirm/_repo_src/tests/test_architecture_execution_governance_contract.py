"""Contracts for feasibility-first, work-conserving repository governance docs."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    """Return repository UTF-8 text for one canonical governance document."""
    return (ROOT / path).read_text(encoding="utf-8")


def test_execution_governance_is_linked_across_canonical_architecture() -> None:
    """ADR 0013 must be discoverable from technical, architecture, and traceability views."""
    architecture = _read("ARCHITECTURE.md")
    trd = _read("docs/TRD.md")
    traceability = _read("docs/traceability/requirements-matrix.md")

    assert "ADR-0013" in architecture
    assert "ADR-0013" in trd
    assert "ADR-0013" in traceability


def test_execution_governance_preserves_feasibility_and_single_writer_invariants() -> None:
    """Canonical docs must explain why one blocker/action cannot terminate useful work."""
    adr = _read("docs/adr/0013-continuous-execution-and-documentation-governance.md")
    trd = _read("docs/TRD.md")
    traceability = _read("docs/traceability/requirements-matrix.md")

    for concept in (
        "work-conserving",
        "exact branch head",
        "active writer",
        "Parallel authority is prohibited",
        "non-actionable under current authority",
    ):
        assert concept.lower() in adr.lower()

    assert "feasibility" in trd.lower()
    assert "single-writer" in trd.lower()
    assert "work-conserving" in traceability.lower()
    assert "single-writer" in traceability.lower()
