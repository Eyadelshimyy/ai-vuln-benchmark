"""The emitted report."""

from __future__ import annotations

from pathlib import Path

from clave.validation.gates import GateConfig
from clave.validation.outcomes import ObjectOutcome, OutcomeSet
from clave.validation.report import ValidationReport

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_GATES = REPOSITORY_ROOT / "configs" / "validation" / "gates.yml"


def _record(object_id: str, seen_instance: bool = True) -> ObjectOutcome:
    """Build one correctly handled record.

    Args:
        object_id: Identifier.
        seen_instance: Whether the instance appeared in training.

    Returns:
        The record.
    """
    return ObjectOutcome(
        object_id=object_id,
        true_class="M-01",
        predicted_class="M-01",
        picked=True,
        routed_channel="CH-PET",
        seen_instance=seen_instance,
        decision_latency_seconds=0.1,
        cycle_time_seconds=0.5,
    )


def _report(*records: ObjectOutcome) -> ValidationReport:
    """Build a report over the committed gate configuration.

    Args:
        records: The records to score.

    Returns:
        The report.
    """
    return ValidationReport.build(
        OutcomeSet("synthetic fixture, not a measurement", records),
        GateConfig.load(COMMITTED_GATES),
    )


def test_run_section_comes_before_the_verdict_section() -> None:
    """Evidence first, conclusion second, never merged."""
    rendered = _report(_record("a")).render()
    assert "## What was run" in rendered
    assert "## What was concluded" in rendered
    assert rendered.index("## What was run") < rendered.index("## What was concluded")


def test_report_names_the_provenance_of_its_records() -> None:
    """A fixture is labeled as one wherever it is read."""
    report = _report(_record("a"))
    assert report.run.provenance == "synthetic fixture, not a measurement"
    assert "synthetic fixture, not a measurement" in report.render()


def test_report_lists_every_gate_including_the_ones_that_passed() -> None:
    """A threshold stays visible after the run."""
    report = _report(_record("a"))
    rendered = report.render()
    for gate in report.verdict.gates:
        assert gate.id in rendered
    assert "overall-accuracy" in rendered
    assert "per-class-accuracy:M-07" in rendered


def test_report_shows_the_threshold_beside_the_observation() -> None:
    """A verdict without its threshold is an opinion."""
    rendered = _report(_record("a")).render()
    assert "threshold" in rendered.lower()
    assert "observed" in rendered.lower()


def test_report_fails_when_any_gate_fails() -> None:
    """The verdict is the conjunction, not a majority."""
    report = _report(_record("a"))
    assert report.passed is False
    assert any(gate.passed for gate in report.verdict.gates)
    assert not all(gate.passed for gate in report.verdict.gates)


def test_run_section_reports_the_counts_that_were_scored() -> None:
    """What was run is stated in numbers, not in adjectives."""
    report = _report(_record("a"), _record("b", seen_instance=False))
    rendered = report.render()
    assert report.run.summary.seen_count == 1
    assert report.run.summary.unseen_count == 1
    assert "presented" in rendered
    assert "missed picks" in rendered
    assert "misroutes" in rendered


def test_report_names_each_confusion_individually() -> None:
    """The three are readable in the emitted text."""
    rendered = _report(_record("a")).render()
    assert "CONF-01" in rendered
    assert "CONF-02" in rendered
    assert "CONF-03" in rendered


def test_report_passes_when_every_gate_is_met() -> None:
    """A clean run says so instead of listing nothing."""
    from clave.taxonomy import MATERIAL_CLASSES
    from clave.validation.gates import GateThresholds

    records = tuple(
        ObjectOutcome(
            object_id=f"obj-{entry.id}",
            true_class=entry.id,
            predicted_class=entry.id,
            picked=True,
            routed_channel=entry.channel,
            seen_instance=index % 2 == 0,
            decision_latency_seconds=0.05,
            cycle_time_seconds=0.4,
        )
        for index, entry in enumerate(MATERIAL_CLASSES)
    )
    config = GateConfig(
        thresholds=GateThresholds(
            min_overall_accuracy=0.5,
            min_per_class_accuracy=0.5,
            min_class_support=1,
            min_pick_success_rate=0.5,
            max_misroute_rate=0.5,
            max_named_confusion_rate=0.5,
            max_decision_latency_p99_seconds=1.0,
            max_cycle_time_p99_seconds=2.0,
            max_unseen_accuracy_drop=0.5,
        ),
        channel_map={entry.id: entry.channel for entry in MATERIAL_CLASSES},
    )
    report = ValidationReport.build(OutcomeSet("fixture", records), config)
    assert report.passed is True
    assert "None. Every gate is met." in report.render()
