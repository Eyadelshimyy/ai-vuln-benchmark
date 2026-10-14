"""Thresholds read from configuration, and the verdicts they produce."""

from __future__ import annotations

from pathlib import Path

import pytest

from clave.validation.gates import (
    GateConfig,
    GateConfigError,
    GateThresholds,
    evaluate,
)
from clave.validation.metrics import ValidationSummary, default_channel_map
from clave.validation.outcomes import ObjectOutcome, OutcomeSet

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_GATES = REPOSITORY_ROOT / "configs" / "validation" / "gates.yml"


def _thresholds(**overrides: float | int) -> GateThresholds:
    """Build thresholds for a test, overriding named fields.

    Args:
        overrides: Fields to replace.

    Returns:
        The thresholds. Every value here belongs to the test that reads it; the
        project's own numbers live in `configs/validation/gates.yml`.
    """
    fields: dict[str, float | int] = {
        "min_overall_accuracy": 0.5,
        "min_per_class_accuracy": 0.5,
        "min_class_support": 1,
        "min_pick_success_rate": 0.5,
        "max_misroute_rate": 0.5,
        "max_named_confusion_rate": 0.5,
        "max_decision_latency_p99_seconds": 1.0,
        "max_cycle_time_p99_seconds": 2.0,
        "max_unseen_accuracy_drop": 0.5,
    }
    fields.update(overrides)
    return GateThresholds(**fields)  # type: ignore[arg-type]


def _record(
    object_id: str,
    true_class: str = "M-01",
    predicted_class: str | None = "M-01",
    picked: bool = True,
    routed_channel: str | None = "CH-PET",
    seen_instance: bool = True,
    decision_latency_seconds: float = 0.1,
    cycle_time_seconds: float | None = 0.5,
) -> ObjectOutcome:
    """Build one record.

    Args:
        object_id: Identifier.
        true_class: The class the object is.
        predicted_class: The class the model said, or None.
        picked: Whether the effector grasped it.
        routed_channel: Where it landed, or None.
        seen_instance: Whether the instance appeared in training.
        decision_latency_seconds: Frame to decision.
        cycle_time_seconds: Decision to placement, or None.

    Returns:
        The record.
    """
    return ObjectOutcome(
        object_id=object_id,
        true_class=true_class,
        predicted_class=predicted_class,
        picked=picked,
        routed_channel=routed_channel,
        seen_instance=seen_instance,
        decision_latency_seconds=decision_latency_seconds,
        cycle_time_seconds=cycle_time_seconds,
    )


def _summary(*outcomes: ObjectOutcome) -> ValidationSummary:
    """Summarize records under the default channel mapping.

    Args:
        outcomes: The records.

    Returns:
        The summary.
    """
    return ValidationSummary.over(
        OutcomeSet("fixture", outcomes), default_channel_map()
    )


def test_thresholds_carry_no_defaults() -> None:
    """No threshold has a value until configuration supplies one."""
    with pytest.raises(TypeError):
        GateThresholds()  # type: ignore[call-arg]


def test_committed_configuration_supplies_every_threshold() -> None:
    """The project's own gate file loads."""
    config = GateConfig.load(COMMITTED_GATES)
    assert config.thresholds.max_cycle_time_p99_seconds > 0.0
    assert len(config.channel_map) == 11


def test_missing_threshold_key_fails_naming_it(tmp_path: Path) -> None:
    """Nothing is substituted for an absent key."""
    path = tmp_path / "gates.yml"
    path.write_text("gates:\n  min_overall_accuracy: 0.8\nchannels:\n  M-01: CH-PET\n")
    with pytest.raises(GateConfigError, match="min_per_class_accuracy"):
        GateConfig.load(path)


def test_missing_channel_entry_fails_naming_the_class(tmp_path: Path) -> None:
    """A partial channel mapping is refused, not completed."""
    committed = COMMITTED_GATES.read_text()
    path = tmp_path / "gates.yml"
    path.write_text(committed.replace("M-07: CH-GLASS", "# removed"))
    with pytest.raises(GateConfigError, match="M-07"):
        GateConfig.load(path)


def test_non_numeric_threshold_fails_naming_the_key(tmp_path: Path) -> None:
    """A threshold that is not a number is not a threshold."""
    committed = COMMITTED_GATES.read_text()
    path = tmp_path / "gates.yml"
    path.write_text(
        committed.replace("min_overall_accuracy: 0.80", "min_overall_accuracy: soon")
    )
    with pytest.raises(GateConfigError, match="min_overall_accuracy"):
        GateConfig.load(path)


def test_configuration_without_a_gates_section_is_refused(tmp_path: Path) -> None:
    """The file has to declare what it configures."""
    path = tmp_path / "gates.yml"
    path.write_text("channels:\n  M-01: CH-PET\n")
    with pytest.raises(GateConfigError, match="gates"):
        GateConfig.load(path)


def test_an_unmet_gate_fails_rather_than_reporting_a_number() -> None:
    """The verdict travels with the observation."""
    summary = _summary(_record("a", predicted_class="M-04"))
    outcomes = {
        gate.id: gate
        for gate in evaluate(summary, _thresholds(min_overall_accuracy=0.9))
    }
    gate = outcomes["overall-accuracy"]
    assert gate.passed is False
    assert gate.observed == 0.0
    assert gate.threshold == 0.9
    assert gate.description


def test_a_met_gate_passes() -> None:
    """A passing gate is reported, not omitted."""
    summary = _summary(_record("a"))
    outcomes = {
        gate.id: gate
        for gate in evaluate(summary, _thresholds(min_overall_accuracy=0.9))
    }
    assert outcomes["overall-accuracy"].passed is True


def test_class_below_minimum_support_fails_as_unmeasured() -> None:
    """A class nobody tested does not pass by having no errors."""
    summary = _summary(_record("a", true_class="M-01"))
    outcomes = {
        gate.id: gate for gate in evaluate(summary, _thresholds(min_class_support=5))
    }
    gate = outcomes["per-class-accuracy:M-01"]
    assert gate.passed is False
    assert "unmeasured" in gate.description
    assert outcomes["per-class-accuracy:M-07"].passed is False


def test_every_taxonomy_class_gets_its_own_gate() -> None:
    """No class escapes by being absent from the records."""
    summary = _summary(_record("a"))
    ids = {gate.id for gate in evaluate(summary, _thresholds())}
    assert sum(1 for gate_id in ids if gate_id.startswith("per-class-accuracy:")) == 11


def test_latency_gate_reads_the_ninety_ninth_percentile() -> None:
    """A mean that hides a tail does not pass this gate."""
    records = [
        _record(f"obj-{index}", decision_latency_seconds=0.01) for index in range(98)
    ]
    records.extend(
        _record(f"slow-{index}", decision_latency_seconds=5.0) for index in range(2)
    )
    summary = _summary(*records)
    outcomes = {
        gate.id: gate
        for gate in evaluate(summary, _thresholds(max_decision_latency_p99_seconds=1.0))
    }
    gate = outcomes["decision-latency-p99"]
    assert summary.decision_latency.mean is not None
    assert summary.decision_latency.mean < 1.0
    assert gate.observed == 5.0
    assert gate.passed is False


def test_cycle_time_gate_reads_the_ninety_ninth_percentile() -> None:
    """Cycle time is gated at the tail for the same reason."""
    summary = _summary(_record("a", cycle_time_seconds=3.0))
    outcomes = {
        gate.id: gate
        for gate in evaluate(summary, _thresholds(max_cycle_time_p99_seconds=2.0))
    }
    assert outcomes["cycle-time-p99"].passed is False


def test_unmeasured_timing_fails_rather_than_passing() -> None:
    """An absent tail is not a tail inside budget."""
    summary = _summary(
        _record("a", picked=False, routed_channel=None, cycle_time_seconds=None)
    )
    outcomes = {gate.id: gate for gate in evaluate(summary, _thresholds())}
    gate = outcomes["cycle-time-p99"]
    assert gate.passed is False
    assert "unmeasured" in gate.description


def test_generalization_drop_is_gated() -> None:
    """A candidate that only memorized fails."""
    summary = _summary(
        _record("a", predicted_class="M-01", seen_instance=True),
        _record("b", predicted_class="M-04", seen_instance=False),
    )
    outcomes = {
        gate.id: gate
        for gate in evaluate(summary, _thresholds(max_unseen_accuracy_drop=0.1))
    }
    gate = outcomes["generalization-drop"]
    assert gate.observed == 1.0
    assert gate.passed is False


def test_generalization_gate_fails_when_no_unseen_instance_was_scored() -> None:
    """A run with no held-out instance proves no generalization."""
    summary = _summary(_record("a", seen_instance=True))
    outcomes = {gate.id: gate for gate in evaluate(summary, _thresholds())}
    gate = outcomes["generalization-drop"]
    assert gate.passed is False
    assert "unmeasured" in gate.description


def test_named_confusion_gates_are_reported_individually() -> None:
    """Each named confusion carries its own verdict."""
    records = [
        _record(
            f"metal-{index}",
            true_class="M-05",
            predicted_class="M-06",
            routed_channel="CH-FERROUS",
        )
        for index in range(4)
    ]
    summary = _summary(*records)
    outcomes = {
        gate.id: gate
        for gate in evaluate(summary, _thresholds(max_named_confusion_rate=0.3))
    }
    assert outcomes["named-confusion:CONF-02"].observed == 1.0
    assert outcomes["named-confusion:CONF-02"].passed is False
    assert "unmeasured" in outcomes["named-confusion:CONF-01"].description


def test_misroute_and_pick_gates_are_separate() -> None:
    """A grasping problem and a routing problem gate apart."""
    summary = _summary(
        _record("a", picked=False, routed_channel=None, cycle_time_seconds=None),
        _record("b", routed_channel="CH-HDPE"),
    )
    outcomes = {
        gate.id: gate
        for gate in evaluate(
            summary, _thresholds(min_pick_success_rate=0.9, max_misroute_rate=0.1)
        )
    }
    assert outcomes["pick-success-rate"].observed == 0.5
    assert outcomes["misroute-rate"].observed == 0.5
    assert outcomes["pick-success-rate"].passed is False
    assert outcomes["misroute-rate"].passed is False


def test_gates_over_an_empty_run_all_fail() -> None:
    """A run that measured nothing does not pass."""
    summary = _summary()
    outcomes = evaluate(summary, _thresholds())
    assert outcomes
    assert not any(gate.passed for gate in outcomes)


def test_configuration_that_is_not_a_mapping_is_refused(tmp_path: Path) -> None:
    """A file that is not a mapping configures nothing."""
    path = tmp_path / "gates.yml"
    path.write_text("- min_overall_accuracy\n")
    with pytest.raises(GateConfigError, match="does not contain a mapping"):
        GateConfig.load(path)


def test_gates_section_that_is_not_a_mapping_is_refused(tmp_path: Path) -> None:
    """The sections have to be mappings to carry keys."""
    path = tmp_path / "gates.yml"
    path.write_text("gates: 0.8\nchannels:\n  M-01: CH-PET\n")
    with pytest.raises(GateConfigError, match="must be mappings"):
        GateConfig.load(path)


def test_class_support_threshold_loads_as_a_whole_number() -> None:
    """A record count is a count, not a fraction of one."""
    config = GateConfig.load(COMMITTED_GATES)
    assert isinstance(config.thresholds.min_class_support, int)
