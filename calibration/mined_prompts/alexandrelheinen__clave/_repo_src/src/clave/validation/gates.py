"""Thresholds, and the verdicts they turn metrics into.

A threshold written after a run describes that run. A threshold committed to a
file before it is a gate. So nothing in this module carries a numeric default:
every value arrives from `configs/validation/gates.yml`, and a missing key fails
at load naming itself rather than being quietly filled in.

An unmet gate is a failure, not a number in a table. The command exits non-zero
and the report says which gate and by how much, because a validation run whose
only output is a number leaves the judgment to whoever reads it last.

Configuration is read with a local helper rather than through
`clave.world.config`. The helper is a few lines and the alternative points a
dependency the wrong way: the harness scores records and has to stay usable on a
machine where the simulated world is not installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from clave.errors import ClaveError
from clave.taxonomy import MATERIAL_CLASSES
from clave.validation.metrics import ValidationSummary


class GateConfigError(ClaveError):
    """The gate configuration is missing a key or carries a value that is not one."""


@dataclass(frozen=True)
class GateThresholds:
    """Every threshold the harness applies. None of them has a default.

    Attributes:
        min_overall_accuracy: Share of all records classified correctly.
        min_per_class_accuracy: Share of one class's records classified as that
            class.
        min_class_support: Records a class needs before its accuracy counts as
            measured.
        min_pick_success_rate: Share of presented objects the effector grasped.
        max_misroute_rate: Share of presented objects placed in the wrong
            material channel.
        max_named_confusion_rate: Share of a named confusion group's records
            predicted as another member of the same group.
        max_decision_latency_p99_seconds: Frame to decision, at the tail.
        max_cycle_time_p99_seconds: Decision to placement, at the tail.
        max_unseen_accuracy_drop: Accuracy lost from seen to unseen instances.
    """

    min_overall_accuracy: float
    min_per_class_accuracy: float
    min_class_support: int
    min_pick_success_rate: float
    max_misroute_rate: float
    max_named_confusion_rate: float
    max_decision_latency_p99_seconds: float
    max_cycle_time_p99_seconds: float
    max_unseen_accuracy_drop: float


@dataclass(frozen=True)
class GateConfig:
    """What a validation run reads before it scores anything.

    Attributes:
        thresholds: Every gate threshold.
        channel_map: Where each taxonomy class belongs on this deployment.
    """

    thresholds: GateThresholds
    channel_map: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> GateConfig:
        """Read a gate configuration file.

        Args:
            path: The YAML file.

        Returns:
            The configuration.

        Raises:
            GateConfigError: If the file is not a mapping, if a section or a
                threshold key is absent, if a threshold is not a number, or if
                the channel mapping omits a taxonomy class.
        """
        parsed = yaml.safe_load(path.read_text())
        if not isinstance(parsed, dict):
            raise GateConfigError(f"{path} does not contain a mapping")
        raw_gates = _require(parsed, "gates", path)
        raw_channels = _require(parsed, "channels", path)
        if not isinstance(raw_gates, dict) or not isinstance(raw_channels, dict):
            raise GateConfigError(f"{path}: 'gates' and 'channels' must be mappings")

        values: dict[str, float | int] = {}
        for field in fields(GateThresholds):
            raw = _require(raw_gates, field.name, path, "gates")
            # Postponed annotations make field.type the string "int" rather
            # than the type, so both forms are accepted here.
            wants_count = field.type in (int, "int")
            values[field.name] = _number(raw, field.name, wants_count, path)

        channel_map: dict[str, str] = {}
        for entry in MATERIAL_CLASSES:
            channel_map[entry.id] = str(
                _require(raw_channels, entry.id, path, "channels")
            )
        return cls(
            thresholds=GateThresholds(**values),  # type: ignore[arg-type]
            channel_map=channel_map,
        )


def _require(mapping: dict[str, Any], key: str, path: Path, section: str = "") -> Any:
    """Read a key, failing with its location when it is absent.

    Args:
        mapping: The mapping to read from.
        key: The key required.
        path: The file, named in the error.
        section: Dotted path of the parent, named in the error.

    Returns:
        The value.

    Raises:
        GateConfigError: If the key is absent. Nothing is substituted, because
            a silently defaulted threshold is a gate nobody agreed to.
    """
    if key not in mapping:
        where = f"{section}.{key}" if section else key
        raise GateConfigError(f"{path}: required key {where!r} is missing")
    return mapping[key]


def _number(raw: Any, key: str, want_integer: bool, path: Path) -> float | int:
    """Read a threshold as a number.

    Args:
        raw: The parsed value.
        key: The key, named in the error.
        want_integer: Whether the threshold is a count rather than a share.
        path: The file, named in the error.

    Returns:
        The number.

    Raises:
        GateConfigError: If the value is not a number. A boolean is refused
            even though Python counts it as an integer, since a threshold of
            True is a typing accident rather than a decision.
    """
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise GateConfigError(f"{path}: threshold {key!r} is not a number: {raw!r}")
    return int(raw) if want_integer else float(raw)


@dataclass(frozen=True)
class GateOutcome:
    """One gate, its threshold, what was observed, and the verdict.

    Attributes:
        id: Stable gate identifier, greppable from a report back to here.
        description: One line naming what the gate asks and what it saw.
        threshold: The number the configuration set.
        observed: The number the run produced, or None when the run measured
            nothing the gate could read.
        passed: Whether the gate is met. An observation of None never passes.
    """

    id: str
    description: str
    threshold: float
    observed: float | None
    passed: bool


def _at_least(
    gate_id: str, label: str, observed: float | None, threshold: float
) -> GateOutcome:
    """Build a gate that asks for at least a threshold.

    Args:
        gate_id: Gate identifier.
        label: What the observed quantity is.
        observed: The observed value, or None when unmeasured.
        threshold: The floor.

    Returns:
        The outcome.
    """
    if observed is None:
        return GateOutcome(
            gate_id,
            f"{label} is unmeasured, and the floor is {threshold}",
            threshold,
            None,
            False,
        )
    return GateOutcome(
        gate_id,
        f"{label} is {observed:.4f}, and the floor is {threshold}",
        threshold,
        observed,
        observed >= threshold,
    )


def _at_most(
    gate_id: str, label: str, observed: float | None, threshold: float
) -> GateOutcome:
    """Build a gate that asks for at most a threshold.

    Args:
        gate_id: Gate identifier.
        label: What the observed quantity is.
        observed: The observed value, or None when unmeasured.
        threshold: The ceiling.

    Returns:
        The outcome.
    """
    if observed is None:
        return GateOutcome(
            gate_id,
            f"{label} is unmeasured, and the ceiling is {threshold}",
            threshold,
            None,
            False,
        )
    return GateOutcome(
        gate_id,
        f"{label} is {observed:.4f}, and the ceiling is {threshold}",
        threshold,
        observed,
        observed <= threshold,
    )


def evaluate(
    summary: ValidationSummary, thresholds: GateThresholds
) -> tuple[GateOutcome, ...]:
    """Apply every gate to a summary.

    Args:
        summary: The metrics the run produced.
        thresholds: The configured thresholds.

    Returns:
        One outcome per gate, including the ones that passed, so a threshold
        stays visible after the run as well as before it.
    """
    outcomes: list[GateOutcome] = [
        _at_least(
            "overall-accuracy",
            "overall accuracy",
            summary.matrix.overall_accuracy,
            thresholds.min_overall_accuracy,
        )
    ]
    outcomes.extend(_class_gates(summary, thresholds))
    outcomes.extend(
        (
            _at_least(
                "pick-success-rate",
                "pick success rate",
                summary.routing.pick_success_rate,
                thresholds.min_pick_success_rate,
            ),
            _at_most(
                "misroute-rate",
                "misroute rate",
                summary.routing.misroute_rate,
                thresholds.max_misroute_rate,
            ),
        )
    )
    outcomes.extend(_confusion_gates(summary, thresholds))
    outcomes.extend(
        (
            _at_most(
                "decision-latency-p99",
                "decision latency p99 in seconds",
                summary.decision_latency.p99,
                thresholds.max_decision_latency_p99_seconds,
            ),
            _at_most(
                "cycle-time-p99",
                "cycle time p99 in seconds",
                summary.cycle_time.p99,
                thresholds.max_cycle_time_p99_seconds,
            ),
            _at_most(
                "generalization-drop",
                "accuracy drop from seen to unseen instances",
                summary.generalization_drop,
                thresholds.max_unseen_accuracy_drop,
            ),
        )
    )
    return tuple(outcomes)


def _class_gates(
    summary: ValidationSummary, thresholds: GateThresholds
) -> list[GateOutcome]:
    """Build one accuracy gate per taxonomy class.

    A class with fewer records than the configured minimum support fails as
    unmeasured. Passing it would mean a class nobody tested clears the gate on
    an empty denominator, which is the quiet way a report becomes untrue.

    Args:
        summary: The metrics the run produced.
        thresholds: The configured thresholds.

    Returns:
        One outcome per class, in taxonomy order.
    """
    accuracy = summary.matrix.per_class_accuracy()
    gates: list[GateOutcome] = []
    for class_id in summary.matrix.class_ids:
        gate_id = f"per-class-accuracy:{class_id}"
        support = summary.matrix.support(class_id)
        if support < thresholds.min_class_support:
            gates.append(
                GateOutcome(
                    gate_id,
                    f"{class_id} accuracy is unmeasured: {support} records, "
                    f"and the minimum support is {thresholds.min_class_support}",
                    thresholds.min_per_class_accuracy,
                    accuracy[class_id],
                    False,
                )
            )
            continue
        gates.append(
            _at_least(
                gate_id,
                f"{class_id} accuracy over {support} records",
                accuracy[class_id],
                thresholds.min_per_class_accuracy,
            )
        )
    return gates


def _confusion_gates(
    summary: ValidationSummary, thresholds: GateThresholds
) -> list[GateOutcome]:
    """Build one gate per named confusion.

    Args:
        summary: The metrics the run produced.
        thresholds: The configured thresholds.

    Returns:
        One outcome per named confusion, in declaration order.
    """
    gates: list[GateOutcome] = []
    for result in summary.named_confusions:
        gate_id = f"named-confusion:{result.confusion.id}"
        if result.support < thresholds.min_class_support:
            gates.append(
                GateOutcome(
                    gate_id,
                    f"{result.confusion.name} is unmeasured: {result.support} "
                    f"records, and the minimum support is "
                    f"{thresholds.min_class_support}",
                    thresholds.max_named_confusion_rate,
                    result.rate,
                    False,
                )
            )
            continue
        gates.append(
            _at_most(
                gate_id,
                f"{result.confusion.name} confusion over {result.support} records",
                result.rate,
                thresholds.max_named_confusion_rate,
            )
        )
    return gates


def all_passed(outcomes: Mapping[str, GateOutcome] | tuple[GateOutcome, ...]) -> bool:
    """Whether every gate is met.

    Args:
        outcomes: The gate outcomes.

    Returns:
        True when every gate passed. An empty set of gates returns True, which
        never happens in practice because `evaluate` always emits the class
        gates.
    """
    values = outcomes.values() if isinstance(outcomes, Mapping) else outcomes
    return all(outcome.passed for outcome in values)
