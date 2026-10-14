"""The comparison table, the recommendation, and the evidence pack.

Two of the five headline metrics the release asks for cannot be measured by
this harness. It scores the proposal loop and does not run the jaw, so every
record carries `picked = False` and no cycle. They appear in the table as
`unmeasured` with the reason attached, because a pick success rate of zero
is arithmetically true and reads as a result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clave.benchmark.config import BenchmarkConfig
from clave.benchmark.suite import ConfigurationResult
from clave.validation.gates import GateConfig, GateOutcome, evaluate
from clave.validation.metrics import ValidationSummary, percentile

UNMEASURED = "unmeasured"
"""What a metric reads when nothing in the system can produce it."""

UNMEASURABLE_METRICS: dict[str, str] = {
    "pick success rate": (
        "the benchmark scores the proposal loop and does not run the jaw, so "
        "every record carries picked = False and the rate would be zero by "
        "construction rather than by measurement"
    ),
    "cycle time": (
        "the same harness records no placement, so no record carries a cycle"
    ),
    "generalization drop": (
        "the benchmark runs the world the models trained on, so every instance "
        "is a seen one and the unseen partition is empty"
    ),
}
"""Every metric the release asks for that this benchmark cannot produce."""


@dataclass(frozen=True)
class ScoredConfiguration:
    """One row, measured and gated.

    Attributes:
        result: What the row produced.
        summary: The metrics over its records, or None when it did not run.
        gates: One outcome per gate that could be evaluated.
    """

    result: ConfigurationResult
    summary: ValidationSummary | None
    gates: tuple[GateOutcome, ...]

    @property
    def name(self) -> str:
        """The row's label."""
        return self.result.configuration.name

    @property
    def accuracy(self) -> float | None:
        """Overall classification accuracy, or None when it did not run."""
        return None if self.summary is None else self.summary.matrix.overall_accuracy

    @property
    def class_support(self) -> dict[str, int]:
        """How many records carry each true class."""
        if self.summary is None:
            return {}
        matrix = self.summary.matrix
        return {
            class_id: matrix.support(class_id)
            for class_id in matrix.class_ids
            if matrix.support(class_id) > 0
        }

    @property
    def majority_class_share(self) -> float | None:
        """What a predictor that always names the most common class would score.

        This is the bar a classifier has to clear before it has learned
        anything. A model below it is worse than a constant, whatever its loss
        curve did.
        """
        support = self.class_support
        if not support:
            return None
        return max(support.values()) / sum(support.values())

    @property
    def latency_p99(self) -> float | None:
        """Frame to published decision at the 99th percentile."""
        if not self.result.latencies_seconds:
            return None
        return percentile(self.result.latencies_seconds, 0.99)

    def as_dict(self) -> dict[str, Any]:
        """Render as plain data for the evidence pack."""
        return {
            "name": self.name,
            "note": self.result.configuration.note,
            "available": self.result.available,
            "unavailable_reason": self.result.unavailable_reason,
            "presented": self.result.presented,
            "decided": self.result.decided,
            "coverage": self.result.coverage,
            "proposals": self.result.proposals,
            "overridden": self.result.overridden,
            "decisions_outside_window": self.result.outside_window,
            "decisions_per_simulated_second": self.result.decision_rate,
            "belt_speeds_meters_per_second": list(self.result.belt_speeds),
            "overall_accuracy": self.accuracy,
            "majority_class_share": self.majority_class_share,
            "class_support": self.class_support,
            "decision_latency_p99_seconds": self.latency_p99,
            "pick_success_rate": UNMEASURED,
            "cycle_time_p99_seconds": UNMEASURED,
            "gates": {
                gate.id: {
                    "threshold": gate.threshold,
                    "observed": gate.observed,
                    "passed": gate.passed,
                }
                for gate in self.gates
            },
        }


def score(result: ConfigurationResult, gates: GateConfig) -> ScoredConfiguration:
    """Measure and gate one row.

    Args:
        result: What the row produced.
        gates: The thresholds every row is held to.

    Returns:
        The scored row. A row that did not run carries no summary and no gates,
        rather than a summary over nothing.
    """
    if not result.available or len(result.outcomes) == 0:
        return ScoredConfiguration(result=result, summary=None, gates=())
    summary = ValidationSummary.over(result.outcomes, gates.channel_map)
    return ScoredConfiguration(
        result=result, summary=summary, gates=evaluate(summary, gates.thresholds)
    )


def recommend(scored: list[ScoredConfiguration]) -> tuple[str, str]:
    """Name the configuration to run, and the measured reason.

    The control is excluded from the recommendation. It reads the world rather
    than the frame, so recommending it would be recommending not to use a model
    at all, which is not a configuration anybody can deploy.

    Args:
        scored: Every row.

    Returns:
        The name and the reason. When nothing is recommendable, the reason says
        why rather than naming a row anyway.
    """
    candidates = [
        row
        for row in scored
        if row.result.available
        and row.result.configuration.predictor == "checkpoints"
        and row.accuracy is not None
    ]
    if not candidates:
        return (
            "none",
            "no trained configuration produced a decision, so there is nothing "
            "to choose between",
        )
    best = max(candidates, key=lambda row: (row.accuracy or 0.0, row.result.coverage))
    others = [row for row in candidates if row is not best]
    if not others:
        return (
            best.name,
            "it is the only trained configuration that ran, so it is a default "
            "rather than a winner",
        )
    runner_up = max(others, key=lambda row: (row.accuracy or 0.0, row.result.coverage))
    reason = (
        f"overall accuracy {best.accuracy:.3f} over {best.result.presented} records "
        f"against {runner_up.accuracy:.3f} over {runner_up.result.presented} for "
        f"{runner_up.name}, at a p99 decision latency of "
        f"{_seconds(best.latency_p99)} against {_seconds(runner_up.latency_p99)}"
    )
    # One misclassified object moves an accuracy by 1/n. A gap narrower than
    # that is a coin landing one way, and calling it a result is how a
    # benchmark starts recommending noise.
    resolution = 1.0 / best.result.presented if best.result.presented else 1.0
    if abs((best.accuracy or 0.0) - (runner_up.accuracy or 0.0)) <= resolution:
        reason += (
            ". The accuracy gap is within what one record would move, so this "
            "is a latency choice rather than an accuracy one"
        )
    return best.name, reason


def _seconds(value: float | None) -> str:
    """Render a duration, or say it was never measured."""
    return UNMEASURED if value is None else f"{value * 1000:.1f} ms"


def _percent(value: float | None) -> str:
    """Render a share, or say it was never measured."""
    return UNMEASURED if value is None else f"{value * 100:.1f}%"


def table(scored: list[ScoredConfiguration]) -> str:
    """Render the comparison.

    Args:
        scored: Every row, in configuration order.

    Returns:
        The table, without a trailing newline.
    """
    lines = [
        "| Configuration | Presented | Decided | Accuracy | Decision p99 | "
        "Overridden | Pick success | Cycle time |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in scored:
        if not row.result.available:
            lines.append(
                f"| {row.name} | Unavailable | | | | | {UNMEASURED} | {UNMEASURED} |"
            )
            continue
        lines.append(
            f"| {row.name} | {row.result.presented} | {row.result.decided} | "
            f"{_percent(row.accuracy)} | {_seconds(row.latency_p99)} | "
            f"{row.result.overridden} | {UNMEASURED} | {UNMEASURED} |"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class EvidencePack:
    """Everything a later reader needs to trace a number back to its run.

    Attributes:
        config: The comparison that produced it.
        scored: Every row.
        machine: What the measurements were taken on.
        threads: Compute threads available.
        environment: Library versions.
        world_digest: Digest of the world configuration every run used.
    """

    config: BenchmarkConfig
    scored: list[ScoredConfiguration]
    machine: str
    threads: int
    environment: dict[str, str]
    world_digest: str

    def as_dict(self) -> dict[str, Any]:
        """Render as plain data a later reader parses without this package."""
        name, reason = recommend(self.scored)
        return {
            "seeds": list(self.config.seeds),
            "seconds_per_run": self.config.seconds_per_run,
            "benchmark_config_digest": self.config.digest,
            "world_config_digest": self.world_digest,
            "machine": self.machine,
            "threads": self.threads,
            "environment": self.environment,
            "recommendation": {"configuration": name, "reason": reason},
            "unmeasurable": UNMEASURABLE_METRICS,
            "configurations": [row.as_dict() for row in self.scored],
        }

    def write(self, path: Path) -> None:
        """Write the pack as JSON.

        Args:
            path: Where to write it.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")

    def render(self) -> str:
        """Render the human-readable comparison.

        Returns:
            The text, without a trailing newline.
        """
        name, reason = recommend(self.scored)
        lines = [
            "## What was run",
            "",
            f"Seeds: {', '.join(str(seed) for seed in self.config.seeds)} · "
            f"{self.config.seconds_per_run:.0f} simulated seconds each · "
            f"{self.machine}, {self.threads} threads",
            "",
            table(self.scored),
            "",
            self._baseline(),
            "",
            "## What could not be measured",
            "",
        ]
        lines.extend(
            f"- **{metric}**: {reason}"
            for metric, reason in UNMEASURABLE_METRICS.items()
        )
        lines.extend(["", "## Recommendation", "", f"**{name}**, because {reason}."])
        failures = [
            (row.name, gate.id)
            for row in self.scored
            for gate in row.gates
            if not gate.passed
        ]
        if failures:
            lines.extend(["", "## Gates not met", ""])
            lines.extend(f"- {name}: {gate}" for name, gate in failures)
        return "\n".join(lines)

    def _baseline(self) -> str:
        """State the bar a classifier has to clear to have learned anything."""
        shares = [
            row.majority_class_share
            for row in self.scored
            if row.majority_class_share is not None
        ]
        if not shares:
            return "No configuration produced a record, so there is no baseline."
        return (
            f"A predictor that always named the most common class would score "
            f"{max(shares) * 100:.1f}% on this population. A row below that "
            f"learned nothing a constant does not already do."
        )

    @property
    def passed(self) -> bool:
        """Whether every gate that could be evaluated was met."""
        return all(gate.passed for row in self.scored for gate in row.gates)
