"""The report a validation run emits.

It has two sections and the order is the point. What was run comes first: where
the records came from, how many there were, and what the metrics say. What was
concluded comes second: every gate, its threshold, what was observed, and
whether it passed. Conflating the two is how an optimistic reading quietly
replaces the evidence, which is why the numbers and the verdicts never share a
section.

The provenance line is the other thing the first section exists for. A report
built over a synthetic fixture says so at the top, so no reader mistakes an
exercise of the harness for a measurement of a trained model.
"""

from __future__ import annotations

from dataclasses import dataclass

from clave.validation.gates import GateConfig, GateOutcome, all_passed, evaluate
from clave.validation.metrics import ValidationSummary
from clave.validation.outcomes import OutcomeSet


def _number(value: float | None, digits: int = 4) -> str:
    """Render a value that may be unmeasured.

    Args:
        value: The value, or None.
        digits: Decimal places.

    Returns:
        The formatted value, or a word saying it was never measured.
    """
    return "unmeasured" if value is None else f"{value:.{digits}f}"


@dataclass(frozen=True)
class RunSection:
    """What was run: the records, where they came from, and what they say.

    Attributes:
        provenance: Where the records came from, in words.
        summary: Every metric the run produced.
    """

    provenance: str
    summary: ValidationSummary

    def render(self) -> str:
        """Render the section.

        Returns:
            The text, without a trailing newline.
        """
        summary = self.summary
        routing = summary.routing
        lines = [
            "## What was run",
            "",
            f"Records read from: {self.provenance}",
            "",
            f"- presented: {routing.presented}",
            f"- picked: {routing.picked}",
            f"- missed picks: {routing.missed}",
            f"- correct routes: {routing.correct}",
            f"- rejects: {routing.rejects}",
            f"- misroutes: {routing.misroutes}",
            f"- seen instances: {summary.seen_count}",
            f"- unseen instances: {summary.unseen_count}",
            "",
            "### Rates",
            "",
            f"- overall accuracy: {_number(summary.matrix.overall_accuracy)}",
            f"- pick success rate: {_number(routing.pick_success_rate)}",
            f"- missed pick rate: {_number(routing.missed_pick_rate)}",
            f"- misroute rate: {_number(routing.misroute_rate)}",
            f"- reject rate: {_number(routing.reject_rate)}",
            f"- end to end sorting rate: {_number(routing.sorting_rate)}",
            f"- accuracy on seen instances: {_number(summary.seen_accuracy)}",
            f"- accuracy on unseen instances: {_number(summary.unseen_accuracy)}",
            f"- generalization drop: {_number(summary.generalization_drop)}",
            "",
            "### Timing, at the tail",
            "",
            f"- decision latency samples: {summary.decision_latency.count}",
            f"- decision latency p50: {_number(summary.decision_latency.p50)}",
            f"- decision latency p95: {_number(summary.decision_latency.p95)}",
            f"- decision latency p99: {_number(summary.decision_latency.p99)}",
            f"- cycle time samples: {summary.cycle_time.count}",
            f"- cycle time p50: {_number(summary.cycle_time.p50)}",
            f"- cycle time p99: {_number(summary.cycle_time.p99)}",
            "",
            "### Per-class accuracy",
            "",
        ]
        accuracy = summary.matrix.per_class_accuracy()
        for class_id in summary.matrix.class_ids:
            support = summary.matrix.support(class_id)
            lines.append(
                f"- {class_id}: {_number(accuracy[class_id])} "
                f"over {support} records, "
                f"{summary.matrix.unpredicted_for(class_id)} unpredicted"
            )
        lines.extend(["", "### Named confusions", ""])
        for result in summary.named_confusions:
            lines.append(
                f"- {result.confusion.id} {result.confusion.name} "
                f"({', '.join(result.confusion.class_ids)}): "
                f"rate {_number(result.rate)} "
                f"over {result.support} records, "
                f"{result.confused_inside} confused inside the group, "
                f"{result.erred_outside} wrong outside it"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class VerdictSection:
    """What was concluded: every gate and whether it was met.

    Attributes:
        gates: One outcome per gate, including the ones that passed.
    """

    gates: tuple[GateOutcome, ...]

    @property
    def passed(self) -> bool:
        """Whether every gate is met."""
        return all_passed(self.gates)

    def render(self) -> str:
        """Render the section.

        Returns:
            The text, without a trailing newline.
        """
        lines = [
            "## What was concluded",
            "",
            "| Gate | Threshold | Observed | Verdict |",
            "| --- | --- | --- | --- |",
        ]
        for gate in self.gates:
            verdict = "pass" if gate.passed else "FAIL"
            lines.append(
                f"| {gate.id} | {gate.threshold} | "
                f"{_number(gate.observed)} | {verdict} |"
            )
        failures = [gate for gate in self.gates if not gate.passed]
        lines.extend(["", "### Unmet gates", ""])
        if not failures:
            lines.append("None. Every gate is met.")
        else:
            lines.extend(f"- {gate.id}: {gate.description}" for gate in failures)
        lines.extend(
            [
                "",
                f"Verdict: {'PASS' if self.passed else 'FAIL'}, "
                f"{len(failures)} of {len(self.gates)} gates unmet.",
            ]
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class ValidationReport:
    """One validation run, split into evidence and conclusion.

    Attributes:
        run: What was run and what the metrics say.
        verdict: What the gates concluded from it.
    """

    run: RunSection
    verdict: VerdictSection

    @classmethod
    def build(cls, outcomes: OutcomeSet, config: GateConfig) -> ValidationReport:
        """Score a set of records against a gate configuration.

        Args:
            outcomes: The records and their provenance.
            config: The thresholds and the class-to-channel mapping.

        Returns:
            The report.

        Raises:
            MetricError: If the channel mapping does not cover a class present
                in the records.
        """
        summary = ValidationSummary.over(outcomes, config.channel_map)
        return cls(
            run=RunSection(outcomes.provenance, summary),
            verdict=VerdictSection(evaluate(summary, config.thresholds)),
        )

    @property
    def passed(self) -> bool:
        """Whether the run passed, meaning every gate is met."""
        return self.verdict.passed

    def render(self) -> str:
        """Render the whole report.

        Returns:
            The text, ending with a newline.
        """
        return (
            f"# Validation report\n\n{self.run.render()}\n\n{self.verdict.render()}\n"
        )
