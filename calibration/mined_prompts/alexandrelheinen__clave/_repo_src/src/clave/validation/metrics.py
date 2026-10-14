"""Metrics over recorded outcomes, as pure functions.

Nothing here opens a file, loads a model, or steps a simulator. Every function
takes a sequence of records and returns a value, so a reported number can be
rechecked by reading the records and the function beside each other.

Three of these metrics exist because earlier documents asked for them by name. A
missed pick is counted apart from a misroute because `docs/waste-taxonomy.md`
establishes that one is a throughput loss and the other contaminates a bale, so
the two have different causes and different fixes. Latency is summarized at the
99th percentile because a pipeline that averages well and misses one object in a
hundred drops that object on the floor. And the class-to-channel mapping arrives
as an argument because the taxonomy makes it a deployment's choice rather than a
property of the material stream.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from clave.errors import ClaveError
from clave.taxonomy import MATERIAL_CLASSES, REJECT_CHANNEL
from clave.validation.confusions import ConfusionResult, confusion_rates
from clave.validation.outcomes import ObjectOutcome, OutcomeSet


class MetricError(ClaveError):
    """A metric was asked for something the records cannot support."""


def default_channel_map() -> dict[str, str]:
    """Build the class-to-channel mapping the taxonomy records as its default.

    Returns:
        One channel per taxonomy class. A deployment with a different number of
        bins supplies its own mapping instead of editing this one.
    """
    return {entry.id: entry.channel for entry in MATERIAL_CLASSES}


CLASS_IDS: tuple[str, ...] = tuple(entry.id for entry in MATERIAL_CLASSES)
"""Taxonomy class identifiers, in the order the taxonomy declares them."""

_INDEX: dict[str, int] = {class_id: i for i, class_id in enumerate(CLASS_IDS)}


@dataclass(frozen=True)
class ConfusionMatrix:
    """Predicted class against true class, over every taxonomy class.

    Attributes:
        class_ids: Row and column order, which is taxonomy order.
        counts: `counts[true][predicted]`, as whole records.
        unpredicted: Per true class, records for which nothing was predicted.
            These are kept out of `counts` so a model that abstains is
            distinguishable from one that guesses wrong.
    """

    class_ids: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]
    unpredicted: tuple[int, ...]

    @classmethod
    def over(cls, outcomes: Sequence[ObjectOutcome]) -> ConfusionMatrix:
        """Build the matrix for a sequence of records.

        Args:
            outcomes: The records to count.

        Returns:
            The matrix, sized by the taxonomy rather than by the data.
        """
        size = len(CLASS_IDS)
        counts = [[0] * size for _ in range(size)]
        unpredicted = [0] * size
        for outcome in outcomes:
            row = _INDEX[outcome.true_class]
            if outcome.predicted_class is None:
                unpredicted[row] += 1
            else:
                counts[row][_INDEX[outcome.predicted_class]] += 1
        return cls(
            class_ids=CLASS_IDS,
            counts=tuple(tuple(row) for row in counts),
            unpredicted=tuple(unpredicted),
        )

    def count(self, true_class: str, predicted_class: str) -> int:
        """Return the records of one true class predicted as another.

        Args:
            true_class: The class the objects are.
            predicted_class: The class the model said.

        Returns:
            The record count.
        """
        return self.counts[_INDEX[true_class]][_INDEX[predicted_class]]

    def unpredicted_for(self, class_id: str) -> int:
        """Return the records of one class for which nothing was predicted.

        Args:
            class_id: The class.

        Returns:
            The record count.
        """
        return self.unpredicted[_INDEX[class_id]]

    def support(self, class_id: str) -> int:
        """Return how many records carry one true class.

        Args:
            class_id: The class.

        Returns:
            The record count, including records with no prediction.
        """
        row = _INDEX[class_id]
        return sum(self.counts[row]) + self.unpredicted[row]

    def correct(self, class_id: str) -> int:
        """Return how many records of one class were predicted as that class.

        Args:
            class_id: The class.

        Returns:
            The record count.
        """
        row = _INDEX[class_id]
        return self.counts[row][row]

    def per_class_accuracy(self) -> dict[str, float | None]:
        """Return accuracy for every taxonomy class.

        Returns:
            One entry per class. A class with no records maps to None, because
            a class nobody tested is unmeasured rather than perfect.
        """
        return {
            class_id: (
                None
                if self.support(class_id) == 0
                else self.correct(class_id) / self.support(class_id)
            )
            for class_id in self.class_ids
        }

    @property
    def total(self) -> int:
        """Return how many records the matrix counted."""
        return sum(sum(row) for row in self.counts) + sum(self.unpredicted)

    @property
    def overall_accuracy(self) -> float | None:
        """Return accuracy across every record.

        Returns:
            The share predicted correctly, or None when there are no records.
        """
        if self.total == 0:
            return None
        return sum(self.correct(class_id) for class_id in self.class_ids) / self.total


@dataclass(frozen=True)
class RoutingCounts:
    """What became of each object presented to the system.

    Attributes:
        presented: Objects that entered the arm's reachable window.
        picked: Objects the effector grasped and moved.
        missed: Objects that left the window untouched. A throughput loss, and
            not a routing error.
        correct: Picked objects placed in the channel their true class belongs
            in.
        rejects: Picked objects placed in the reject channel while their true
            class has a material channel. No bale is contaminated, and the
            material is not recovered.
        misroutes: Picked objects placed in a different material channel. This
            is the count that contaminates a bale.
    """

    presented: int
    picked: int
    missed: int
    correct: int
    rejects: int
    misroutes: int

    @classmethod
    def over(
        cls,
        outcomes: Sequence[ObjectOutcome],
        channel_map: Mapping[str, str],
    ) -> RoutingCounts:
        """Count routing outcomes against a class-to-channel mapping.

        Args:
            outcomes: The records to count.
            channel_map: Where each taxonomy class belongs on this deployment.

        Returns:
            The counts.

        Raises:
            MetricError: If a record's true class has no entry in the mapping.
                A partial mapping would silently turn every uncovered class
                into a misroute.
        """
        presented = len(outcomes)
        picked = 0
        correct = 0
        rejects = 0
        misroutes = 0
        for outcome in outcomes:
            if outcome.true_class not in channel_map:
                raise MetricError(
                    f"the channel mapping has no entry for {outcome.true_class!r}"
                )
            if not outcome.picked:
                continue
            picked += 1
            if outcome.routed_channel == channel_map[outcome.true_class]:
                correct += 1
            elif outcome.routed_channel == REJECT_CHANNEL:
                rejects += 1
            else:
                misroutes += 1
        return cls(
            presented=presented,
            picked=picked,
            missed=presented - picked,
            correct=correct,
            rejects=rejects,
            misroutes=misroutes,
        )

    def _rate(self, count: int) -> float | None:
        """Return a count as a share of the objects presented.

        Args:
            count: The numerator.

        Returns:
            The share, or None when nothing was presented.
        """
        if self.presented == 0:
            return None
        return count / self.presented

    @property
    def pick_success_rate(self) -> float | None:
        """Share of presented objects the effector grasped and moved."""
        return self._rate(self.picked)

    @property
    def missed_pick_rate(self) -> float | None:
        """Share of presented objects that left the window untouched."""
        return self._rate(self.missed)

    @property
    def misroute_rate(self) -> float | None:
        """Share of presented objects placed in the wrong material channel."""
        return self._rate(self.misroutes)

    @property
    def reject_rate(self) -> float | None:
        """Share of presented objects the system declined to route."""
        return self._rate(self.rejects)

    @property
    def sorting_rate(self) -> float | None:
        """Share of presented objects both picked and correctly routed.

        This is the end to end number: it falls when the grasp fails and when
        the classification fails, which is why it is reported beside the two
        rather than instead of them.
        """
        return self._rate(self.correct)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a percentile of a series by the nearest-rank method.

    Nearest rank takes the value at index `ceil(fraction * count) - 1` of the
    sorted series, so the result is a sample the system actually produced
    rather than an interpolation between two of them. The method is stated
    because two tools that interpolate differently disagree about p99 by a
    millisecond and nobody can tell which is right.

    Args:
        values: The series. Order does not matter.
        fraction: A share in (0, 1]. Pass 0.99 for p99.

    Returns:
        The value at that rank.

    Raises:
        MetricError: If the series is empty or the fraction is out of range.
    """
    if not values:
        raise MetricError("cannot take a percentile of an empty series")
    if not 0.0 < fraction <= 1.0:
        raise MetricError(f"fraction {fraction} is outside (0, 1]")
    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered))
    return ordered[min(rank, len(ordered)) - 1]


@dataclass(frozen=True)
class TimingSummary:
    """A duration series, summarized at its tail as well as its middle.

    Attributes:
        count: Samples behind every figure below. A p99 over eleven samples is
            the maximum wearing a percentile's name, which is why this travels
            with the rest.
        mean: Arithmetic mean, or None when there are no samples.
        p50: Median, or None.
        p95: 95th percentile, or None.
        p99: 99th percentile, or None. This is the figure the project gates on.
        maximum: Largest sample, or None.
    """

    count: int
    mean: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    maximum: float | None

    @classmethod
    def over(cls, values: Sequence[float]) -> TimingSummary:
        """Summarize a series.

        Args:
            values: The durations, in seconds.

        Returns:
            The summary. An empty series reports None throughout rather than
            zero, because zero is a passing number and nothing was measured.
        """
        if not values:
            return cls(count=0, mean=None, p50=None, p95=None, p99=None, maximum=None)
        return cls(
            count=len(values),
            mean=sum(values) / len(values),
            p50=percentile(values, 0.50),
            p95=percentile(values, 0.95),
            p99=percentile(values, 0.99),
            maximum=max(values),
        )


def split_by_instance(
    outcomes: Sequence[ObjectOutcome],
) -> tuple[tuple[ObjectOutcome, ...], tuple[ObjectOutcome, ...]]:
    """Partition records into seen and unseen object instances.

    Args:
        outcomes: The records to partition.

    Returns:
        The seen records first, then the unseen ones.
    """
    seen = tuple(outcome for outcome in outcomes if outcome.seen_instance)
    unseen = tuple(outcome for outcome in outcomes if not outcome.seen_instance)
    return seen, unseen


@dataclass(frozen=True)
class ValidationSummary:
    """Every metric a validation run produces, before any gate is applied.

    Attributes:
        provenance: Where the records came from, carried through so a report
            cannot present a fixture as a measurement.
        matrix: The confusion matrix over all records.
        routing: What became of each presented object.
        decision_latency: Frame to decision, over the records that carry one.
            A record for an object the system never decided about carries none,
            and including a zero for it would report an instant decision that
            never happened.
        cycle_time: Decision to placement, over the records that carry one.
        named_confusions: The three confusions the taxonomy names.
        seen_accuracy: Accuracy over instances seen in training, or None.
        unseen_accuracy: Accuracy over instances never seen, or None.
        seen_count: Records from seen instances.
        unseen_count: Records from unseen instances.
    """

    provenance: str
    matrix: ConfusionMatrix
    routing: RoutingCounts
    decision_latency: TimingSummary
    cycle_time: TimingSummary
    named_confusions: tuple[ConfusionResult, ...]
    seen_accuracy: float | None
    unseen_accuracy: float | None
    seen_count: int
    unseen_count: int

    @classmethod
    def over(
        cls, outcomes: OutcomeSet, channel_map: Mapping[str, str]
    ) -> ValidationSummary:
        """Compute every metric over one set of records.

        Args:
            outcomes: The records and their provenance.
            channel_map: Where each taxonomy class belongs on this deployment.

        Returns:
            The summary.

        Raises:
            MetricError: If the channel mapping does not cover a class present
                in the records.
        """
        records = outcomes.outcomes
        seen, unseen = split_by_instance(records)
        return cls(
            provenance=outcomes.provenance,
            matrix=ConfusionMatrix.over(records),
            routing=RoutingCounts.over(records, channel_map),
            decision_latency=TimingSummary.over(
                [
                    record.decision_latency_seconds
                    for record in records
                    if record.decision_latency_seconds is not None
                ]
            ),
            cycle_time=TimingSummary.over(
                [
                    record.cycle_time_seconds
                    for record in records
                    if record.cycle_time_seconds is not None
                ]
            ),
            named_confusions=confusion_rates(records),
            seen_accuracy=ConfusionMatrix.over(seen).overall_accuracy,
            unseen_accuracy=ConfusionMatrix.over(unseen).overall_accuracy,
            seen_count=len(seen),
            unseen_count=len(unseen),
        )

    @property
    def generalization_drop(self) -> float | None:
        """Accuracy lost going from seen instances to unseen ones.

        Returns:
            The drop, or None when either partition is empty. A candidate that
            only memorized its training instances shows a large positive drop.
        """
        if self.seen_accuracy is None or self.unseen_accuracy is None:
            return None
        return self.seen_accuracy - self.unseen_accuracy
