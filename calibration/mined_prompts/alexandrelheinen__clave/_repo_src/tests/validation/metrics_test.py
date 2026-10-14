"""Classification, routing, and timing metrics."""

from __future__ import annotations

import pytest

from clave.taxonomy import CLASS_COUNT
from clave.validation.metrics import (
    ConfusionMatrix,
    MetricError,
    RoutingCounts,
    TimingSummary,
    ValidationSummary,
    default_channel_map,
    percentile,
    split_by_instance,
)
from clave.validation.outcomes import ObjectOutcome, OutcomeSet


def _record(
    object_id: str = "obj",
    true_class: str = "M-01",
    predicted_class: str | None = "M-01",
    picked: bool = True,
    routed_channel: str | None = "CH-PET",
    seen_instance: bool = True,
    decision_latency_seconds: float = 0.1,
    cycle_time_seconds: float | None = 0.5,
) -> ObjectOutcome:
    """Build a record, overriding named fields.

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


def test_matrix_spans_every_taxonomy_class() -> None:
    """The matrix is sized by the taxonomy, not by the data."""
    matrix = ConfusionMatrix.over([_record()])
    assert len(matrix.class_ids) == CLASS_COUNT
    assert len(matrix.counts) == CLASS_COUNT
    assert all(len(row) == CLASS_COUNT for row in matrix.counts)


def test_a_misclassification_lands_off_the_diagonal() -> None:
    """A wrong prediction is placed where it can be read."""
    matrix = ConfusionMatrix.over([_record(true_class="M-05", predicted_class="M-06")])
    assert matrix.count("M-05", "M-06") == 1
    assert matrix.count("M-05", "M-05") == 0


def test_named_confusion_still_appears_in_the_general_matrix() -> None:
    """Naming a confusion removes nothing from the aggregate."""
    matrix = ConfusionMatrix.over(
        [
            _record(true_class="M-08", predicted_class="M-09"),
            _record(true_class="M-02", predicted_class="M-02"),
        ]
    )
    assert matrix.count("M-08", "M-09") == 1
    assert matrix.overall_accuracy == 0.5


def test_absent_prediction_is_counted_apart_from_a_wrong_one() -> None:
    """Predicting nothing is not the same as predicting wrong."""
    matrix = ConfusionMatrix.over(
        [
            _record(true_class="M-03", predicted_class=None),
            _record(true_class="M-03", predicted_class="M-04"),
        ]
    )
    assert matrix.unpredicted_for("M-03") == 1
    assert matrix.count("M-03", "M-04") == 1
    assert matrix.support("M-03") == 2
    assert matrix.per_class_accuracy()["M-03"] == 0.0


def test_a_class_with_no_records_is_unmeasured_rather_than_accurate() -> None:
    """Zero support yields no accuracy figure at all."""
    matrix = ConfusionMatrix.over([_record(true_class="M-01")])
    accuracy = matrix.per_class_accuracy()
    assert accuracy["M-01"] == 1.0
    assert accuracy["M-07"] is None
    assert matrix.support("M-07") == 0


def test_an_empty_matrix_reports_no_overall_accuracy() -> None:
    """The rule holds for the aggregate too."""
    assert ConfusionMatrix.over([]).overall_accuracy is None


def test_seen_and_unseen_instances_are_scored_separately() -> None:
    """Generalization is separable from memorization."""
    records = [
        _record(true_class="M-01", predicted_class="M-01", seen_instance=True),
        _record(true_class="M-01", predicted_class="M-04", seen_instance=False),
    ]
    seen, unseen = split_by_instance(records)
    assert ConfusionMatrix.over(seen).overall_accuracy == 1.0
    assert ConfusionMatrix.over(unseen).overall_accuracy == 0.0


def test_missed_pick_is_not_a_routing_error() -> None:
    """A throughput loss is not a contaminated bale."""
    counts = RoutingCounts.over(
        [
            _record(object_id="a", picked=False, routed_channel=None),
            _record(
                object_id="b",
                true_class="M-05",
                predicted_class="M-06",
                routed_channel="CH-FERROUS",
            ),
        ],
        default_channel_map(),
    )
    assert counts.presented == 2
    assert counts.missed == 1
    assert counts.misroutes == 1
    assert counts.picked == 1
    assert counts.correct == 0


def test_reject_is_counted_apart_from_a_misroute() -> None:
    """An abstention costs recovery; a misroute costs a bale."""
    counts = RoutingCounts.over(
        [
            _record(object_id="a", true_class="M-01", routed_channel="CH-REJECT"),
            _record(object_id="b", true_class="M-01", routed_channel="CH-HDPE"),
        ],
        default_channel_map(),
    )
    assert counts.rejects == 1
    assert counts.misroutes == 1


def test_residue_routed_to_reject_is_a_correct_route() -> None:
    """The reject channel is where residue belongs."""
    counts = RoutingCounts.over(
        [_record(true_class="M-11", routed_channel="CH-REJECT")],
        default_channel_map(),
    )
    assert counts.correct == 1
    assert counts.rejects == 0


def test_a_supplied_channel_map_changes_what_counts_as_a_misroute() -> None:
    """The mapping is configuration, not a constant in code."""
    records = [_record(true_class="M-01", routed_channel="CH-PLASTIC")]
    strict = RoutingCounts.over(records, default_channel_map())
    merged = RoutingCounts.over(
        records, {**default_channel_map(), "M-01": "CH-PLASTIC"}
    )
    assert strict.misroutes == 1
    assert merged.misroutes == 0
    assert merged.correct == 1


def test_pick_success_and_sorting_rate_differ_when_a_pick_is_misrouted() -> None:
    """Grasping well is not the same as sorting well."""
    counts = RoutingCounts.over(
        [
            _record(object_id="a", true_class="M-01", routed_channel="CH-PET"),
            _record(object_id="b", true_class="M-01", routed_channel="CH-HDPE"),
        ],
        default_channel_map(),
    )
    assert counts.pick_success_rate == 1.0
    assert counts.sorting_rate == 0.5
    assert counts.missed_pick_rate == 0.0
    assert counts.misroute_rate == 0.5


def test_counts_over_no_records_report_no_rates() -> None:
    """An empty run has no rate, rather than a rate of zero."""
    counts = RoutingCounts.over([], default_channel_map())
    assert counts.pick_success_rate is None
    assert counts.sorting_rate is None
    assert counts.missed_pick_rate is None
    assert counts.misroute_rate is None
    assert counts.reject_rate is None


def test_routing_refuses_a_class_the_channel_map_does_not_cover() -> None:
    """A partial mapping is an error, not a silent reject."""
    with pytest.raises(MetricError, match="M-01"):
        RoutingCounts.over([_record(true_class="M-01")], {"M-02": "CH-HDPE"})


def test_percentile_returns_an_observed_sample() -> None:
    """Nearest rank returns a value the system produced."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert percentile(values, 0.5) == 0.3
    assert percentile(values, 0.99) == 0.5
    assert percentile(values, 0.2) == 0.1


def test_percentile_is_order_independent() -> None:
    """Two runs over the same records agree exactly."""
    assert percentile([0.5, 0.1, 0.3], 0.99) == percentile([0.3, 0.5, 0.1], 0.99)


def test_percentile_refuses_an_empty_series() -> None:
    """An empty series has no tail to report."""
    with pytest.raises(MetricError, match="empty"):
        percentile([], 0.99)


def test_percentile_refuses_a_fraction_outside_its_range() -> None:
    """The method is defined on (0, 1]."""
    with pytest.raises(MetricError, match="fraction"):
        percentile([0.1], 1.5)


def test_timing_summary_carries_its_sample_count() -> None:
    """A tail figure travels with its support."""
    summary = TimingSummary.over([0.1, 0.2, 0.3, 0.4])
    assert summary.count == 4
    assert summary.p99 == 0.4
    assert summary.maximum == 0.4
    assert summary.p50 == 0.2
    assert summary.mean == pytest.approx(0.25)


def test_empty_timing_summary_is_unmeasured_rather_than_zero() -> None:
    """Nothing measured reports nothing, not a passing zero."""
    summary = TimingSummary.over([])
    assert summary.count == 0
    assert summary.p99 is None
    assert summary.mean is None
    assert summary.maximum is None


def test_summary_reads_latency_from_every_record_and_cycle_time_from_picks() -> None:
    """The two series have different denominators."""
    outcomes = OutcomeSet(
        "fixture",
        [
            _record(
                object_id="a",
                decision_latency_seconds=0.2,
                cycle_time_seconds=1.0,
            ),
            _record(
                object_id="b",
                picked=False,
                routed_channel=None,
                decision_latency_seconds=0.4,
                cycle_time_seconds=None,
            ),
        ],
    )
    summary = ValidationSummary.over(outcomes, default_channel_map())
    assert summary.decision_latency.count == 2
    assert summary.cycle_time.count == 1
    assert summary.decision_latency.p99 == 0.4


def test_summary_is_a_pure_function_of_its_records() -> None:
    """The same records produce the same summary, twice."""
    outcomes = OutcomeSet(
        "fixture",
        [
            _record(
                object_id="a",
                true_class="M-05",
                predicted_class="M-06",
                routed_channel="CH-FERROUS",
            ),
            _record(object_id="b", seen_instance=False),
        ],
    )
    first = ValidationSummary.over(outcomes, default_channel_map())
    second = ValidationSummary.over(outcomes, default_channel_map())
    assert first == second


def test_summary_reports_the_drop_from_seen_to_unseen() -> None:
    """Generalization is a number, not an impression."""
    outcomes = OutcomeSet(
        "fixture",
        [
            _record(object_id="a", predicted_class="M-01", seen_instance=True),
            _record(
                object_id="b",
                predicted_class="M-04",
                seen_instance=False,
                routed_channel="CH-PET",
            ),
        ],
    )
    summary = ValidationSummary.over(outcomes, default_channel_map())
    assert summary.seen_accuracy == 1.0
    assert summary.unseen_accuracy == 0.0
    assert summary.generalization_drop == 1.0


def test_generalization_drop_is_unmeasured_without_both_partitions() -> None:
    """A drop needs two accuracies to be a drop."""
    outcomes = OutcomeSet("fixture", [_record(seen_instance=True)])
    summary = ValidationSummary.over(outcomes, default_channel_map())
    assert summary.unseen_accuracy is None
    assert summary.generalization_drop is None


def test_summary_carries_the_named_confusions() -> None:
    """The three travel with every summary."""
    summary = ValidationSummary.over(
        OutcomeSet("fixture", [_record()]), default_channel_map()
    )
    assert len(summary.named_confusions) == 3


def test_default_channel_map_covers_every_taxonomy_class() -> None:
    """The default mapping is complete before it is overridden."""
    assert len(default_channel_map()) == CLASS_COUNT
