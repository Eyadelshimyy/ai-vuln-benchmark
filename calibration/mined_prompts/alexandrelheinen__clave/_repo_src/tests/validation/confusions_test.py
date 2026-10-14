"""The three confusions a color camera cannot resolve.

That a named confusion stays inside the general matrix is proved in
`metrics_test.py`, where the matrix lives.
"""

from __future__ import annotations

from clave.validation.confusions import NAMED_CONFUSIONS, confusion_rates
from clave.validation.outcomes import ObjectOutcome


def _record(object_id: str, true_class: str, predicted: str | None) -> ObjectOutcome:
    """Build a picked, correctly routed record with the given classification.

    Args:
        object_id: Identifier.
        true_class: The class the object is.
        predicted: The class the model said, or None.

    Returns:
        The record.
    """
    return ObjectOutcome(
        object_id=object_id,
        true_class=true_class,
        predicted_class=predicted,
        picked=True,
        routed_channel="CH-PET",
        seen_instance=True,
        decision_latency_seconds=0.1,
        cycle_time_seconds=0.5,
    )


def test_three_confusions_are_declared() -> None:
    """The taxonomy names three, and all three are here."""
    assert len(NAMED_CONFUSIONS) == 3
    assert [confusion.id for confusion in NAMED_CONFUSIONS] == [
        "CONF-01",
        "CONF-02",
        "CONF-03",
    ]


def test_each_confusion_names_its_classes_and_its_reason() -> None:
    """A confusion carries what it spans and why."""
    by_id = {confusion.id: confusion for confusion in NAMED_CONFUSIONS}
    assert by_id["CONF-01"].class_ids == ("M-01", "M-03", "M-04")
    assert by_id["CONF-02"].class_ids == ("M-05", "M-06")
    assert by_id["CONF-03"].class_ids == ("M-08", "M-09")
    for confusion in NAMED_CONFUSIONS:
        assert confusion.reason
        assert confusion.name


def test_error_inside_a_group_is_counted_apart_from_one_outside() -> None:
    """The two error kinds do not merge."""
    outcomes = [
        _record("a", "M-05", "M-06"),
        _record("b", "M-05", "M-05"),
        _record("c", "M-06", "M-01"),
        _record("d", "M-06", None),
    ]
    results = {result.confusion.id: result for result in confusion_rates(outcomes)}
    metals = results["CONF-02"]
    assert metals.support == 4
    assert metals.confused_inside == 1
    assert metals.erred_outside == 2
    assert metals.rate == 0.25


def test_a_group_with_no_records_reports_no_rate() -> None:
    """An unmeasured group is unmeasured, not perfect."""
    results = {result.confusion.id: result for result in confusion_rates([])}
    assert results["CONF-01"].support == 0
    assert results["CONF-01"].rate is None


def test_a_group_whose_every_record_is_correct_reports_a_zero_rate() -> None:
    """Zero confusion is distinguishable from no data."""
    results = {
        result.confusion.id: result
        for result in confusion_rates([_record("a", "M-08", "M-08")])
    }
    assert results["CONF-03"].support == 1
    assert results["CONF-03"].rate == 0.0
