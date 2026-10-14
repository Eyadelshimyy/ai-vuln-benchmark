"""The three separations a color camera cannot make.

`docs/waste-taxonomy.md` names them and asks this step to report them
specifically rather than folding them into general error. The reason is
diagnostic: a model losing accuracy inside one of these groups has hit a sensor
limit, and a model losing it elsewhere has learned less than it should have. One
of those calls for a near-infrared or magnetic sensor and the other calls for
more training, so an aggregate that merges them points at neither.

Naming a confusion removes nothing from the aggregate. Every record counted here
is also counted in the confusion matrix in `metrics.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from clave.validation.outcomes import ObjectOutcome


@dataclass(frozen=True)
class NamedConfusion:
    """One group of classes a color image cannot tell apart.

    Attributes:
        id: Stable identifier of the form `CONF-NN`, never reused.
        name: Human-readable name. Presentation only.
        class_ids: The taxonomy classes the group spans.
        reason: Why the separation is unavailable from a color image, and what
            a recovery facility uses instead.
    """

    id: str
    name: str
    class_ids: tuple[str, ...]
    reason: str


NAMED_CONFUSIONS: tuple[NamedConfusion, ...] = (
    NamedConfusion(
        "CONF-01",
        "Transparent resins",
        ("M-01", "M-03", "M-04"),
        "Clear PET, clear PP and clear PS share color and shape. A facility "
        "separates them by near-infrared absorption, which reads the resin "
        "rather than the appearance.",
    ),
    NamedConfusion(
        "CONF-02",
        "Can metals",
        ("M-05", "M-06"),
        "An aluminum beverage can and a steel food can are both cylindrical "
        "metal. A facility separates them magnetically, and a crushed can "
        "carries little of the proportion signal that remains.",
    ),
    NamedConfusion(
        "CONF-03",
        "Fiber flute",
        ("M-08", "M-09"),
        "The fluted layer that distinguishes corrugated board from flat "
        "paperboard is visible edge-on and invisible face-on.",
    ),
)
"""The three confusions, in the order the taxonomy introduces them."""


@dataclass(frozen=True)
class ConfusionResult:
    """How one named confusion fared over a set of records.

    Attributes:
        confusion: The group this describes.
        support: Records whose true class is in the group.
        confused_inside: Records predicted as a different member of the group.
        erred_outside: Records that were wrong in some other way, including
            records with no prediction at all.
        correct: Records predicted as their own class.
    """

    confusion: NamedConfusion
    support: int
    confused_inside: int
    erred_outside: int
    correct: int

    @property
    def rate(self) -> float | None:
        """Share of the group's records confused with another group member.

        Returns:
            The share, or None when the group has no records, since a group
            nobody tested is unmeasured rather than perfect.
        """
        if self.support == 0:
            return None
        return self.confused_inside / self.support


def confusion_rates(outcomes: Sequence[ObjectOutcome]) -> tuple[ConfusionResult, ...]:
    """Score every named confusion over a sequence of records.

    Args:
        outcomes: The records to score.

    Returns:
        One result per named confusion, in declaration order.
    """
    results: list[ConfusionResult] = []
    for confusion in NAMED_CONFUSIONS:
        members = set(confusion.class_ids)
        support = 0
        confused_inside = 0
        erred_outside = 0
        correct = 0
        for outcome in outcomes:
            if outcome.true_class not in members:
                continue
            support += 1
            if outcome.predicted_class == outcome.true_class:
                correct += 1
            elif outcome.predicted_class in members:
                confused_inside += 1
            else:
                erred_outside += 1
        results.append(
            ConfusionResult(
                confusion=confusion,
                support=support,
                confused_inside=confused_inside,
                erred_outside=erred_outside,
                correct=correct,
            )
        )
    return tuple(results)
