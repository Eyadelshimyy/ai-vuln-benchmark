"""What happened to one object, and the set of those records a run scores.

The record below is a placeholder. v0.6.0 owns rollout capture and will publish
the real format, at which point this file's reader is deleted and one adapter
converts a rollout record into these fields. Nothing else has to move, because
every metric in this package is a function of the fields rather than of the
file: no metric opens a path, and no metric knows that a JSON representation
exists.

A record describes an object that was presented to the system, meaning it
entered the arm's reachable window. An object that never entered the window is
outside this harness's boundary, since nothing could have been done about it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from clave.errors import ClaveError
from clave.taxonomy import BY_ID


class OutcomeError(ClaveError):
    """A record names a class the taxonomy lacks, or contradicts itself."""


@dataclass(frozen=True)
class ObjectOutcome:
    """One object presented to the system, and what the system did with it.

    Attributes:
        object_id: Identifier of the object instance, used in error messages.
        true_class: The material class the object actually is, `M-NN`.
        predicted_class: The class the model predicted, or None when it
            predicted nothing.
        picked: Whether the effector grasped the object and moved it. An object
            that left the window untouched is a throughput loss rather than a
            routing error, which is why this is separate from the channel.
        routed_channel: Where a picked object was placed, or None when it was
            not picked.
        seen_instance: Whether this object instance appeared in training.
            Generalization is measured over the instances where this is False.
        decision_latency_seconds: Time from frame to published decision, or
            None when the system published no decision for this object. An
            object nobody decided about has no decision latency, and a zero
            there would read as an instant decision.
        cycle_time_seconds: Time the object occupied the system, from decision
            to placement, or None when it was never picked.
    """

    object_id: str
    true_class: str
    predicted_class: str | None
    picked: bool
    routed_channel: str | None
    seen_instance: bool
    decision_latency_seconds: float | None
    cycle_time_seconds: float | None


@dataclass(frozen=True)
class OutcomeSet:
    """Every record a validation run scores, with where they came from.

    Attributes:
        provenance: Where the records came from, in words. A synthetic fixture
            says so here, which is what keeps a report from reading as a
            measurement of a trained model.
        outcomes: The records, in the order they were recorded.
    """

    provenance: str
    outcomes: tuple[ObjectOutcome, ...]

    def __init__(self, provenance: str, outcomes: Iterable[ObjectOutcome]) -> None:
        """Build a set, refusing any record that cannot be scored.

        Args:
            provenance: Where the records came from.
            outcomes: The records.

        Raises:
            OutcomeError: If a record names a class outside the taxonomy, if
                its picked flag and routed channel disagree, or if it carries a
                negative duration.
        """
        frozen = tuple(outcomes)
        for outcome in frozen:
            _check(outcome)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "outcomes", frozen)

    def __len__(self) -> int:
        """Return how many records the set holds."""
        return len(self.outcomes)


def _check(outcome: ObjectOutcome) -> None:
    """Refuse a record that cannot be scored consistently.

    Args:
        outcome: The record to check.

    Raises:
        OutcomeError: Naming the offending value.
    """
    if outcome.true_class not in BY_ID:
        raise OutcomeError(
            f"{outcome.object_id}: true class {outcome.true_class!r} "
            "is outside the taxonomy"
        )
    if outcome.predicted_class is not None and outcome.predicted_class not in BY_ID:
        raise OutcomeError(
            f"{outcome.object_id}: predicted class {outcome.predicted_class!r} "
            "is outside the taxonomy"
        )
    if outcome.picked and outcome.routed_channel is None:
        raise OutcomeError(f"{outcome.object_id}: was picked but landed nowhere")
    if not outcome.picked and outcome.routed_channel is not None:
        raise OutcomeError(
            f"{outcome.object_id}: was not picked but carries a routed channel"
        )
    if (
        outcome.decision_latency_seconds is not None
        and outcome.decision_latency_seconds < 0.0
    ):
        raise OutcomeError(f"{outcome.object_id}: decision_latency_seconds is negative")
    if outcome.predicted_class is None and outcome.decision_latency_seconds is not None:
        raise OutcomeError(
            f"{outcome.object_id}: carries a decision latency but no prediction"
        )
    if outcome.cycle_time_seconds is not None and outcome.cycle_time_seconds < 0.0:
        raise OutcomeError(f"{outcome.object_id}: cycle_time_seconds is negative")


def load_outcomes(path: Path) -> OutcomeSet:
    """Read a records file into a set.

    This reader is the placeholder half of the placeholder. The format is a
    JSON object carrying a provenance string and a list of records whose keys
    are the record's field names. v0.6.0 replaces it.

    Args:
        path: The records file.

    Returns:
        The set, validated.

    Raises:
        OutcomeError: If the file is not an object, if it declares no
            provenance, or if a record is missing a field.
    """
    parsed: Any = json.loads(path.read_text())
    if not isinstance(parsed, dict):
        raise OutcomeError(f"{path} does not contain an object")
    if "provenance" not in parsed:
        raise OutcomeError(
            f"{path} declares no 'provenance'. A records file says where its "
            "records came from, so a fixture is never read as a measurement."
        )
    names = [field.name for field in fields(ObjectOutcome)]
    records: list[ObjectOutcome] = []
    for index, raw in enumerate(parsed.get("outcomes", [])):
        missing = [name for name in names if name not in raw]
        if missing:
            raise OutcomeError(
                f"{path}: record {index} is missing {', '.join(missing)}"
            )
        records.append(ObjectOutcome(**{name: raw[name] for name in names}))
    return OutcomeSet(str(parsed["provenance"]), records)
