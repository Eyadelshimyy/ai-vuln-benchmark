"""What a dataset actually contains.

Counts come from the examples present, never from the proportions a
configuration requested. A class present in the taxonomy and absent from the
data produces a confusion matrix column of zeros, which reads as a model failure
unless somebody wrote down that the data never had one.

Composition is reported per split part as well as overall, because a class
present overall can still be absent from the test part.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from clave.data.examples import Rollout
from clave.taxonomy import MATERIAL_CLASSES


@dataclass(frozen=True)
class PartComposition:
    """What one split part contains.

    Attributes:
        example_count: Frames in this part.
        class_counts: Material class to instance count.
        absent_classes: Taxonomy classes with no instance here.
    """

    example_count: int
    class_counts: dict[str, int]
    absent_classes: tuple[str, ...]


def compose(rollouts: tuple[Rollout, ...]) -> PartComposition:
    """Measure what a group of rollouts contains.

    Args:
        rollouts: The rollouts in one part.

    Returns:
        Measured counts, with absent taxonomy classes named.
    """
    counts: Counter[str] = Counter()
    examples = 0
    for rollout in rollouts:
        for example in rollout.examples:
            examples += 1
            counts.update(example.material_classes)
    absent = tuple(
        entry.id for entry in MATERIAL_CLASSES if counts.get(entry.id, 0) == 0
    )
    return PartComposition(
        example_count=examples,
        class_counts=dict(sorted(counts.items())),
        absent_classes=absent,
    )


def compose_parts(
    rollouts: tuple[Rollout, ...], parts: dict[str, tuple[str, ...]]
) -> dict[str, PartComposition]:
    """Measure every split part separately, plus the whole.

    Args:
        rollouts: Every rollout in the dataset.
        parts: Part name to the rollout identities it holds.

    Returns:
        Part name to its composition, with an `overall` entry added.
    """
    by_id = {rollout.rollout_id: rollout for rollout in rollouts}
    result = {
        name: compose(tuple(by_id[rid] for rid in members if rid in by_id))
        for name, members in parts.items()
    }
    result["overall"] = compose(rollouts)
    return result
