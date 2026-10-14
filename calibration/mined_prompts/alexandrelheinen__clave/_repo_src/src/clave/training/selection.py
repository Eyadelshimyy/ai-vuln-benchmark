"""Which epoch a held-out score keeps.

Higher is better for agreement and for mean intersection over union. An
equal score is not an improvement. The last epoch stays in the resume
checkpoint either way.
"""

from __future__ import annotations

from dataclasses import dataclass

from clave.errors import ClaveError

CLASSIFIER = "resnet50-baseline"
DETECTOR = "faster-rcnn-mobilenetv3"


class SelectionError(ClaveError):
    """A candidate has no held-out metric."""


@dataclass(frozen=True)
class SelectionState:
    """The best score so far and how long it has stood."""

    best_score: float | None
    best_epoch: int | None
    epochs_without_improvement: int

    def observe(
        self, score: float, epoch: int, patience: int
    ) -> tuple[SelectionState, bool]:
        """Record one epoch's held-out score.

        Args:
            score: Agreement or mean intersection over union.
            epoch: Epoch index, starting at zero.
            patience: Epochs without improvement before the run stops.

        Returns:
            The next state, and whether this score replaced the best.
        """
        if self.best_score is None or score > self.best_score:
            return SelectionState(score, epoch, 0), True
        stale = self.epochs_without_improvement + 1
        return SelectionState(self.best_score, self.best_epoch, stale), False

    def stops(self, patience: int) -> bool:
        """Whether patience is already spent."""
        return self.epochs_without_improvement >= patience


def fresh_selection() -> SelectionState:
    """A run that has not scored a held-out frame yet."""
    return SelectionState(None, None, 0)


def metric_for(candidate: str) -> str:
    """The held-out number this candidate is selected on.

    Args:
        candidate: Registry name.

    Returns:
        `agreement` or `mean_iou`.

    Raises:
        SelectionError: If the candidate is not a perception model with a
            corpus score.
    """
    if candidate == CLASSIFIER:
        return "agreement"
    if candidate == DETECTOR:
        return "mean_iou"
    raise SelectionError(
        f"{candidate} has no held-out metric; selection is agreement for "
        f"{CLASSIFIER} and mean intersection over union for {DETECTOR}"
    )
