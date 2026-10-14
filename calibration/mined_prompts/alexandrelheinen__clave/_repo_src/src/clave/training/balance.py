"""Positive-class weights for the multi-label classifier.

M-02 is about half the labels that appear. `inverse` gives a rare visible
class a larger share of the gradient. A class the pick never shows stays at
weight 1, because there is no positive to amplify.
"""

from __future__ import annotations

from collections.abc import Sequence

from clave.errors import ClaveError


class BalanceError(ClaveError):
    """A checkpoint and a class-balance setting disagree."""


def inverse_pos_weight(positives: Sequence[int], frames: int) -> tuple[float, ...]:
    """Weight of each positive class.

    Args:
        positives: Frames in which that class is visible, one count per class.
        frames: Frames in the pick.

    Returns:
        `(frames - positives) / positives` per class, or 1 when the class
        never appears or the pick is empty.
    """
    weights: list[float] = []
    for count in positives:
        if frames <= 0 or count <= 0:
            weights.append(1.0)
        else:
            weights.append((frames - count) / count)
    return tuple(weights)


def resolve_class_weights(
    stored: Sequence[float] | None,
    *,
    checkpoint: bool,
    mode: str,
    length: int,
) -> tuple[float, ...] | None:
    """Choose the weight vector a resume is allowed to train with.

    Args:
        stored: The vector in the checkpoint, if it has one.
        checkpoint: Whether a checkpoint file exists.
        mode: `none` or `inverse`.
        length: Taxonomy length the vector has to match.

    Returns:
        None when the mode is `none`. The stored vector when a resume has
        one. None when there is no checkpoint yet, so the caller counts.

    Raises:
        BalanceError: If a resume under `inverse` has no vector, or the
            vector length is not the taxonomy length.
    """
    if mode == "none":
        return None
    if not checkpoint:
        return None
    if stored is None:
        raise BalanceError(
            "checkpoint has no class weights, so this resume would count "
            "the pick again and change the loss. Remove the checkpoint to "
            "start a run that stores inverse weights"
        )
    if len(stored) != length:
        raise BalanceError(
            f"checkpoint has {len(stored)} class weights and the taxonomy has {length}"
        )
    return tuple(float(value) for value in stored)
