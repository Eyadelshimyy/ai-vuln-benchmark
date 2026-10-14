"""The loss each candidate is trained against.

An objective belongs to a candidate rather than to a stage: a detector and a
classifier both perceive, and are trained by entirely different losses.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

Objective = Callable[[Any, Any], Any]


def multilabel_presence(model: Any, batch: tuple[Any, Any]) -> Any:
    """Binary cross entropy over which classes appear in the frame.

    Several objects share a frame, so a single-label objective would be ill
    posed regardless of the architecture.
    """
    import torch

    images, targets = batch
    return torch.nn.functional.binary_cross_entropy_with_logits(model(images), targets)


def detection_loss(model: Any, batch: tuple[Any, Any]) -> Any:
    """The detector's own composite loss.

    torchvision detectors return a dictionary of losses when called in training
    mode with targets, so the objective sums what the architecture itself
    defines rather than inventing one.
    """
    images, targets = batch
    model.train()
    losses = model(images, targets)
    return sum(losses.values())


def pick_position(model: Any, batch: tuple[Any, Any, Any]) -> Any:
    """Mean squared error against the position the expert chose.

    Demonstrations are the only pick signal the simulation produces. The state
    input is zeroed because the dataset carries no arm proprioception, so this
    trains a vision-only policy.
    """
    import torch

    images, states, actions = batch
    predicted = model(images, states)
    if predicted.shape != actions.shape:
        predicted = predicted[..., : actions.shape[-1]]
    return torch.nn.functional.mse_loss(predicted, actions)


def act_loss(model: Any, batch: dict[str, Any]) -> Any:
    """The policy's own training loss.

    lerobot policies compute their loss internally and return it, so this asks
    the architecture rather than reimplementing an objective its authors
    already chose.
    """
    loss, _ = model.forward(batch)
    return loss


OBJECTIVES: dict[str, Objective] = {
    "resnet50-baseline": multilabel_presence,
    "faster-rcnn-mobilenetv3": detection_loss,
    "behavior-cloning-baseline": pick_position,
    "act": act_loss,
}


def batches_for(
    candidate: str,
    examples: Any,
    batch_size: int,
    window_exit: float,
    act_chunk_size: int = 10,
    augment: bool = False,
    input_side: int | None = None,
) -> Iterator[Any]:
    """Select the batch shape a candidate's objective expects.

    Args:
        candidate: Registry name.
        examples: Training examples.
        batch_size: Examples per batch.
        window_exit: Belt coordinate where the reachable window ends.
        act_chunk_size: Actions predicted per observation for ACT.
        augment: Whether to apply data augmentations to perception batches.
        input_side: Square side each image is resized to. None keeps the
            camera resolution. Action chunking already resizes to its own
            input, which is smaller than this side.

    Returns:
        An iterator of batches.

    Raises:
        KeyError: If the candidate has no objective, which means it cannot be
            trained from the signal this simulation produces.
    """
    from clave.training import adapters

    if candidate == "faster-rcnn-mobilenetv3":
        return adapters.detection_batches(
            examples, batch_size, augment=augment, input_side=input_side
        )
    if candidate == "resnet50-baseline":
        return adapters.classification_batches(
            examples, batch_size, augment=augment, input_side=input_side
        )
    if candidate == "act":
        return adapters.act_batches(examples, batch_size, window_exit, act_chunk_size)
    if candidate == "behavior-cloning-baseline":
        return adapters.policy_batches(
            examples, batch_size, window_exit, input_side=input_side
        )
    raise KeyError(candidate)
