"""Score a perception candidate against a validation corpus.

The comparison uses the labels the simulator wrote. Identity is one of those
labels and is not an input: a score that was handed the object's id would be
measuring the recorder.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from clave.candidates.base import Stage
from clave.candidates.registry import REGISTRY
from clave.data.examples import Example
from clave.errors import ClaveError
from clave.progress import Progress, frame_total
from clave.taxonomy import MATERIAL_CLASSES
from clave.training.adapters import CLASS_INDEX

DECISION_AT = "decision_threshold"
MATCH_AT = "match_iou"


class PerceptionEvalError(ClaveError):
    """A corpus cannot be scored with the candidate that was asked for."""


@dataclass(frozen=True)
class ScoreThresholds:
    """The two cuts a perception score applies.

    Attributes:
        decision_threshold: Probability at or above which a class counts as
            present. One half is the sign of the logit.
        match_iou: Intersection over union at or above which a predicted box
            counts as the ground-truth box.
    """

    decision_threshold: float
    match_iou: float

    @classmethod
    def load(cls, path: Path) -> ScoreThresholds:
        """Read the thresholds.

        Args:
            path: The YAML file.

        Returns:
            The thresholds.

        Raises:
            PerceptionEvalError: If a key is absent.
        """
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or "validation" not in raw:
            raise PerceptionEvalError(f"{path} has no 'validation' section")
        section = raw["validation"]

        def need(key: str) -> float:
            if key not in section:
                raise PerceptionEvalError(
                    f"required configuration key 'validation.{key}' is missing"
                )
            return float(section[key])

        return cls(decision_threshold=need(DECISION_AT), match_iou=need(MATCH_AT))


@dataclass(frozen=True)
class PerceptionScore:
    """What a candidate agreed with the corpus about.

    Attributes:
        candidate: Registry name.
        dataset_digest: Digest of the validation corpus.
        frames: Examples that were scored.
        overall_agreement: Fraction of class bits, or of ground-truth boxes,
            that matched.
        mean_iou: Mean best intersection over union, for a detector. None for
            a classifier, which does not predict boxes.
    """

    candidate: str
    dataset_digest: str
    frames: int
    overall_agreement: float
    mean_iou: float | None


def bit_counts(
    predicted: NDArray[np.bool_],
    truth: NDArray[np.bool_],
) -> tuple[int, int]:
    """Class bits that match, and how many bits were compared.

    Args:
        predicted: Present or absent for one batch, shape (frames, classes).
        truth: The same shape, from the labels.

    Returns:
        Matching bits and compared bits. Both are zero when the arrays cannot
        be lined up.
    """
    if predicted.ndim != 2 or truth.ndim != 2 or predicted.shape[1] != truth.shape[1]:
        return 0, 0
    width = min(len(predicted), len(truth))
    if width == 0:
        return 0, 0
    compared = predicted[:width] == truth[:width]
    return int(np.sum(compared)), int(compared.size)


def presence_agreement(
    predicted: NDArray[np.bool_],
    truth: NDArray[np.bool_],
) -> float:
    """Fraction of class bits that agree.

    Args:
        predicted: Present or absent, shape (frames, classes).
        truth: The same shape, from the labels.

    Returns:
        The agreement, or zero when there is nothing to score.
    """
    if predicted.shape != truth.shape or predicted.size == 0:
        return 0.0
    return float(np.mean(predicted == truth))


def mean_best_iou(
    predicted: list[NDArray[np.float64]],
    truth: list[NDArray[np.float64]],
    match_iou: float,
) -> tuple[float, float]:
    """Match each ground-truth box to the prediction that overlaps it most.

    Args:
        predicted: One (N, 4) array of boxes per frame, as x_min, y_min, x_max,
            y_max.
        truth: One (M, 4) array of ground-truth boxes per frame.
        match_iou: Overlap at or above which a ground-truth box counts as found.

    Returns:
        Mean best overlap, and the fraction of ground-truth boxes that cleared
        `match_iou`. Both are zero when no ground-truth box was present.
    """
    overlaps: list[float] = []
    matched = 0
    total = 0
    for pred, gold in zip(predicted, truth, strict=True):
        for box in gold:
            total += 1
            if len(pred) == 0:
                overlaps.append(0.0)
                continue
            best = max(_iou(box, other) for other in pred)
            overlaps.append(best)
            if best >= match_iou:
                matched += 1
    if total == 0:
        return 0.0, 0.0
    return float(np.mean(overlaps)), matched / total


def _iou(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    """Intersection over union of two boxes."""
    x_min = max(float(left[0]), float(right[0]))
    y_min = max(float(left[1]), float(right[1]))
    x_max = min(float(left[2]), float(right[2]))
    y_max = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    if intersection == 0.0:
        return 0.0
    area_left = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    area_right = max(0.0, float(right[2] - right[0])) * max(
        0.0, float(right[3] - right[1])
    )
    union = area_left + area_right - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def truth_presence(examples: tuple[Example, ...]) -> NDArray[np.bool_]:
    """Multi-hot presence of each taxonomy class, from visible labels."""
    target = np.zeros((len(examples), len(MATERIAL_CLASSES)), dtype=np.bool_)
    for row, example in enumerate(examples):
        for label in example.visible_labels:
            target[row, CLASS_INDEX[label.material_class]] = True
    return target


def truth_boxes(examples: tuple[Example, ...]) -> list[NDArray[np.float64]]:
    """Visible boxes, one array per example."""
    boxes: list[NDArray[np.float64]] = []
    for example in examples:
        rows = [
            [float(value) for value in label.bbox]
            for label in example.visible_labels
            if label.bbox is not None
        ]
        boxes.append(np.asarray(rows, dtype=np.float64).reshape(-1, 4))
    return boxes


def score_candidate(
    dataset: Path,
    candidate: str,
    checkpoint: Path,
    thresholds: ScoreThresholds,
) -> PerceptionScore:
    """Load a perception checkpoint and score it on the validation split.

    Args:
        dataset: Verified validation corpus directory.
        candidate: Registry name.
        checkpoint: A `.pt` file written by `clave train`.
        thresholds: Decision and overlap cuts.

    Returns:
        The score.

    Raises:
        PerceptionEvalError: If the candidate is not a perception model this
            scorer knows, or the checkpoint names a different candidate.
    """
    from clave.data.dataset import read as read_dataset
    from clave.training.adapters import classification_batches, detection_batches
    from clave.training.runner import _candidate_for

    spec = _spec(candidate)
    if spec.stage is not Stage.PERCEPTION:
        raise PerceptionEvalError(
            f"{candidate} is a policy candidate; this corpus scores perception"
        )
    if not checkpoint.is_file():
        raise PerceptionEvalError(f"checkpoint {checkpoint} is missing")

    import torch

    loaded = _candidate_for(candidate).load()
    if not loaded.loaded:
        raise PerceptionEvalError(
            loaded.unavailable_reason or f"{candidate} did not load"
        )
    state = torch.load(checkpoint, weights_only=False)
    if state.get("candidate") != candidate:
        raise PerceptionEvalError(
            f"{checkpoint} holds a checkpoint for {state.get('candidate')!r}, "
            f"not {candidate!r}"
        )
    model: Any = loaded.model
    model.load_state_dict(state["model"])
    model.eval()
    stored_side = state.get("input_side_pixels")
    input_side = int(stored_side) if stored_side is not None else None

    description = read_dataset(dataset)
    part = description.role or "validation"
    from clave.data.dataset import iter_split

    frames = 0
    bits_equal = 0
    bits = 0
    overlap_sum = 0.0
    overlap_count = 0
    matched_boxes = 0
    if candidate not in ("faster-rcnn-mobilenetv3", "resnet50-baseline"):
        raise PerceptionEvalError(
            f"{candidate} has no corpus scorer; the known ones are "
            "resnet50-baseline and faster-rcnn-mobilenetv3"
        )
    live_bits = _RunningBits()
    live_boxes = _RunningBoxes()
    agreement = 0.0
    mean_iou: float | None = None
    bar = Progress(frame_total(description, part), f"validate {candidate}", "frame")
    try:
        for rollout in iter_split(dataset, part):
            examples = rollout.examples
            frames += len(examples)
            if candidate == "faster-rcnn-mobilenetv3":
                mean_iou, agreement, count = _score_detector(
                    model,
                    examples,
                    thresholds,
                    detection_batches,
                    bar,
                    live_boxes,
                    input_side,
                )
                overlap_sum += mean_iou * count
                overlap_count += count
                matched_boxes += int(round(agreement * count))
            else:
                equal, total = _score_classifier(
                    model,
                    examples,
                    thresholds,
                    classification_batches,
                    bar,
                    live_bits,
                    input_side,
                )
                bits_equal += equal
                bits += total
            del rollout
            gc.collect()
        if candidate == "faster-rcnn-mobilenetv3":
            agreement = matched_boxes / overlap_count if overlap_count else 0.0
            mean_iou = overlap_sum / overlap_count if overlap_count else 0.0
            bar.update(0, agreement=agreement, iou=mean_iou)
        else:
            agreement = bits_equal / bits if bits else 0.0
            mean_iou = None
            bar.update(0, agreement=agreement)
    finally:
        bar.close()
    return PerceptionScore(
        candidate=candidate,
        dataset_digest=description.digest,
        frames=frames,
        overall_agreement=agreement,
        mean_iou=mean_iou,
    )


@dataclass
class _RunningBits:
    """Class bits scored so far, across rollouts."""

    equal: int = 0
    total: int = 0

    def add(self, equal: int, total: int) -> None:
        """Add one batch."""
        self.equal += equal
        self.total += total

    @property
    def agreement(self) -> float:
        """Fraction of class bits that matched."""
        if self.total == 0:
            return 0.0
        return self.equal / self.total


@dataclass
class _RunningBoxes:
    """Box overlaps scored so far, across rollouts.

    The posted agreement rounds each batch the same way the finished score
    rounds a rollout, so the bar can move before the rollout is done. The
    number printed at the end is still the rollout-level score.
    """

    overlap_sum: float = 0.0
    count: int = 0
    matched: int = 0

    def add(self, mean_iou: float, agreement: float, count: int) -> None:
        """Add one batch of boxes."""
        self.overlap_sum += mean_iou * count
        self.count += count
        self.matched += int(round(agreement * count))

    def metrics(self) -> tuple[float, float]:
        """Running agreement and mean overlap."""
        if self.count == 0:
            return 0.0, 0.0
        return self.matched / self.count, self.overlap_sum / self.count


def _spec(name: str) -> Any:
    """Find a candidate spec by registry name."""
    for candidate, _builder in REGISTRY:
        if candidate.spec.name == name:
            return candidate.spec
    raise PerceptionEvalError(f"no candidate named {name!r} in the registry")


def _score_classifier(
    model: Any,
    examples: tuple[Example, ...],
    thresholds: ScoreThresholds,
    batches: Any,
    bar: Progress,
    live: _RunningBits,
    input_side: int | None,
) -> tuple[int, int]:
    """Agreement of a multi-label head with visible classes.

    Returns:
        Class bits that matched, and the number of class bits scored.
    """
    import torch

    truth = truth_presence(examples)
    if len(examples) == 0:
        return 0, 0
    equal_bits = 0
    total_bits = 0
    cursor = 0
    with torch.no_grad():
        for images, _targets in batches(
            examples, batch_size=1, augment=False, input_side=input_side
        ):
            logits = model(images)
            present = torch.sigmoid(logits) >= thresholds.decision_threshold
            predicted = present.detach().cpu().numpy().astype(np.bool_)
            width = int(predicted.shape[0])
            matched, compared = bit_counts(predicted, truth[cursor : cursor + width])
            cursor += width
            equal_bits += matched
            total_bits += compared
            live.add(matched, compared)
            bar.update(width, agreement=live.agreement)
    return equal_bits, total_bits


def _score_detector(
    model: Any,
    examples: tuple[Example, ...],
    thresholds: ScoreThresholds,
    batches: Any,
    bar: Progress,
    live: _RunningBoxes,
    input_side: int | None,
) -> tuple[float, float, int]:
    """Overlap of predicted boxes with the ground-truth boxes.

    Returns:
        Mean best overlap, the fraction of boxes that cleared the match cut,
        and how many ground-truth boxes were scored.
    """
    import torch

    usable = tuple(item for item in examples if item.visible_labels)
    gold = truth_boxes(usable)
    predicted: list[NDArray[np.float64]] = []
    cursor = 0
    model.eval()
    with torch.no_grad():
        for images, _targets in batches(
            usable, batch_size=1, augment=False, input_side=input_side
        ):
            outputs = model(images)
            batch: list[NDArray[np.float64]] = []
            for output in outputs:
                boxes = output["boxes"].detach().cpu().numpy()
                drawn = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
                batch.append(drawn)
                predicted.append(drawn)
            batch_gold = gold[cursor : cursor + len(batch)]
            cursor += len(batch)
            batch_iou, batch_found = mean_best_iou(
                batch, batch_gold, thresholds.match_iou
            )
            batch_count = sum(len(boxes) for boxes in batch_gold)
            live.add(batch_iou, batch_found, batch_count)
            shown = live.metrics()
            bar.update(len(batch), agreement=shown[0], iou=shown[1])
    while len(predicted) < len(gold):
        predicted.append(np.zeros((0, 4), dtype=np.float64))
    kept = predicted[: len(gold)]
    mean_iou, agreement = mean_best_iou(kept, gold, thresholds.match_iou)
    count = sum(len(boxes) for boxes in gold)
    return mean_iou, agreement, count


def score_kept_examples(
    model: Any,
    candidate: str,
    examples: tuple[Example, ...],
    thresholds: ScoreThresholds,
    input_side: int | None,
    bar: Progress,
) -> tuple[int, int, float, int]:
    """Score one group of kept frames with the live model.

    Args:
        model: The weights currently being trained.
        candidate: Registry name.
        examples: Frames from one rollout, already limited to the pick.
        thresholds: Decision and overlap cuts.
        input_side: Side the checkpoint resizes to.
        bar: The selection bar for this epoch.

    Returns:
        Matching class bits, compared class bits, the sum of best overlaps,
        and the number of ground-truth boxes. A classifier fills the bit
        counts. A detector fills the overlap sum.
    """
    from clave.training.adapters import classification_batches, detection_batches

    if candidate == "resnet50-baseline":
        equal, total = _score_classifier(
            model,
            examples,
            thresholds,
            classification_batches,
            bar,
            _RunningBits(),
            input_side,
        )
        return equal, total, 0.0, 0
    mean_iou, _agreement, count = _score_detector(
        model,
        examples,
        thresholds,
        detection_batches,
        bar,
        _RunningBoxes(),
        input_side,
    )
    return 0, 0, mean_iou * count, count
