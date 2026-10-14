"""Tests for the dataset adapters."""

import numpy as np
import pytest

from clave.data.examples import Example, ObjectLabel
from clave.training.adapters import (
    CLASS_INDEX,
    classification_batches,
    detection_batches,
    policy_batches,
)


def example(classes: tuple[str, ...], visible: bool) -> Example:
    """Build an example whose labels are visible or not."""

    labels = tuple(
        ObjectLabel(
            object_id=index,
            material_class=material,
            channel="CH-PET",
            position=np.asarray((0.0, 0.0, 0.4), dtype=np.float64),
            in_reachable_window=True,
            bbox=(10, 10, 30, 30) if visible else None,
        )
        for index, material in enumerate(classes)
    )
    return Example(
        frame=np.zeros((32, 32, 3), dtype=np.uint8),
        labels=labels,
        simulated_time=0.0,
        seed=0,
        config_digest="d",
    )


def test_class_index_covers_the_whole_taxonomy() -> None:
    """A head sized to the taxonomy needs every class to have an index."""
    assert len(CLASS_INDEX) == 11


def test_classification_targets_are_multi_label() -> None:
    """A frame holds several objects, so one label per frame is ill posed."""
    pytest.importorskip("torch")
    images, targets = next(
        classification_batches((example(("M-01", "M-07"), True),), 1)
    )
    assert tuple(images.shape) == (1, 3, 32, 32)
    assert int(targets.sum()) == 2


def test_detection_drops_labels_whose_objects_are_not_in_frame() -> None:
    """An off-frame label has no pixels to regress toward."""
    pytest.importorskip("torch")
    assert not list(detection_batches((example(("M-01",), False),), 1))


def test_detection_yields_boxes_for_visible_labels() -> None:
    """Visible labels do train the detector."""
    pytest.importorskip("torch")
    batches = list(detection_batches((example(("M-01",), True),), 1))
    assert batches
    _, targets = batches[0]
    assert targets[0]["boxes"].shape == (1, 4)
    assert int(targets[0]["labels"][0]) == CLASS_INDEX["M-01"] + 1


def test_policy_batches_carry_a_zeroed_state() -> None:
    """The dataset has no proprioception, so the policy is vision only."""
    pytest.importorskip("torch")
    batches = list(policy_batches((example(("M-01",), True),), 1, 0.17))
    assert batches
    _, states, actions = batches[0]  # noqa: E501
    assert float(states.abs().sum()) == 0.0
    assert actions.shape == (1, 3)


def test_ac_mem_01_a_batch_is_resized_to_the_configured_side() -> None:
    """AC-MEM-01: the model sees the configured square, not the camera frame."""
    pytest.importorskip("torch")
    images, _targets = next(
        classification_batches((example(("M-01",), True),), 1, input_side=8)
    )
    assert tuple(images.shape) == (1, 3, 8, 8)


def test_ac_mem_02_boxes_scale_with_the_image() -> None:
    """AC-MEM-02: a box moves by the same factors as the resized frame."""
    pytest.importorskip("torch")
    batches = list(detection_batches((example(("M-01",), True),), 1, input_side=16))
    images, targets = batches[0]
    assert tuple(images[0].shape) == (3, 16, 16)
    box = [float(value) for value in targets[0]["boxes"][0]]
    assert box == pytest.approx([5.0, 5.0, 15.0, 15.0])


def test_classification_batches_with_augmentation() -> None:
    """Augmented classification batches preserve shapes and targets."""
    pytest.importorskip("torch")
    images, targets = next(
        classification_batches((example(("M-01", "M-07"), True),), 1, augment=True)
    )
    assert tuple(images.shape) == (1, 3, 32, 32)
    assert int(targets.sum()) == 2


def test_detection_batches_with_augmentation() -> None:
    """Augmented detection batches preserve box shape and class labels."""
    pytest.importorskip("torch")
    batches = list(detection_batches((example(("M-01",), True),), 1, augment=True))
    assert batches
    images, targets = batches[0]
    assert len(images) == 1
    assert targets[0]["boxes"].shape == (1, 4)
    assert int(targets[0]["labels"][0]) == CLASS_INDEX["M-01"] + 1
