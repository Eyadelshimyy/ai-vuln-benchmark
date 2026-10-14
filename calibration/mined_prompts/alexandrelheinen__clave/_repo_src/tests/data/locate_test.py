"""Resolving a dataset by role."""

from pathlib import Path

import numpy as np
import pytest

from clave.data.dataset import DatasetError, write
from clave.data.examples import CameraCapture, Example, ObjectLabel, Rollout
from clave.data.locate import resolve_dataset

ROOT = Path(__file__).resolve().parents[2]


def _corpus(directory: Path, role: str, world: str) -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    label = ObjectLabel(
        1,
        "M-02",
        "CH-HDPE",
        np.zeros(3, dtype=np.float64),
        True,
    )
    capture = CameraCapture("gate_wide", frame, (label,), None)
    example = Example(
        frame=frame,
        labels=(label,),
        simulated_time=0.0,
        seed=0,
        config_digest=world,
        captures=(capture,),
    )
    write(
        directory,
        (Rollout("rollout_000", 0, (example,), belt_speed=0.25),),
        {role: ("rollout_000",)},
        0,
        world,
        role=role,
        campaign_id="c",
    )


def test_train_refuses_a_validation_corpus(tmp_path: Path) -> None:
    """AC-CORPUS-04: train does not read the validation half."""
    _corpus(tmp_path, "validation", "world")
    with pytest.raises(DatasetError, match="validation corpus"):
        resolve_dataset(ROOT, tmp_path, role="train")


def test_a_corpus_from_another_world_is_refused(tmp_path: Path) -> None:
    """AC-CORPUS-04: a corpus digest from another world is refused."""
    _corpus(tmp_path, "train", "not-the-tree")
    with pytest.raises(DatasetError, match="world"):
        resolve_dataset(ROOT, tmp_path, role="train")
