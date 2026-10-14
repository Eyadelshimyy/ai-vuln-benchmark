"""Checkpoint fields and the held-out score, without loading a model."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from clave.data.dataset import DatasetDescription, DatasetFile
from clave.training.config import TrainingConfig
from clave.training.objectives import OBJECTIVES
from clave.training.runner import (
    EpochRecord,
    TrainingRun,
    _candidate_for,
    _checkpoint_path,
    _checkpoint_payload,
    _draw_sample,
    _objective,
    _record_selection,
    _sample_path,
    _select_epoch,
    _selection_from,
    _validation_from,
    _validation_sample_path,
)
from clave.training.sample import FrameSample, SampleError
from clave.training.selection import SelectionError, SelectionState, fresh_selection

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "configs" / "training" / "default.yml"
WORLD = ROOT / "configs" / "world" / "sorting_line.yml"
THRESHOLDS = ROOT / "configs" / "validation" / "corpus.yml"


class _Weights:
    """A stand-in for the tensors a checkpoint stores."""

    def state_dict(self) -> dict[str, int]:
        return {"weight": 1}

    def eval(self) -> None:
        """The selection pass asks the model to stop training."""


def _config(tmp_path: Path) -> TrainingConfig:
    """A configuration that writes into a temporary directory."""
    return replace(
        TrainingConfig.load(SHIPPED),
        dataset=tmp_path,
        checkpoints=tmp_path,
    )


def _sample(looks: int = 3) -> FrameSample:
    """A pick with nothing left to read."""
    return FrameSample(
        samples_per_crossing=looks,
        dataset_digest="a" * 64,
        along_travel_meters=0.92,
        capture_interval_seconds=0.2,
        seed=0,
        picks={},
    )


def _run() -> TrainingRun:
    """An empty run record."""
    return TrainingRun(
        candidate="resnet50-baseline",
        seed=0,
        config_digest="b" * 64,
        dataset_digest="c" * 64,
        machine="test",
        threads=1,
        environment={},
        input_side_pixels=224,
        resident_limit_bytes=1,
    )


def test_a_run_record_is_the_json_a_later_tool_reads(tmp_path: Path) -> None:
    """The record names the candidate and survives without this package."""
    run = _run()
    run.epochs.append(EpochRecord(0, 0.2, 1.0, 0.5))
    path = tmp_path / "resnet50-baseline.run.json"
    run.write(path)
    assert '"resnet50-baseline"' in path.read_text()
    with pytest.raises(KeyError, match="no candidate"):
        _candidate_for("not-a-candidate")


def test_a_checkpoint_records_the_pick_and_the_selection(tmp_path: Path) -> None:
    """Resume and the preferred epoch read the same mapping."""
    config = _config(tmp_path)
    sample = _sample()
    selection = SelectionState(0.4, 1, 0)
    bare = _checkpoint_payload(
        config,
        _Weights(),
        epoch=1,
        input_side_pixels=224,
        sample=sample,
        weights=None,
        selection=selection,
        validation_sample=None,
        optimizer=None,
    )
    assert "optimizer" not in bare
    assert bare["class_weights"] is None
    assert bare["selection"]["validation_sample"] is None
    assert bare["input_side_pixels"] == 224
    held = _sample(looks=3)
    full = _checkpoint_payload(
        config,
        _Weights(),
        epoch=1,
        input_side_pixels=224,
        sample=sample,
        weights=(1.0, 4.0),
        selection=selection,
        validation_sample=held,
        optimizer=_Weights(),
    )
    assert full["optimizer"] == {"weight": 1}
    assert full["class_weights"] == [1.0, 4.0]
    assert full["selection"]["validation_sample"] == held.payload()
    assert _checkpoint_path(config) == tmp_path / "resnet50-baseline.pt"
    assert _sample_path(config) == tmp_path / "resnet50-baseline.sample.json"
    assert _validation_sample_path(config) == (
        tmp_path / "resnet50-baseline.validation-sample.json"
    )


def test_a_resume_reads_the_selection_the_checkpoint_stored(tmp_path: Path) -> None:
    """A missing selection starts over, and a different N is refused."""
    assert _selection_from(None) == fresh_selection()
    stored = _selection_from(
        {"best_score": 0.4, "best_epoch": 1, "epochs_without_improvement": 2}
    )
    assert stored == SelectionState(0.4, 1, 2)
    blank = _selection_from({"epochs_without_improvement": "stale"})
    assert blank.epochs_without_improvement == 0
    config = _config(tmp_path)
    assert _validation_from(None, config) is None
    assert _validation_from({}, config) is None
    sample = _sample()
    assert _validation_from({"validation_sample": sample.payload()}, config) == sample
    other = replace(config, samples_per_crossing=5)
    with pytest.raises(SampleError, match="asks for 5"):
        _validation_from({"validation_sample": sample.payload()}, other)


def test_the_run_record_keeps_the_best_held_out_score(tmp_path: Path) -> None:
    """The last checkpoint and the run record name the same selection."""
    run = _run()
    selection = SelectionState(0.4, 1, 2)
    _record_selection(run, selection, (1.0, 2.0), "agreement")
    assert run.best_score == pytest.approx(0.4)
    assert run.best_epoch == 1
    assert run.epochs_without_improvement == 2
    assert run.selection_metric == "agreement"
    assert run.class_weights == [1.0, 2.0]
    _record_selection(run, fresh_selection(), None, None)
    assert run.class_weights is None
    assert _objective("resnet50-baseline", None) is OBJECTIVES["resnet50-baseline"]
    assert callable(_objective("resnet50-baseline", (1.0,)))
    assert (
        _select_epoch(
            _config(tmp_path), _Weights(), object(), _sample(), 0, 224, THRESHOLDS
        )
        is None
    )


def test_an_empty_validation_pick_scores_zero(tmp_path: Path) -> None:
    """No kept frame is a score of zero, and the cuts have to be named."""
    config = replace(
        _config(tmp_path),
        validation_dataset=tmp_path,
        patience=3,
    )
    assert (
        _select_epoch(config, _Weights(), object(), _sample(), 0, 224, THRESHOLDS)
        == 0.0
    )
    detector = replace(config, candidate="faster-rcnn-mobilenetv3")
    assert (
        _select_epoch(detector, _Weights(), object(), _sample(), 0, 224, THRESHOLDS)
        == 0.0
    )
    with pytest.raises(SelectionError, match="thresholds"):
        _select_epoch(config, _Weights(), object(), _sample(), 0, 224, None)


def test_a_pick_uses_the_archive_when_the_rollout_recorded_a_speed(
    tmp_path: Path,
) -> None:
    """Belt speed turns the footprint into a stride, and a missing speed does not."""
    config = _config(tmp_path)
    quiet = DatasetDescription(
        digest="d" * 64,
        seed=0,
        config_digest="e" * 64,
        example_count=4,
        parts={"train": ["rollout_000"]},
        composition={},
        files=(
            DatasetFile(
                "rollout_000.npz",
                "f" * 64,
                10,
                4,
                0,
                belt_speed_meters_per_second=None,
            ),
        ),
    )
    kept = _draw_sample(config, quiet, WORLD, tmp_path, "train")
    assert kept.picks["rollout_000"] == (0, 1, 2, 3)
    times = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float64)
    np.savez(tmp_path / "rollout_001.npz", times=times)
    moving = DatasetDescription(
        digest="d" * 64,
        seed=0,
        config_digest="e" * 64,
        example_count=30,
        parts={"validation": ["rollout_001"]},
        composition={},
        files=(
            DatasetFile(
                "rollout_001.npz",
                "f" * 64,
                10,
                30,
                0,
                belt_speed_meters_per_second=0.25,
            ),
        ),
    )
    drawn = _draw_sample(config, moving, WORLD, tmp_path, "validation")
    assert drawn.capture_interval_seconds == pytest.approx(0.2)
    assert drawn.picks["rollout_001"]
