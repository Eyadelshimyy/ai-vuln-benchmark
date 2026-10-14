"""The epoch a held-out score keeps."""

from pathlib import Path

import pytest

from clave.training.config import TrainingConfig, TrainingConfigError
from clave.training.selection import SelectionError, fresh_selection, metric_for


def _base() -> str:
    """A training file with the keys every run requires."""
    return (
        "training:\n"
        "  candidate: resnet50-baseline\n"
        "  dataset: datasets/synthetic\n"
        "  checkpoints: runs\n"
        "  epochs: 10\n"
        "  batch_size: 1\n"
        "  learning_rate: 0.0001\n"
        "  seed: 0\n"
        "  samples_per_crossing: 3\n"
        "  window_exit_meters: 1.034\n"
        "  act_chunk_size: 10\n"
        "  accumulation_steps: 1\n"
        "  class_balance: none\n"
    )


def test_ac_select_01_patience_belongs_with_a_validation_dataset(
    tmp_path: Path,
) -> None:
    """AC-SELECT-01: patience and the validation dataset arrive together."""
    plain = tmp_path / "plain.yml"
    plain.write_text(_base())
    loaded = TrainingConfig.load(plain)
    assert loaded.validation_dataset is None
    assert loaded.patience is None
    patient = tmp_path / "patient.yml"
    patient.write_text(_base() + "  patience: 3\n")
    with pytest.raises(TrainingConfigError, match="validation_dataset"):
        TrainingConfig.load(patient)
    half = tmp_path / "half.yml"
    half.write_text(_base() + "  validation_dataset: datasets/corpus/validation\n")
    with pytest.raises(TrainingConfigError, match="patience"):
        TrainingConfig.load(half)
    both = tmp_path / "both.yml"
    both.write_text(
        _base()
        + "  validation_dataset: datasets/corpus/validation\n"
        + "  patience: 3\n"
    )
    selected = TrainingConfig.load(both)
    assert selected.patience == 3
    assert selected.validation_dataset == Path("datasets/corpus/validation")


def test_ac_select_02_a_worse_score_counts_toward_patience() -> None:
    """AC-SELECT-02: a higher score replaces the best, and patience stops the run."""
    state, kept = fresh_selection().observe(0.2, epoch=0, patience=3)
    assert kept is True
    assert state.best_score == pytest.approx(0.2)
    assert state.best_epoch == 0
    assert state.stops(3) is False
    state, kept = state.observe(0.2, epoch=1, patience=3)
    assert kept is False
    assert state.epochs_without_improvement == 1
    state, _kept = state.observe(0.1, epoch=2, patience=3)
    state, kept = state.observe(0.05, epoch=3, patience=3)
    assert kept is False
    assert state.stops(3) is True
    assert state.best_epoch == 0
    improved, replaced = state.observe(0.4, epoch=4, patience=3)
    assert replaced is True
    assert improved.best_epoch == 4
    assert improved.epochs_without_improvement == 0


def test_ac_select_03_the_metric_follows_the_candidate() -> None:
    """AC-SELECT-03: agreement for the classifier, overlap for the detector."""
    assert metric_for("resnet50-baseline") == "agreement"
    assert metric_for("faster-rcnn-mobilenetv3") == "mean_iou"
    with pytest.raises(SelectionError, match="behavior-cloning-baseline"):
        metric_for("behavior-cloning-baseline")
