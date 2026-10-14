"""Tests for training configuration."""

from pathlib import Path

import pytest

from clave.training.config import TrainingConfig, TrainingConfigError

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "configs" / "training" / "default.yml"


def test_the_shipped_configuration_loads() -> None:
    """Every hyperparameter comes from a file."""

    config = TrainingConfig.load(SHIPPED)
    assert config.epochs > 0
    assert config.batch_size > 0
    assert config.learning_rate > 0


def test_the_candidate_can_be_overridden_so_one_file_serves_all() -> None:
    """Training a different architecture is not a code change."""
    assert TrainingConfig.load(SHIPPED, "act").candidate == "act"


def test_a_missing_key_fails_naming_it(tmp_path: Path) -> None:
    """Nothing is defaulted in code."""
    path = tmp_path / "bad.yml"
    path.write_text("training:\n  candidate: x\n")
    with pytest.raises(TrainingConfigError, match="training.dataset"):
        TrainingConfig.load(path)


def test_a_file_without_a_training_section_is_refused(tmp_path: Path) -> None:
    """A configuration that is not one should say so."""
    path = tmp_path / "empty.yml"
    path.write_text("other: 1\n")
    with pytest.raises(TrainingConfigError, match="no 'training' section"):
        TrainingConfig.load(path)


def test_a_count_below_one_is_refused(tmp_path: Path) -> None:
    """Looks, balance, and patience are integers the loop can use."""
    body = (
        "training:\n"
        "  candidate: resnet50-baseline\n"
        "  dataset: datasets/synthetic\n"
        "  checkpoints: runs\n"
        "  epochs: 1\n"
        "  batch_size: 1\n"
        "  learning_rate: 0.0001\n"
        "  seed: 0\n"
        "  samples_per_crossing: 3\n"
        "  window_exit_meters: 1.034\n"
        "  act_chunk_size: 10\n"
        "  accumulation_steps: 1\n"
        "  class_balance: none\n"
    )
    looks = tmp_path / "looks.yml"
    looks.write_text(body.replace("samples_per_crossing: 3", "samples_per_crossing: 0"))
    with pytest.raises(TrainingConfigError, match="samples_per_crossing"):
        TrainingConfig.load(looks)
    balance = tmp_path / "balance.yml"
    balance.write_text(body.replace("class_balance: none", "class_balance: weighted"))
    with pytest.raises(TrainingConfigError, match="class_balance"):
        TrainingConfig.load(balance)
    patience = tmp_path / "patience.yml"
    patience.write_text(
        body + "  validation_dataset: datasets/corpus/validation\n  patience: 0\n"
    )
    with pytest.raises(TrainingConfigError, match="patience"):
        TrainingConfig.load(patience)


def test_the_digest_covers_the_settings_that_change_a_run() -> None:
    """Two runs are comparable by value."""
    first = TrainingConfig.load(SHIPPED)
    second = TrainingConfig.load(SHIPPED)
    assert first.digest == second.digest
    assert TrainingConfig.load(SHIPPED, "act").digest != first.digest
