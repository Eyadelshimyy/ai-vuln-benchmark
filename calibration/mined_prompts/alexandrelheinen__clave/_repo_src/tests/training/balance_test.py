"""Positive weights for the classes the pick actually shows."""

import logging
from pathlib import Path

import pytest

from clave.training.balance import (
    BalanceError,
    inverse_pos_weight,
    resolve_class_weights,
)
from clave.training.config import TrainingConfig, TrainingConfigError

ROOT = Path(__file__).resolve().parents[2]


def _text(balance: str, candidate: str = "resnet50-baseline") -> str:
    """A training file that differs only in the balance mode."""
    return (
        "training:\n"
        f"  candidate: {candidate}\n"
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
        f"  class_balance: {balance}\n"
    )


def test_ac_balance_01_inverse_is_only_for_the_classifier(tmp_path: Path) -> None:
    """AC-BALANCE-01: inverse is a classifier setting, and it changes the digest."""
    missing = tmp_path / "missing.yml"
    missing.write_text(_text("none").replace("  class_balance: none\n", ""))
    with pytest.raises(TrainingConfigError, match="class_balance"):
        TrainingConfig.load(missing)
    detector = tmp_path / "detector.yml"
    detector.write_text(_text("inverse", "faster-rcnn-mobilenetv3"))
    with pytest.raises(TrainingConfigError, match="resnet50-baseline"):
        TrainingConfig.load(detector)
    none_path = tmp_path / "none.yml"
    inverse_path = tmp_path / "inverse.yml"
    none_path.write_text(_text("none"))
    inverse_path.write_text(_text("inverse"))
    none = TrainingConfig.load(none_path)
    inverse = TrainingConfig.load(inverse_path)
    assert none.class_balance == "none"
    assert inverse.class_balance == "inverse"
    assert none.digest != inverse.digest
    shipped = TrainingConfig.load(
        ROOT / "configs" / "training" / "classification_full.yml"
    )
    assert shipped.class_balance == "inverse"


def test_ac_balance_02_a_rare_class_outweighs_a_common_one() -> None:
    """AC-BALANCE-02: weight is the negative count over the positive count."""
    weights = inverse_pos_weight([8, 2, 0], frames=10)
    assert weights[0] == pytest.approx(2 / 8)
    assert weights[1] == pytest.approx(8 / 2)
    assert weights[2] == 1.0
    assert inverse_pos_weight([0, 0], frames=0) == (1.0, 1.0)


def test_ac_balance_03_a_resume_reloads_the_stored_weights() -> None:
    """AC-BALANCE-03: a resume keeps the vector that trained these weights."""
    stored = (1.0, 4.0)
    assert (
        resolve_class_weights(stored, checkpoint=True, mode="inverse", length=2)
        == stored
    )
    assert resolve_class_weights(stored, checkpoint=True, mode="none", length=2) is None
    assert (
        resolve_class_weights(None, checkpoint=False, mode="inverse", length=2) is None
    )
    with pytest.raises(BalanceError, match="no class weights"):
        resolve_class_weights(None, checkpoint=True, mode="inverse", length=2)
    with pytest.raises(BalanceError, match="taxonomy"):
        resolve_class_weights((1.0,), checkpoint=True, mode="inverse", length=2)


def test_class_weight_count_logs_before_the_first_archive(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The weight walk names its length before it opens an archive."""
    from clave.training.runner import _count_positives
    from clave.training.sample import FrameSample

    logged_before_open: list[bool] = []

    def _open(
        dataset: object,
        description: object,
        rollout_id: str,
        indexes: tuple[int, ...],
    ) -> tuple[object, ...]:
        del dataset, description, rollout_id
        logged_before_open.append(
            any(
                "class weights 0/2 rollouts" in record.message
                for record in caplog.records
            )
        )
        return (type("Example", (), {"visible_labels": ()})(),) * len(indexes)

    monkeypatch.setattr("clave.training.runner._examples_at", _open)
    sample = FrameSample(
        samples_per_crossing=1,
        dataset_digest="digest",
        along_travel_meters=0.92,
        capture_interval_seconds=0.2,
        seed=0,
        picks={"rollout_000": (0,), "rollout_001": (1,)},
    )
    with caplog.at_level(logging.INFO, logger="clave.training.runner"):
        _counts, frames = _count_positives(Path("."), None, sample)
    assert frames == 2
    assert logged_before_open == [True, True]
