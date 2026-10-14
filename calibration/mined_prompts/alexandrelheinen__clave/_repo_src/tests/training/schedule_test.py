"""Which rollouts share an optimizer step."""

from pathlib import Path

import pytest

from clave.training.config import TrainingConfig, TrainingConfigError
from clave.training.schedule import accumulation_schedule

ROOT = Path(__file__).resolve().parents[2]


def test_ac_accum_01_one_step_keeps_archive_order(tmp_path: Path) -> None:
    """AC-ACCUM-01: one microbatch per window, archives in recorded order."""
    bare = (
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
        "  class_balance: none\n"
    )
    missing = tmp_path / "steps.yml"
    missing.write_text(bare)
    with pytest.raises(TrainingConfigError, match="accumulation_steps"):
        TrainingConfig.load(missing)
    present = tmp_path / "present.yml"
    present.write_text(bare + "  accumulation_steps: 1\n")
    assert TrainingConfig.load(present).accumulation_steps == 1
    zero = tmp_path / "zero.yml"
    zero.write_text(bare + "  accumulation_steps: 0\n")
    with pytest.raises(TrainingConfigError, match="accumulation_steps"):
        TrainingConfig.load(zero)
    picks = {"a": (0, 1, 2), "b": (0, 1)}
    windows = accumulation_schedule(
        picks,
        rollout_order=("a", "b"),
        batch_size=2,
        accumulation_steps=1,
        seed=0,
        epoch=0,
    )
    assert all(len(window) == 1 for window in windows)
    assert [window[0][0] for window in windows] == ["a", "a", "b"]
    with pytest.raises(ValueError, match="accumulation_steps"):
        accumulation_schedule(
            picks,
            rollout_order=("a",),
            batch_size=1,
            accumulation_steps=0,
            seed=0,
            epoch=0,
        )
    with pytest.raises(ValueError, match="batch_size"):
        accumulation_schedule(
            picks,
            rollout_order=("a",),
            batch_size=0,
            accumulation_steps=1,
            seed=0,
            epoch=0,
        )


def test_ac_accum_02_a_window_uses_distinct_rollouts() -> None:
    """AC-ACCUM-02: a full window takes one microbatch from each rollout."""
    picks = {"a": (0, 1), "b": (0, 1), "c": (0, 1), "d": (0, 1)}
    windows = accumulation_schedule(
        picks,
        rollout_order=("a", "b", "c", "d"),
        batch_size=1,
        accumulation_steps=3,
        seed=1,
        epoch=0,
    )
    first = windows[0]
    assert len(first) == 3
    assert len({rollout_id for rollout_id, _indexes in first}) == 3


def test_ac_accum_03_the_tail_may_repeat_a_rollout() -> None:
    """AC-ACCUM-03: a short tail may reuse a rollout, and the seed repeats."""
    picks = {"a": (0, 1, 2, 3), "b": (0, 1, 2, 3)}
    windows = accumulation_schedule(
        picks,
        rollout_order=("a", "b"),
        batch_size=1,
        accumulation_steps=3,
        seed=2,
        epoch=4,
    )
    again = accumulation_schedule(
        picks,
        rollout_order=("a", "b"),
        batch_size=1,
        accumulation_steps=3,
        seed=2,
        epoch=4,
    )
    assert windows == again
    repeated = [
        window
        for window in windows
        if len({rollout_id for rollout_id, _indexes in window}) < len(window)
    ]
    assert repeated
