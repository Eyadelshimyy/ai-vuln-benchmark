"""Which frames a crossing keeps, and which pick a checkpoint reloads."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from clave.data.dataset import DatasetDescription, DatasetFile
from clave.training.config import TrainingConfig, TrainingConfigError
from clave.training.sample import (
    FrameSample,
    SampleError,
    build_frame_sample,
    capture_interval_seconds,
    crossing_stride,
    kept_indexes,
    resolve_stored_sample,
)
from clave.world.config import load
from clave.world.scene import detection_along_travel_meters

ROOT = Path(__file__).resolve().parents[2]
ALONG = 0.920
INTERVAL = 0.2


def _file(name: str, speed: float | None, frames: int = 150) -> DatasetFile:
    """One archive record."""
    return DatasetFile(
        name,
        "a" * 64,
        10,
        frames,
        0,
        belt_speed_meters_per_second=speed,
    )


def _description(
    files: tuple[DatasetFile, ...], members: list[str]
) -> DatasetDescription:
    """A training description that names those archives."""
    return DatasetDescription(
        digest="b" * 64,
        seed=0,
        config_digest="c" * 64,
        example_count=sum(item.frame_count for item in files),
        parts={"train": members},
        composition={},
        files=files,
    )


def test_ac_sample_01_stride_keeps_n_looks_per_crossing() -> None:
    """AC-SAMPLE-01: floor(K / N) frames apart, or every frame when that is 0."""
    speeds = (0.25, 0.30, 0.35)
    expected = {3: (6, 5, 4), 5: (3, 3, 2)}
    for samples, strides in expected.items():
        for speed, stride in zip(speeds, strides, strict=True):
            crossing = ALONG / (speed * INTERVAL)
            assert crossing_stride(crossing, samples) == stride
            indexes = kept_indexes(
                150,
                belt_speed=speed,
                along_travel_meters=ALONG,
                capture_interval_seconds=INTERVAL,
                samples_per_crossing=samples,
                phase=0,
            )
            assert indexes[0] == 0
            assert indexes[1] - indexes[0] == stride
    short = ALONG / (0.35 * INTERVAL)
    assert math.floor(short / 100) < 1
    assert crossing_stride(short, 100) == 1
    assert kept_indexes(
        150,
        belt_speed=0.35,
        along_travel_meters=ALONG,
        capture_interval_seconds=INTERVAL,
        samples_per_crossing=100,
        phase=0,
    ) == tuple(range(150))


def test_ac_sample_02_a_rollout_without_belt_speed_keeps_every_frame() -> None:
    """AC-SAMPLE-02: no speed means the crossing length is unknown."""
    description = _description(
        (_file("rollout_000.npz", None),),
        ["rollout_000"],
    )
    sample = build_frame_sample(
        description,
        along_travel_meters=ALONG,
        capture_interval_seconds=INTERVAL,
        samples_per_crossing=3,
        seed=0,
    )
    assert sample.picks["rollout_000"] == tuple(range(150))


def test_ac_sample_03_the_same_seed_draws_the_same_indexes() -> None:
    """AC-SAMPLE-03: the phase is part of the stored pick, and it stays put."""
    description = _description(
        (_file("rollout_000.npz", 0.25), _file("rollout_001.npz", 0.35)),
        ["rollout_000", "rollout_001"],
    )
    first = build_frame_sample(
        description,
        along_travel_meters=ALONG,
        capture_interval_seconds=INTERVAL,
        samples_per_crossing=3,
        seed=0,
    )
    second = build_frame_sample(
        description,
        along_travel_meters=ALONG,
        capture_interval_seconds=INTERVAL,
        samples_per_crossing=3,
        seed=0,
    )
    assert first.picks == second.picks
    assert first.digest == second.digest
    for indexes in first.picks.values():
        stride = indexes[1] - indexes[0]
        assert 0 <= indexes[0] < stride
        assert indexes == tuple(range(indexes[0], 150, stride))


def test_ac_sample_04_the_configuration_names_samples_per_crossing(
    tmp_path: Path,
) -> None:
    """AC-SAMPLE-04: N is a training setting, and two values are different runs."""
    shipped = TrainingConfig.load(ROOT / "configs" / "training" / "default.yml")
    assert shipped.samples_per_crossing == 3
    bare = (
        "training:\n"
        "  candidate: resnet50-baseline\n"
        "  dataset: datasets/synthetic\n"
        "  checkpoints: runs\n"
        "  epochs: 1\n"
        "  batch_size: 1\n"
        "  learning_rate: 0.0001\n"
        "  seed: 0\n"
        "  window_exit_meters: 1.034\n"
        "  act_chunk_size: 10\n"
    )
    missing = tmp_path / "missing.yml"
    missing.write_text(bare)
    with pytest.raises(TrainingConfigError, match="samples_per_crossing"):
        TrainingConfig.load(missing)
    three = tmp_path / "three.yml"
    five = tmp_path / "five.yml"
    tail = "  accumulation_steps: 1\n  class_balance: none\n"
    three.write_text(bare + "  samples_per_crossing: 3\n" + tail)
    five.write_text(bare + "  samples_per_crossing: 5\n" + tail)
    assert TrainingConfig.load(five).samples_per_crossing == 5
    assert TrainingConfig.load(three).digest != TrainingConfig.load(five).digest


def test_ac_sample_05_a_checkpoint_reloads_its_stored_pick() -> None:
    """AC-SAMPLE-05: resume uses the stored indexes, and a different N is refused."""
    description = _description((_file("rollout_000.npz", 0.25),), ["rollout_000"])
    stored = build_frame_sample(
        description,
        along_travel_meters=ALONG,
        capture_interval_seconds=INTERVAL,
        samples_per_crossing=3,
        seed=1,
    )
    assert (
        resolve_stored_sample(stored, checkpoint=True, samples_per_crossing=3) is stored
    )
    assert resolve_stored_sample(None, checkpoint=False, samples_per_crossing=3) is None
    with pytest.raises(SampleError, match="no stored frame sample"):
        resolve_stored_sample(None, checkpoint=True, samples_per_crossing=3)
    with pytest.raises(SampleError, match="asks for 5"):
        resolve_stored_sample(stored, checkpoint=True, samples_per_crossing=5)


def test_a_stored_pick_roundtrips_through_its_manifest(tmp_path: Path) -> None:
    """The manifest is the pick a later run reloads."""
    description = _description(
        (_file("rollout_000.npz", 0.25, frames=4),),
        ["rollout_000"],
    )
    sample = build_frame_sample(
        description,
        along_travel_meters=ALONG,
        capture_interval_seconds=INTERVAL,
        samples_per_crossing=3,
        seed=1,
    )
    assert sample.frame_count == len(sample.picks["rollout_000"])
    path = tmp_path / "resnet50-baseline.sample.json"
    sample.write(path)
    stored = json.loads(path.read_text())
    assert stored["digest"] == sample.digest
    assert FrameSample.from_payload(stored["sample"]) == sample


def test_a_stored_pick_refuses_a_field_that_is_not_the_one_it_wrote() -> None:
    """A checkpoint with a damaged pick does not train on a guess."""
    description = _description(
        (_file("rollout_000.npz", None, frames=2),),
        ["rollout_000"],
    )
    payload = build_frame_sample(
        description,
        along_travel_meters=ALONG,
        capture_interval_seconds=INTERVAL,
        samples_per_crossing=1,
        seed=0,
    ).payload()
    with pytest.raises(SampleError, match="no picks"):
        FrameSample.from_payload({**payload, "picks": []})
    with pytest.raises(SampleError, match="not a list"):
        FrameSample.from_payload({**payload, "picks": {"rollout_000": (0, 1)}})
    damaged = dict(payload)
    del damaged["seed"]
    with pytest.raises(SampleError, match="missing"):
        FrameSample.from_payload(damaged)
    with pytest.raises(SampleError, match="integer"):
        FrameSample.from_payload({**payload, "seed": True})
    with pytest.raises(SampleError, match="number"):
        FrameSample.from_payload({**payload, "along_travel_meters": True})


def test_a_crossing_refuses_fewer_than_one_look() -> None:
    """N below one is not a sample."""
    with pytest.raises(SampleError, match="at least one look"):
        crossing_stride(10.0, 0)


def test_a_part_the_description_does_not_have_is_refused() -> None:
    """Validation is sampled only when the description has that part."""
    description = _description((_file("rollout_000.npz", None),), ["rollout_000"])
    with pytest.raises(SampleError, match="no validation part"):
        build_frame_sample(
            description,
            along_travel_meters=ALONG,
            capture_interval_seconds=INTERVAL,
            samples_per_crossing=1,
            seed=0,
            part="validation",
        )


def test_a_rollout_without_a_file_record_needs_the_archive(tmp_path: Path) -> None:
    """A legacy description counts frames from the archive itself."""
    description = _description((), ["rollout_000"])
    with pytest.raises(SampleError, match="no file record"):
        build_frame_sample(
            description,
            along_travel_meters=ALONG,
            capture_interval_seconds=INTERVAL,
            samples_per_crossing=1,
            seed=0,
        )
    frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    times = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float64)
    np.savez(tmp_path / "rollout_000.npz", frames=frames, times=times)
    sample = build_frame_sample(
        description,
        along_travel_meters=ALONG,
        capture_interval_seconds=INTERVAL,
        samples_per_crossing=3,
        seed=0,
        dataset=tmp_path,
    )
    assert sample.picks["rollout_000"] == (0, 1, 2, 3)
    assert capture_interval_seconds(tmp_path, "rollout_000") == pytest.approx(0.2)


def test_a_capture_interval_needs_two_increasing_timestamps(tmp_path: Path) -> None:
    """One timestamp, or a clock that does not advance, is not an interval."""
    single = tmp_path / "one.npz"
    np.savez(single, times=np.array([0.0]))
    with pytest.raises(SampleError, match="fewer than two"):
        capture_interval_seconds(tmp_path, "one")
    stuck = tmp_path / "stuck.npz"
    np.savez(stuck, times=np.array([1.0, 1.0]))
    with pytest.raises(SampleError, match="capture interval"):
        capture_interval_seconds(tmp_path, "stuck")


def test_ac_sample_06_the_shipped_gate_covers_0_920_m_along_travel() -> None:
    """AC-SAMPLE-06: the crossing length is the along-belt footprint, not fovy."""
    raw = load(ROOT / "configs" / "world" / "sorting_line.yml")
    assert detection_along_travel_meters(raw) == pytest.approx(0.920, abs=0.001)
