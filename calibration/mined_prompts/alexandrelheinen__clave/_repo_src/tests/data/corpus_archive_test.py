"""Format-2 archives round-trip without a simulator."""

from pathlib import Path

import numpy as np

from clave.data.dataset import FORMAT_CORPUS, load_split, write
from clave.data.examples import CameraCapture, Example, ObjectLabel, Rollout


def _label() -> ObjectLabel:
    return ObjectLabel(
        object_id=7,
        material_class="M-02",
        channel="CH-HDPE",
        position=np.asarray((0.1, -0.02, 0.91), dtype=np.float64),
        in_reachable_window=True,
        bbox=(1, 2, 3, 4),
        orientation=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.3, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.1),
        object_name="silicon_bottle",
    )


def test_a_corpus_archive_round_trips_cameras_and_pose(tmp_path: Path) -> None:
    """AC-CORPUS-02: cameras, serial, pose and the instance map survive."""
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    instance = np.zeros((4, 5), dtype=np.uint16)
    instance[1, 2] = 7
    other = frame.copy()
    other[0, 0] = 9
    captures = (
        CameraCapture("gate_wide", frame, (_label(),), instance),
        CameraCapture("pick_wide", other, (_label(),), instance),
    )
    example = Example(
        frame=frame,
        labels=captures[0].labels,
        simulated_time=1.5,
        seed=3,
        config_digest="world",
        captures=captures,
    )
    rollout = Rollout(
        "rollout_000",
        seed=3,
        examples=(example,),
        belt_speed=0.30,
        spacing_meters=0.35,
    )
    written = write(
        tmp_path,
        (rollout,),
        {"train": ("rollout_000",)},
        seed=0,
        config_digest="world",
        role="train",
        campaign_id="campaign",
    )
    assert written.format_version == FORMAT_CORPUS
    assert written.files[0].belt_speed_meters_per_second == 0.30
    assert written.files[0].spacing_meters == 0.35
    assert written.files[0].camera_ids == ("gate_wide", "pick_wide")

    restored = load_split(tmp_path, "train", camera="pick_wide")
    got = restored[0].examples[0]
    assert got.frame[0, 0, 0] == 9
    assert got.labels[0].object_id == 7
    assert got.labels[0].orientation == (1.0, 0.0, 0.0, 0.0)
    assert got.labels[0].linear_velocity == (0.3, 0.0, 0.0)
    assert got.labels[0].object_name == "silicon_bottle"
    assert got.captures[0].instance_ids is not None
    assert int(got.captures[0].instance_ids[1, 2]) == 7
    assert restored[0].belt_speed == 0.30
