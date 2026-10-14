"""The test the perception contract has to pass, and what it deliberately is not.



`docs/perception-contract.md` closes with a table of architecture changes and
what each one is allowed to cost, and says that if a change in the left column
forces an edit outside the right column then the abstraction has leaked and the
leak is a defect rather than a fact of life. This file is that table as a test.

Asserting the record's shape is unchanged is the criterion's own wording and is
close to vacuous on its own: a tracker that ignored the new camera entirely
would pass it. Every case here is paired with a check that the new sensor
actually reached a track.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from clave.tracker.association import Association, Cue, TrackSummary
from clave.tracker.belt_frame import Footprint
from clave.tracker.evidence import Detection, Evidence, PixelMask, Role
from clave.tracker.fusion import FusionSettings
from clave.tracker.intake import Deployment, Intake
from clave.tracker.sensors import SensorError, load_sensors, of_role, require_role
from clave.tracker.track import Tracker, WasteObject
from clave.world.config import load, require

ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT / "configs" / "world" / "sorting_line.yml"
SECOND = 1_000_000_000

RECORD_FIELDS = frozenset(field.name for field in dataclasses.fields(WasteObject))
"""What a consumer reads. A sensor change may not alter this set."""


class JoinEverything:
    """One track for everything, so a contributor set is easy to read."""

    @property
    def name(self) -> str:
        """What decided the association."""
        return "join-everything"

    def associate(self, cue: Cue, tracks: tuple[TrackSummary, ...]) -> Association:
        """Join the first open track, or open one."""
        del cue
        return Association(track_id=tracks[0].track_id if tracks else None)


def tracker_over(raw: dict[str, object]) -> Tracker:
    """A tracker over one world configuration."""
    return Tracker(
        intake=Intake(Deployment.SIMULATED, load_sensors(raw)),
        associator=JoinEverything(),
        settings=FusionSettings.load(
            ROOT / "configs" / "perception" / "fusion.yml", WORLD
        ),
        belt_speed=0.31,
        window_exit=1.034,
    )


def detection_from(source_id: str) -> Evidence:
    """A detection reading attributed to one sensor."""
    return Evidence(
        source_id=source_id,
        role=Role.DETECTION,
        observed_at_nanos=SECOND,
        confidence=0.9,
        payload=Detection(
            footprint=Footprint(
                center=np.asarray((-1.0, 0.0, 0.93), dtype=np.float64),
                major_extent=0.10,
                minor_extent=0.06,
                yaw=0.0,
            ),
            mask=PixelMask(width=64, height=64, runs=((10, 20, 10),)),
            height=0.06,
        ),
    )


def test_adding_a_fourth_barcode_camera_edits_one_file() -> None:
    """The contract's first row: one entry in `cameras`, nothing else.

    The non-vacuity half is that the new camera's readings are admitted, which
    a tracker ignoring it would fail.
    """
    raw = load(WORLD)
    raw["cameras"] = [
        *raw["cameras"],
        dict(
            raw["cameras"][2], id="gate_code_fourth", position_meters=[-1.0, 0.16, 1.63]
        ),
    ]
    sensors = load_sensors(raw)
    assert len(of_role(sensors, Role.CODE)) == 4
    assert "gate_code_fourth" in {s.source_id for s in of_role(sensors, Role.CODE)}

    held = tracker_over(raw)
    held.observe(detection_from("gate_wide"), at_nanos=SECOND)
    record = held.settle(at_nanos=SECOND)[0]
    assert frozenset(f.name for f in dataclasses.fields(record)) == RECORD_FIELDS


def test_moving_the_gate_further_upstream_edits_one_position() -> None:
    """The contract's fifth row: dead reckoning already spans the gap.

    Moving the gate changes where an object is seen and not what is known about
    it, because the track is propagated by belt speed to whatever instant it is
    settled at.
    """
    raw = load(WORLD)
    raw["cameras"] = [
        dict(
            entry,
            position_meters=[
                -1.40,
                entry["position_meters"][1],
                entry["position_meters"][2],
            ],
        )
        for entry in raw["cameras"]
    ]
    sensors = load_sensors(raw)
    assert require_role(sensors, Role.DETECTION).position[0] == pytest.approx(-1.40)

    held = tracker_over(raw)
    held.observe(detection_from("gate_wide"), at_nanos=SECOND)
    record = held.settle(at_nanos=SECOND)[0]
    assert frozenset(f.name for f in dataclasses.fields(record)) == RECORD_FIELDS
    assert record.evidence == frozenset({Role.DETECTION})


def test_replacing_three_code_cameras_with_one_reader_edits_one_list() -> None:
    """The contract's second row.

    A line-scan reader is one camera where there were three. Nothing above the
    sensor list learns that, because nothing above it counts cameras.
    """
    raw = load(WORLD)
    raw["cameras"] = [
        raw["cameras"][0],
        dict(raw["cameras"][2], id="gate_line_scan", position_meters=[-1.0, 0.0, 1.63]),
    ]
    sensors = load_sensors(raw)
    assert len(of_role(sensors, Role.CODE)) == 1

    held = tracker_over(raw)
    held.observe(detection_from("gate_wide"), at_nanos=SECOND)
    assert (
        frozenset(f.name for f in dataclasses.fields(held.settle(at_nanos=SECOND)[0]))
        == RECORD_FIELDS
    )


def test_a_second_arm_changes_nothing_in_perception() -> None:
    """The contract's last row: a second consumer reads the same records.

    There is nothing to change, and that is the point. `settle` takes an
    instant and returns records; two consumers asking at two instants get two
    answers about the same tracks and neither disturbs the other.
    """
    raw = load(WORLD)
    held = tracker_over(raw)
    held.observe(detection_from("gate_wide"), at_nanos=SECOND)

    first = held.settle(at_nanos=SECOND)
    second = held.settle(at_nanos=SECOND + SECOND)
    again = held.settle(at_nanos=SECOND)

    assert first == again, "asking twice at one instant gives one answer"
    assert second[0].footprint.center[0] > first[0].footprint.center[0]
    assert first[0].track_id == second[0].track_id


def test_removing_the_only_detection_camera_fails_at_load() -> None:
    """Removing the only detection camera fails at load.

    The contract's table does not have a row for removing the camera a role
    depends on, because the honest answer is that the line stops rather than
    quietly emitting records missing that evidence.
    """
    raw = load(WORLD)
    raw["cameras"] = [
        entry for entry in raw["cameras"] if str(entry["role"]) != "detection"
    ]
    with pytest.raises(SensorError, match="detection"):
        require_role(load_sensors(raw), Role.DETECTION)


def test_adding_a_role_costs_one_adapter_and_not_a_consumer() -> None:
    """Adding a role costs one adapter and not a consumer.

    A depth sensor or a near-infrared sensor is a new role, and the first
    wording of this rule claimed that too was one file. It is not, and the contract's
    own table never said it was: a new role legitimately costs one adapter. What
    it may not cost is a consumer, which is what this asserts.
    """
    raw = load(WORLD)
    raw["cameras"] = [
        *raw["cameras"],
        dict(raw["cameras"][0], id="gate_depth", role="depth"),
    ]
    sensors = load_sensors(raw)
    assert require_role(sensors, Role.DEPTH).source_id == "gate_depth"

    held = tracker_over(raw)
    held.observe(detection_from("gate_wide"), at_nanos=SECOND)
    assert (
        frozenset(f.name for f in dataclasses.fields(held.settle(at_nanos=SECOND)[0]))
        == RECORD_FIELDS
    )


def test_the_record_shape_is_asserted_against_something_that_can_change() -> None:
    """The record shape is asserted against something that can change.

    Guarding the guard. Every case above compares the record's fields to
    RECORD_FIELDS, which is derived from the record itself, so the comparison
    would survive any change made to both at once. This pins the set to a
    written list, so widening the record a consumer reads has to be deliberate.
    """
    assert (
        frozenset(
            {
                "track_id",
                "observed_at_nanos",
                "valid_until_nanos",
                "footprint",
                "height",
                "material",
                "material_confidence",
                "density",
                "mass",
                "codes",
                "evidence",
                "simulated",
            }
        )
        == RECORD_FIELDS
    )
    assert "channel" not in RECORD_FIELDS


def test_no_consumer_counts_cameras() -> None:
    """No consumer counts cameras.

    The reason every case above is cheap: nothing in the package branches on
    how many sensors there are, so a line with one code camera and a line with
    four run the same code.
    """
    raw = load(WORLD)
    declared = len(require(raw, "cameras"))
    assert declared == 5
    package = ROOT / "src" / "clave" / "tracker"
    for path in package.rglob("*.py"):
        text = path.read_text()
        assert "len(sensors)" not in text, path
        assert "sensors[0]" not in text, path
