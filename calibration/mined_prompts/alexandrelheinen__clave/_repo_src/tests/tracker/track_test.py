"""The seam, the belief it fills, and the record a consumer reads.



The seam is the point of this file. `learned-tracker` replaces one
implementation of `Associator` and nothing above it, so what an associator may
read has to be a type rather than a discipline, and the tracker's own tests
construct it with a stub defined here rather than with the shipped one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clave.tracker.association import (
    Association,
    Cue,
    SimulatorIdentity,
    TrackSummary,
)
from clave.tracker.belt_frame import Footprint
from clave.tracker.evidence import (
    Code,
    Detection,
    Evidence,
    GroundTruth,
    Height,
    PixelMask,
    Role,
)
from clave.tracker.fusion import FusionSettings
from clave.tracker.intake import Deployment, Intake
from clave.tracker.sensors import load_sensors
from clave.tracker.track import Tracker, WasteObject
from clave.world.config import load

from .association_contract import check_an_associator

ROOT = Path(__file__).resolve().parents[2]
SECOND = 1_000_000_000
BELT_SPEED = 0.31
WINDOW_EXIT = 1.034


def settings() -> FusionSettings:
    """The shipped fusion settings."""
    return FusionSettings.load(
        ROOT / "configs" / "perception" / "fusion.yml",
        ROOT / "configs" / "world" / "sorting_line.yml",
    )


def a_footprint(x: float = -1.0, y: float = 0.0) -> Footprint:
    """A box resting on the belt."""
    return Footprint(
        center=np.asarray((x, y, 0.93), dtype=np.float64),
        major_extent=0.10,
        minor_extent=0.06,
        yaw=0.0,
    )


class StubAssociator:
    """Joins everything to one track, so a tracker test needs no real rule.

    Nothing above the protocol changes when the implementation does. These
    tests construct the tracker with this rather than
    with `SimulatorIdentity`, which is what makes that checkable by reading one
    import.
    """

    @property
    def name(self) -> str:
        """What decided the association."""
        return "stub"

    def associate(self, cue: Cue, tracks: tuple[TrackSummary, ...]) -> Association:
        """Join the first open track, or open one."""
        del cue
        return Association(track_id=tracks[0].track_id if tracks else None)


def tracker(associator: object | None = None) -> Tracker:
    """A tracker over the shipped line."""
    return Tracker(
        intake=Intake(
            Deployment.SIMULATED,
            load_sensors(load(ROOT / "configs" / "world" / "sorting_line.yml")),
        ),
        associator=associator or StubAssociator(),  # type: ignore[arg-type]
        settings=settings(),
        belt_speed=BELT_SPEED,
        window_exit=WINDOW_EXIT,
    )


def detection(at: int = SECOND, x: float = -1.0, source: str = "gate_wide") -> Evidence:
    """A reading the wide camera could have produced."""
    return Evidence(
        source_id=source,
        role=Role.DETECTION,
        observed_at_nanos=at,
        confidence=0.9,
        payload=Detection(
            footprint=a_footprint(x=x),
            mask=PixelMask(width=64, height=64, runs=((10, 20, 10),)),
            height=0.06,
        ),
    )


def label(at: int = SECOND, object_id: int = 3, class_id: str = "M-06") -> Evidence:
    """What the simulator offers through the sensor path."""
    return Evidence(
        source_id="simulator",
        role=Role.GROUND_TRUTH,
        observed_at_nanos=at,
        confidence=1.0,
        payload=GroundTruth(
            object_id=object_id,
            material_class=class_id,
            position=np.asarray((-1.0, 0.0, 0.93), dtype=np.float64),
        ),
    )


def test_the_shipped_associator_satisfies_the_seam() -> None:
    """The shipped associator satisfies the seam."""
    check_an_associator(SimulatorIdentity())


def test_the_shipped_associator_says_it_reads_the_simulator() -> None:
    """The shipped associator says it reads the simulator.

    Its name goes into a run report, so it has to leave no room for a reader to
    mistake it for perception.
    """
    assert "simulator" in SimulatorIdentity().name


def test_a_cue_carries_only_what_an_associator_may_read() -> None:
    """A cue carries only what an associator may read.

    The seam is a closed type rather than a discipline, so what a learned model
    is allowed to look at is checkable by reading one dataclass.
    """
    import dataclasses

    assert {field.name for field in dataclasses.fields(Cue)} == {
        "observed_at_nanos",
        "footprint",
        "digits",
        "pick_point",
        "height",
        "label",
    }


def test_the_simulator_associator_joins_by_the_label_it_was_given() -> None:
    """This is the identity `learned-tracker` takes away."""
    associator = SimulatorIdentity()
    summary = TrackSummary(
        track_id=7,
        footprint=a_footprint(),
        height=None,
        digits=(),
        label=3,
        last_updated_nanos=SECOND,
    )
    joined = associator.associate(
        Cue(observed_at_nanos=SECOND, footprint=a_footprint(), label=3), (summary,)
    )
    assert joined.track_id == 7

    other = associator.associate(
        Cue(observed_at_nanos=SECOND, footprint=a_footprint(), label=4), (summary,)
    )
    assert other.track_id is None


def test_the_simulator_associator_falls_back_to_geometry_for_pixels() -> None:
    """The simulator associator falls back to geometry for pixels.

    A detection carries no identity, because the adapter discards the only one
    it has. So the shipped associator is asymmetric on
    purpose: ground truth for the label, geometry for the pixels, and it is the
    geometric branch that replaces `association_radius_meters`.
    """
    associator = SimulatorIdentity()
    near = TrackSummary(
        track_id=7,
        footprint=a_footprint(x=-1.00),
        height=None,
        digits=(),
        label=3,
        last_updated_nanos=SECOND,
    )
    joined = associator.associate(
        Cue(observed_at_nanos=SECOND, footprint=a_footprint(x=-1.01)), (near,)
    )
    assert joined.track_id == 7

    far = associator.associate(
        Cue(observed_at_nanos=SECOND, footprint=a_footprint(x=0.9)), (near,)
    )
    assert far.track_id is None


def test_a_tracker_holds_one_track_and_settles_it_into_a_record() -> None:
    """A tracker holds one track and settles it into a record."""
    held = tracker()
    held.observe(detection(), at_nanos=SECOND)
    held.observe(label(), at_nanos=SECOND)
    records = held.settle(at_nanos=SECOND)
    assert len(records) == 1
    assert isinstance(records[0], WasteObject)
    assert records[0].material == "M-06"


def test_the_record_names_no_sensor() -> None:
    """The record names no sensor.

    `evidence` names roles, so a consumer can report that a decision used a
    code without knowing which of three cameras read it.
    """
    import dataclasses

    held = tracker()
    held.observe(detection(source="gate_wide"), at_nanos=SECOND)
    record = held.settle(at_nanos=SECOND)[0]

    rendered = {
        field.name: getattr(record, field.name) for field in dataclasses.fields(record)
    }
    assert "gate_wide" not in repr(rendered)
    assert record.evidence == frozenset({Role.DETECTION})


def test_the_record_carries_no_channel() -> None:
    """The record carries no channel.

    The routing policy owns the channel, so perception could only ever write a
    null into that field. An always-absent field is worse than an absent one.
    """
    import dataclasses

    assert "channel" not in {field.name for field in dataclasses.fields(WasteObject)}


def test_grasp_geometry_is_computed_from_the_footprint() -> None:
    """Grasp geometry is computed from the footprint.

    A stored grasp axis can disagree with the footprint it came from and a
    computed one cannot.
    """
    held = tracker()
    held.observe(detection(), at_nanos=SECOND)
    record = held.settle(at_nanos=SECOND)[0]
    assert record.grasp_width == pytest.approx(record.footprint.minor_extent)
    assert record.grasp_point[0] == pytest.approx(record.footprint.center[0])
    assert record.surface_normal == pytest.approx((0.0, 0.0, 1.0))


def test_the_record_discloses_what_came_from_the_simulator() -> None:
    """The record discloses what came from the simulator.

    No per-instance classifier exists, so a material that arrived through
    `GroundTruth` is the simulator's label rather than a perceived value. A
    record that does not say so launders one into the other, which is the
    failure this spec exists to end.
    """
    held = tracker()
    held.observe(detection(), at_nanos=SECOND)
    held.observe(label(), at_nanos=SECOND)
    record = held.settle(at_nanos=SECOND)[0]
    assert "material" in record.simulated
    assert record.evidence >= frozenset({Role.GROUND_TRUTH})


def test_a_record_built_from_sensors_alone_discloses_nothing_simulated() -> None:
    """The disclosure is earned rather than always present."""
    held = tracker()
    held.observe(detection(), at_nanos=SECOND)
    assert held.settle(at_nanos=SECOND)[0].simulated == frozenset()


def test_a_code_stays_attached_as_the_track_travels() -> None:
    """A code stays attached as the track travels.

    A symbol read at the gate is still attached a metre downstream, which is
    the whole reason a track exists rather than a per-frame record.
    """
    held = tracker()
    held.observe(detection(at=SECOND, x=-1.0), at_nanos=SECOND)
    held.observe(
        Evidence(
            source_id="gate_code_center",
            role=Role.CODE,
            observed_at_nanos=SECOND,
            confidence=0.9,
            payload=Code(symbology="UPC_A", digits="037600138727"),
        ),
        at_nanos=SECOND,
    )
    # Inside the retirement bound, because a track nobody has observed for
    # longer than that is dropped rather than carried forward.
    later = SECOND + SECOND
    record = held.settle(at_nanos=later)[0]
    assert record.codes == ("037600138727",)
    assert record.footprint.center[0] == pytest.approx(-1.0 + BELT_SPEED * 1.0)


def test_a_track_is_propagated_to_the_instant_it_is_settled_at() -> None:
    """Move the clock, never the object."""
    held = tracker()
    held.observe(detection(at=SECOND, x=-1.0), at_nanos=SECOND)
    at_gate = held.settle(at_nanos=SECOND)[0].footprint.center[0]
    downstream = held.settle(at_nanos=SECOND + SECOND)[0].footprint.center[0]
    assert at_gate == pytest.approx(-1.0)
    assert downstream == pytest.approx(-1.0 + BELT_SPEED * 1.0)


def test_valid_until_is_the_measured_window_exit() -> None:
    """The pose expires where the arm can no longer reach it."""
    held = tracker()
    held.observe(detection(at=SECOND, x=0.0), at_nanos=SECOND)
    record = held.settle(at_nanos=SECOND)[0]
    travel = (WINDOW_EXIT - 0.0) / BELT_SPEED
    assert record.valid_until_nanos == pytest.approx(SECOND + travel * SECOND, rel=1e-6)


def test_the_height_in_the_record_comes_from_the_detection_that_estimated_it() -> None:
    """The height in the record comes from the detection that estimated it.

    The line declares no depth sensor, so nothing measures a height and the
    estimate travels with the detection. The contract's change table says
    adding one costs a `Height` adapter and a fusion rule preferring a measured
    height over a class prior, and the rule is already here waiting for it.
    """
    held = tracker()
    held.observe(detection(), at_nanos=SECOND)
    assert held.settle(at_nanos=SECOND)[0].height == pytest.approx(0.06)


def test_a_depth_reading_has_no_sensor_on_this_line_to_have_come_from() -> None:
    """A depth reading has no sensor on this line to have come from.

    No camera produces the depth role, so a `Height` claiming to come from the
    detection camera is refused at the boundary rather than folded. That is the
    intake doing the job it exists for, on the only path into a track.
    """
    from clave.tracker.intake import IntakeError

    held = tracker()
    held.observe(detection(), at_nanos=SECOND)
    with pytest.raises(IntakeError, match="depth"):
        held.observe(
            Evidence(
                source_id="gate_wide",
                role=Role.DEPTH,
                observed_at_nanos=SECOND,
                confidence=0.9,
                payload=Height(top_surface=0.99),
            ),
            at_nanos=SECOND,
        )


def test_nothing_enters_a_track_without_crossing_the_intake() -> None:
    """Nothing enters a track without crossing the intake.

    The property the boundary existed for since task 2, now that there is a
    track for it to be the only path to.
    """
    from clave.tracker.intake import IntakeError

    held = Tracker(
        intake=Intake(
            Deployment.HARDWARE,
            load_sensors(load(ROOT / "configs" / "world" / "sorting_line.yml")),
        ),
        associator=StubAssociator(),
        settings=settings(),
        belt_speed=BELT_SPEED,
        window_exit=WINDOW_EXIT,
    )
    with pytest.raises(IntakeError, match="simulator"):
        held.observe(label(), at_nanos=SECOND)
    assert held.settle(at_nanos=SECOND) == ()


def test_two_objects_produce_two_tracks_with_distinct_identities() -> None:
    """A `track_id` is never reused."""
    held = tracker(associator=SimulatorIdentity())
    held.observe(label(object_id=3, class_id="M-06"), at_nanos=SECOND)
    held.observe(label(object_id=4, class_id="M-09"), at_nanos=SECOND)
    records = held.settle(at_nanos=SECOND)
    assert len({record.track_id for record in records}) == 2


def test_a_detection_joins_the_track_ground_truth_opened() -> None:
    """A detection joins the track ground truth opened.

    The integration defect the unit tests all missed. Ground truth carries a
    position but no extent, and the association gate scales with the track's own
    extent, so a track opened by a label had a gate of nothing and no later
    detection could ever join it. One object ran as two tracks, one carrying the
    material and the other carrying the geometry, and every test here passed
    because each used a stub associator that joins everything.

    It took driving the real associator over a real rollout to see it, which is
    the argument for doing that at all.
    """
    held = Tracker(
        intake=Intake(
            Deployment.SIMULATED,
            load_sensors(load(ROOT / "configs" / "world" / "sorting_line.yml")),
        ),
        associator=SimulatorIdentity(),
        settings=settings(),
        belt_speed=BELT_SPEED,
        window_exit=WINDOW_EXIT,
        unmeasured_extent=0.180,
    )
    held.observe(label(object_id=3, class_id="M-06"), at_nanos=SECOND)
    # A detection lands centimetres from where the label says the object is:
    # the label gives the body origin and the adapter computes a centroid from
    # pixels at the imaged surface. Placing both at one point would make any
    # gate pass and prove nothing.
    held.observe(detection(at=SECOND, x=-0.97), at_nanos=SECOND)

    records = held.settle(at_nanos=SECOND)
    assert len(records) == 1, "one object is one track, not two"
    assert records[0].evidence == frozenset({Role.GROUND_TRUTH, Role.DETECTION})
    assert records[0].material == "M-06"


def test_ground_truth_joins_a_detection_track_opened_first() -> None:
    """Ground truth labels a detection track when it arrives second."""
    held = Tracker(
        intake=Intake(
            Deployment.SIMULATED,
            load_sensors(load(ROOT / "configs" / "world" / "sorting_line.yml")),
        ),
        associator=SimulatorIdentity(),
        settings=settings(),
        belt_speed=BELT_SPEED,
        window_exit=WINDOW_EXIT,
        unmeasured_extent=0.180,
    )
    held.observe(detection(at=SECOND, x=-0.97), at_nanos=SECOND)
    held.observe(label(object_id=3, class_id="M-06"), at_nanos=SECOND)

    records = held.settle(at_nanos=SECOND)
    assert len(records) == 1
    assert records[0].material == "M-06"
    assert records[0].evidence == frozenset({Role.GROUND_TRUTH, Role.DETECTION})
    assert records[0].height is not None, "the geometry reached the same track"


def test_without_an_unmeasured_extent_the_two_would_not_have_met() -> None:
    """Without an unmeasured extent the two would not have met.

    Guarding the fix rather than the symptom. At zero the gate is nothing, which
    is exactly the state the demo exposed, so this pins why the parameter
    exists.
    """
    held = Tracker(
        intake=Intake(
            Deployment.SIMULATED,
            load_sensors(load(ROOT / "configs" / "world" / "sorting_line.yml")),
        ),
        associator=SimulatorIdentity(),
        settings=settings(),
        belt_speed=BELT_SPEED,
        window_exit=WINDOW_EXIT,
        unmeasured_extent=0.0,
    )
    held.observe(label(object_id=3, class_id="M-06"), at_nanos=SECOND)
    held.observe(detection(at=SECOND, x=-0.97), at_nanos=SECOND)
    assert len(held.settle(at_nanos=SECOND)) == 2


def test_a_track_nobody_has_observed_is_retired() -> None:
    """AC-TRACK-36: a track nobody has observed is retired.

    Carried forward instead, it is dead reckoned indefinitely. Measured
    before this existed, records past the arm's reach had a median error of
    ten metres and outnumbered the ones inside the sensing gate nineteen to
    one, and the arm chose its next object from among them.
    """
    held = tracker()
    held.observe(detection(at=SECOND, x=-1.0), at_nanos=SECOND)
    bound = held.settings.retire_after
    assert held.settle(at_nanos=SECOND + int(bound * 0.9 * SECOND))
    assert held.retired == 0
    assert held.settle(at_nanos=SECOND + int(bound * 1.1 * SECOND)) == ()
    assert held.retired == 1


def test_a_track_still_being_observed_is_not_retired() -> None:
    """AC-TRACK-36: a track still being observed is not retired.

    The non-vacuous half. A bound that dropped everything would pass the
    test above and describe a tracker that tracks nothing.
    """
    held = tracker()
    at = SECOND
    for _ in range(12):
        held.observe(detection(at=at, x=-1.0), at_nanos=at)
        assert held.settle(at_nanos=at), "retired while still being seen"
        at += SECOND // 2
    assert held.retired == 0
