"""Ordering the markers into a queue.

Etapa A holds the metric and the head. Anchors, the hysteresis that damps
them, and recomputing on change arrive with etapa B, so nothing here asserts
stability yet.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from clave.control.selection import (
    PickabilityRule,
    Selector,
    all_pickability_rules,
    pickable_in_belt,
)
from clave.control.settings import ControlSettings, SelectionSettings
from clave.tracker.markers import GraspMarker, color_for
from clave.world.config import load

ROOT = Path(__file__).resolve().parents[2]
NANOS_PER_SECOND = 1_000_000_000
BELT_SPEED = 0.30


def everywhere(_: NDArray[np.float64]) -> bool:
    """An arm trusted over the whole world, for tests about ordering."""
    return True


def selector(
    exit_weight: float = 1.0,
    admits: Callable[[NDArray[np.float64]], bool] | None = None,
    pickability: PickabilityRule | None = None,
) -> Selector:
    """A selector with the weight under test."""
    return Selector(
        settings(exit_weight),
        admits or everywhere,
        pickability=pickability,
    )


def settings(exit_weight: float = 1.0) -> SelectionSettings:
    """Selection settings with the weight under test."""
    shipped = ControlSettings.load(load(ROOT / "configs" / "runtime" / "control.yml"))
    return SelectionSettings(
        exit_weight=exit_weight, anchor_radius=shipped.selection.anchor_radius
    )


def marker(
    track_id: int,
    x: float,
    y: float = 0.0,
    leaves_in_seconds: float = 5.0,
    oriented: bool = True,
    reachable: bool = True,
) -> GraspMarker:
    """One marker, with only the fields selection reads set."""
    return GraspMarker(
        track_id=track_id,
        valid_until_nanos=int(leaves_in_seconds * NANOS_PER_SECOND),
        grasp=np.asarray((x, y, 0.945), dtype=np.float64),
        flange=np.asarray((x, y, 1.035), dtype=np.float64),
        pads=(
            (
                np.asarray((x, y - 0.03, 0.945), dtype=np.float64),
                np.asarray((x, y + 0.03, 0.945), dtype=np.float64),
            )
            if oriented
            else ()
        ),
        pad_size=np.asarray((0.006, 0.02, 0.025), dtype=np.float64),
        closing_axis=1.5708 if oriented else None,
        opening=0.06,
        oriented=oriented,
        reachable=reachable,
        extent=0.12,
        color=color_for(track_id),
    )


def test_an_empty_belt_produces_an_empty_queue() -> None:
    """An empty belt produces an empty queue."""
    queue = selector().update(
        (),
        np.asarray((0.0, 0.0, 1.2), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert queue.order == ()
    assert queue.head is None


def test_the_nearer_of_two_equally_urgent_markers_comes_first() -> None:
    """AC-MOVE-01: the nearer of two equally urgent markers comes first."""
    near = marker(1, x=0.10, leaves_in_seconds=5.0)
    far = marker(2, x=0.80, leaves_in_seconds=5.0)
    queue = selector().update(
        (far, near),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert queue.head is not None
    assert queue.head.track_id == 1


def test_the_more_urgent_of_two_equally_near_markers_comes_first() -> None:
    """AC-MOVE-01: the more urgent of two equally near markers comes first.

    Equally near in the sense that matters: both the same distance from the
    flange, one about to run out of belt.
    """
    leaving = marker(1, x=0.30, y=0.20, leaves_in_seconds=0.5)
    staying = marker(2, x=0.30, y=-0.20, leaves_in_seconds=6.0)
    queue = selector().update(
        (staying, leaving),
        np.asarray((0.30, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert queue.head is not None
    assert queue.head.track_id == 1


def test_urgency_can_outrank_distance_and_the_weight_says_by_how_much() -> None:
    """AC-MOVE-01: urgency can outrank distance, and the weight says by how much.

    The same pair orders one way at a weight that ignores the deadline and the
    other way at a weight that respects it, which is what makes the weight the
    knob rather than a decoration.
    """
    urgent_and_far = marker(1, x=0.90, leaves_in_seconds=0.2)
    idle_and_near = marker(2, x=0.10, leaves_in_seconds=8.0)
    pair = (urgent_and_far, idle_and_near)
    flange = np.asarray((0.0, 0.0, 1.035), dtype=np.float64)

    ignoring = selector(exit_weight=0.0).update(pair, flange, BELT_SPEED, 0)
    assert ignoring.head is not None
    assert ignoring.head.track_id == 2

    respecting = selector(exit_weight=2.0).update(pair, flange, BELT_SPEED, 0)
    assert respecting.head is not None
    assert respecting.head.track_id == 1


def test_the_order_is_built_from_where_the_flange_will_be() -> None:
    """AC-MOVE-02: the order is built from where the flange will be.

    Scoring every candidate from where the flange stands now would put the
    two near the origin first and the far one last. Scoring each next
    candidate from the previous one walks the chain instead, which is what
    nearest-neighbour construction means.
    """
    chain = (
        marker(1, x=0.20, leaves_in_seconds=9.0),
        marker(3, x=1.00, leaves_in_seconds=9.0),
        marker(2, x=0.60, leaves_in_seconds=9.0),
    )
    queue = selector(exit_weight=0.0).update(
        chain,
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert tuple(item.track_id for item in queue.order) == (1, 2, 3)


def test_a_marker_the_effector_cannot_open_to_is_left_out() -> None:
    """A marker the effector cannot open to is left out.

    An unreachable pose is not a candidate. Ordering it and then refusing it
    at the servo wastes a visit the arm could have spent on something it can
    take.
    """
    wide = marker(1, x=0.10, reachable=False)
    queue = selector().update(
        (wide,),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert queue.order == ()


def test_a_marker_outside_the_belt_is_left_out() -> None:
    """AC-MOVE-45: an object outside the belt footprint never enters the pick queue."""
    outside_x = marker(1, x=1.51)
    outside_y = marker(2, x=0.20, y=0.26)
    instance = selector(pickability=pickable_in_belt(length=3.0, width=0.5))
    queue = instance.update(
        (outside_x, outside_y),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert queue.order == ()


def test_pickability_rules_can_be_composed() -> None:
    """AC-MOVE-45: additional pick conditions can be added without changing Selector."""

    def only_track_two(candidate: GraspMarker) -> bool:
        return candidate.track_id == 2

    rules = all_pickability_rules(pickable_in_belt(3.0, 0.5), only_track_two)
    assert rules(marker(2, x=0.20))
    assert not rules(marker(1, x=0.20))
    assert not rules(marker(2, x=1.51))


def test_pickability_rules_compose_inside_selector() -> None:
    """AC-MOVE-45: custom pickability rules compose with belt bounds in the queue."""

    def only_even_tracks(candidate: GraspMarker) -> bool:
        return candidate.track_id % 2 == 0

    composed = all_pickability_rules(pickable_in_belt(3.0, 0.5), only_even_tracks)
    instance = selector(pickability=composed)
    valid_even = marker(2, x=0.20)
    valid_odd = marker(1, x=0.20)
    outside_even = marker(4, x=1.60)

    queue = instance.update(
        (valid_even, valid_odd, outside_even),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert [c.track_id for c in queue.order] == [2]


def test_a_marker_already_past_the_window_is_left_out() -> None:
    """A marker already past the window is left out.

    Its pose expired, so there is no belt left to reach it over and no
    ordering that helps.
    """
    gone = marker(1, x=0.10, leaves_in_seconds=1.0)
    queue = selector().update(
        (gone,),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        at_nanos=2 * NANOS_PER_SECOND,
    )
    assert queue.order == ()


def test_a_candidate_carries_what_the_task_layer_needs() -> None:
    """A candidate carries what the task layer needs.

    The flange pose and the closing axis travel with the candidate, so the
    task machine never has to go back to the marker list to find them.
    """
    queue = selector().update(
        (marker(4, x=0.25, y=0.10),),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    head = queue.head
    assert head is not None
    assert head.track_id == 4
    assert head.flange == pytest.approx(
        np.asarray((0.25, 0.10, 1.035), dtype=np.float64),
    )
    assert head.closing_axis == pytest.approx(1.5708)
    assert head.distance_before_leaving == pytest.approx(5.0 * BELT_SPEED)


def test_an_unoriented_marker_carries_no_closing_axis() -> None:
    """AC-MOVE-08: an unoriented marker carries no closing axis."""
    queue = selector().update(
        (marker(1, x=0.25, oriented=False),),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert queue.head is not None
    assert queue.head.closing_axis is None


def test_a_stopped_belt_leaves_nothing_urgent() -> None:
    """A stopped belt leaves nothing urgent.

    With no belt speed no object is running out of belt, so the deadline term
    vanishes and the ordering is by travel alone. This is the degenerate case
    the metric has to survive rather than divide by.
    """
    pair = (
        marker(1, x=0.90, leaves_in_seconds=0.2),
        marker(2, x=0.10, leaves_in_seconds=8.0),
    )
    queue = selector().update(
        pair,
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        0.0,
        0,
    )
    assert queue.head is not None
    assert queue.head.track_id == 2


def test_a_pose_the_arm_is_not_trusted_over_is_left_out() -> None:
    """AC-MOVE-01: a pose the arm is not trusted over is left out.

    Different from a jaw that does not open wide enough, and conflating the
    two fed the controller markers a metre outside the annulus and faulted
    every visit. The region comes from the arm rather than from a constant
    here, so the two cannot drift.
    """
    upstream = marker(1, x=-1.00)
    near = marker(2, x=0.30)

    def inside_the_annulus(pose: NDArray[np.float64]) -> bool:
        return bool(abs(float(pose[0])) <= 0.80)

    queue = selector(admits=inside_the_annulus).update(
        (upstream, near), np.asarray((0.0, 0.0, 1.035), dtype=np.float64), BELT_SPEED, 0
    )
    assert tuple(item.track_id for item in queue.order) == (2,)


def travelled(base: GraspMarker, seconds: float, speed: float) -> GraspMarker:
    """The same marker, carried along the belt by the time given.

    Only `x` moves, which is what `clave.tracker.belt_frame` says the belt
    does, and the window closes by the same amount.
    """
    import dataclasses

    shift = speed * seconds
    return dataclasses.replace(
        base,
        pinch_position_belt=np.asarray(
            (
                base.pinch_position_belt[0] + shift,
                base.pinch_position_belt[1],
                base.pinch_position_belt[2],
            ),
            dtype=np.float64,
        ),
        flange_position_world=np.asarray(
            (
                base.flange_position_world[0] + shift,
                base.flange_position_world[1],
                base.flange_position_world[2],
            ),
            dtype=np.float64,
        ),
    )


def nudged(base: GraspMarker, by: float) -> GraspMarker:
    """The same marker, displaced across the belt by an estimation error."""
    import dataclasses

    return dataclasses.replace(
        base,
        pinch_position_belt=np.asarray(
            (
                base.pinch_position_belt[0],
                base.pinch_position_belt[1] + by,
                base.pinch_position_belt[2],
            ),
            dtype=np.float64,
        ),
        flange_position_world=np.asarray(
            (
                base.flange_position_world[0],
                base.flange_position_world[1] + by,
                base.flange_position_world[2],
            ),
            dtype=np.float64,
        ),
    )


def test_the_first_call_builds_an_order() -> None:
    """AC-MOVE-17: the first call builds an order."""
    queue = selector().update(
        (marker(1, x=0.20), marker(2, x=0.60)),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert queue.recomputed is True
    assert len(queue.order) == 2


def test_belt_travel_alone_does_not_reorder_the_queue() -> None:
    """AC-MOVE-17: belt travel alone does not reorder the queue.

    Every candidate loses belt at the same rate, so travel subtracts a common
    term from every deadline and changes no ordering. Anchors are compared
    after being carried along the belt for exactly this reason: an object
    doing what the belt makes it do has not moved in the frame that matters.
    """
    pair = (
        marker(1, x=0.20, leaves_in_seconds=9.0),
        marker(2, x=0.60, leaves_in_seconds=9.0),
    )
    one = selector()
    first = one.update(
        pair,
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert first.recomputed is True

    for tick in range(1, 12):
        seconds = tick * 0.5
        carried = tuple(travelled(item, seconds, BELT_SPEED) for item in pair)
        later = one.update(
            carried,
            np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
            BELT_SPEED,
            at_nanos=int(seconds * NANOS_PER_SECOND),
        )
        assert later.recomputed is False, f"reordered on travel alone at {seconds} s"
        assert tuple(item.track_id for item in later.order) == (1, 2)


def test_an_estimate_jittering_under_the_radius_does_not_reorder() -> None:
    """AC-MOVE-16: an estimate jittering under the radius does not reorder."""
    pair = (marker(1, x=0.20), marker(2, x=0.60))
    one = selector()
    one.update(
        pair,
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )

    radius = settings().anchor_radius
    for tick, sign in enumerate((1, -1, 1, -1), start=1):
        jittered = (nudged(pair[0], sign * radius * 0.4), pair[1])
        carried = tuple(travelled(item, tick * 0.1, BELT_SPEED) for item in jittered)
        later = one.update(
            carried,
            np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
            BELT_SPEED,
            at_nanos=int(tick * 0.1 * NANOS_PER_SECOND),
        )
        assert later.recomputed is False


def test_an_estimate_moving_past_the_radius_reorders() -> None:
    """AC-MOVE-16: an estimate moving past the radius reorders."""
    pair = (marker(1, x=0.20), marker(2, x=0.60))
    one = selector()
    one.update(
        pair,
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )

    moved = (nudged(pair[0], settings().anchor_radius * 3.0), pair[1])
    later = one.update(
        moved,
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert later.recomputed is True


def test_a_track_appearing_reorders() -> None:
    """AC-MOVE-17: a track appearing reorders."""
    one = selector()
    one.update(
        (marker(1, x=0.20),),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    later = one.update(
        (marker(1, x=0.20), marker(2, x=0.10)),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert later.recomputed is True
    assert tuple(item.track_id for item in later.order) == (2, 1)


def test_a_track_retiring_reorders() -> None:
    """AC-MOVE-17: a track retiring reorders."""
    one = selector()
    pair = (marker(1, x=0.20), marker(2, x=0.60))
    one.update(
        pair,
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    later = one.update(
        (pair[1],),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert later.recomputed is True
    assert tuple(item.track_id for item in later.order) == (2,)


def test_the_arm_moving_does_not_reorder_the_queue() -> None:
    """AC-MOVE-17: the arm moving does not reorder the queue.

    The flange is an input to the cost, so re-sorting every tick would let
    the arm's own travel swap its target underneath it. Recomputing on change
    means the flange is read once, when something actually happened.
    """
    one = selector()
    pair = (marker(1, x=0.20), marker(2, x=0.60))
    one.update(
        pair,
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    later = one.update(
        pair,
        np.asarray((0.55, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert later.recomputed is False
    assert tuple(item.track_id for item in later.order) == (1, 2)


def test_a_candidate_carries_the_live_pose_and_the_anchor_apart() -> None:
    """AC-MOVE-16: a candidate carries the live pose and the anchor apart.

    The anchor is quantised so the ordering holds still. The pose handed to
    the task layer is not, because an arm commanded to a quantised pose jumps
    by the radius every time the anchor catches up.
    """
    one = selector()
    base = marker(1, x=0.20)
    one.update(
        (base,),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )

    radius = settings().anchor_radius
    later = one.update(
        (nudged(base, radius * 0.4),),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    head = later.order[0]
    assert head.flange[1] == pytest.approx(radius * 0.4)
    assert head.anchor[1] == pytest.approx(0.0)


def test_an_anchor_is_forgotten_when_its_track_goes() -> None:
    """An anchor is forgotten when its track goes.

    A track id the world reuses would otherwise be scored where a different
    object stood, and the selector would grow for the length of a run.
    """
    one = selector()
    one.update(
        (marker(1, x=0.20),),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    one.update(
        (),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert one.anchor_count == 0


def test_a_rebuild_behind_the_head_leaves_the_head_alone() -> None:
    """AC-MOVE-17: a rebuild behind the head leaves the head alone.

    Rebuilding is cheap; swapping the head mid-traverse is not. A track at
    the back of the queue moving past its radius has to rebuild the order,
    and it must not take the arm off a nearer, more urgent target that has
    not moved.
    """
    one = selector()
    head = marker(1, x=0.10, leaves_in_seconds=1.0)
    tail = marker(2, x=0.90, leaves_in_seconds=9.0)
    first = one.update(
        (head, tail),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert first.head is not None
    assert first.head.track_id == 1

    shifted = nudged(tail, settings().anchor_radius * 4.0)
    later = one.update(
        (head, shifted),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert later.reasons == frozenset({"anchor"})
    assert later.head is not None
    assert later.head.track_id == 1


def test_the_reasons_say_which_trigger_fired() -> None:
    """AC-MOVE-17: the reasons say which trigger fired.

    Counting rebuilds together hides whether the anchors damp anything: a
    rebuild because a track appeared is the design working, and one because
    an anchor moved is the estimate having genuinely shifted.
    """
    one = selector()
    first = one.update(
        (marker(1, x=0.20),),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert first.reasons == frozenset({"appeared"})

    gone = one.update(
        (),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        0,
    )
    assert gone.reasons == frozenset({"retired"})


def test_selector_carries_candidate_with_instantaneous_velocity() -> None:
    """Selector carries candidate with its own instantaneous velocity when present."""
    base = marker(1, x=0.20)
    slow = GraspMarker(
        track_id=base.track_id,
        valid_until_nanos=base.valid_until_nanos,
        grasp=base.grasp,
        flange=base.flange,
        pads=base.pads,
        pad_size=base.pad_size,
        closing_axis=base.closing_axis,
        opening=base.opening,
        oriented=base.oriented,
        reachable=base.reachable,
        extent=base.extent,
        color=base.color,
        channel=base.channel,
        velocity_world=np.asarray((0.10, 0.0, 0.0), dtype=np.float64),
    )
    one = selector()
    first = one.update(
        (slow,),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        at_nanos=0,
    )
    assert first.head is not None
    assert first.head.flange_position_world[0] == pytest.approx(0.20)

    # If the object physically moves at 0.10 m/s (from 0.20 to 0.30 m along x in 1s):
    base_advanced = marker(1, x=0.30)
    import dataclasses

    moved_slow = dataclasses.replace(
        base_advanced,
        velocity_world=np.asarray((0.10, 0.0, 0.0), dtype=np.float64),
    )
    later = one.update(
        (moved_slow,),
        np.asarray((0.0, 0.0, 1.035), dtype=np.float64),
        BELT_SPEED,
        at_nanos=1_000_000_000,
    )
    # The anchor correctly followed the 0.10 m/s velocity, so no reanchor occurred!
    assert "anchor" not in later.reasons
    assert later.head is not None
    assert later.head.anchor_position_belt[0] == pytest.approx(0.30)
