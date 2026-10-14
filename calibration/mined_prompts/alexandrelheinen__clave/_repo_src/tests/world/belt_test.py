"""Tests for the conveyor drive and reachability."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from clave.world import arm as armmod
from clave.world import belt, config, scene
from clave.world.scene import BeltGeometry, SceneLayout, TakeawayDrive

ROOT = Path(__file__).resolve().parents[2]
_ARM = config.load(ROOT / "configs" / "world" / "sorting_line.yml")["arm"]
_REACH_MIN = float(_ARM["reach_min_meters"])
_REACH_MAX = float(_ARM["reach_max_meters"])
_TOOL_ABOVE = (
    float(_ARM["tool_above_base_meters"][0]),
    float(_ARM["tool_above_base_meters"][1]),
)
CONFIG = ROOT / "configs" / "world" / "sorting_line.yml"


def layout(speed: float, offset: float) -> SceneLayout:
    """A minimal layout for reachability arithmetic, needing no physics.

    The reach bounds are the arm's own and are not parameters here: they come
    from the model rather than from a number a test picks, which is what keeps
    this arithmetic describing the manipulator the world actually has.
    """

    return SceneLayout(
        belt=BeltGeometry(
            length=3.0,
            width=1.0,
            surface_height=0.90,
            speed=speed,
            height_tolerance=0.05,
        ),
        arm_base=np.asarray((0.0, offset, 0.90), dtype=np.float64),
        reach_min=_REACH_MIN,
        reach_max=_REACH_MAX,
        tool_above_base=_TOOL_ABOVE,
        pedestal=(0.30, 0.30),
        channels=("CH-PET",),
        objects=(),
        pool_size=0,
        timestep=0.002,
        takeaway=TakeawayDrive(
            speed=0.20,
            height_min=0.25,
            height_max=0.45,
            mouth_half_width=0.15,
            past_mouth=1.50,
            toward_mouth=0.10,
        ),
    )


def test_the_report_states_the_annulus_window_and_budget() -> None:
    """The report states the annulus window and budget."""
    report = belt.reach_report(layout(speed=0.2, offset=-0.34))
    assert report.reach_min == _REACH_MIN
    assert report.reach_max == _REACH_MAX
    assert report.belt_offset == pytest.approx(0.34)
    assert report.window_length > 0
    assert report.time_budget == pytest.approx(report.window_length / 0.2)
    assert report.reachable


def test_a_belt_outside_reach_yields_no_window() -> None:
    """A world where the arm cannot touch the belt must say so.

    The offset has to clear the outer radius now. A UR10e reaches 1.25 m, so
    the 0.9 m that stranded the previous arm leaves this one comfortably on
    the belt.
    """
    report = belt.reach_report(layout(speed=0.2, offset=-(_REACH_MAX + 0.5)))
    assert report.window_length == 0.0
    assert not report.reachable


def test_a_faster_belt_shortens_the_budget() -> None:
    """The budget is what every perception latency fits inside."""
    slow = belt.reach_report(layout(speed=0.1, offset=-0.34))
    fast = belt.reach_report(layout(speed=0.4, offset=-0.34))
    assert fast.time_budget < slow.time_budget


def built(
    seed: int,
) -> tuple[dict[str, Any], np.random.Generator, Any, Any, SceneLayout]:
    """Build the shipped world at one seed, with a conveyor ready to drive.

    Args:
        seed: The seed.

    Returns:
        The configuration, the generator, the model, its state and the plan.
    """
    raw = config.load(CONFIG)
    rng = np.random.default_rng(seed)
    model, data, plan = scene.build(raw, rng, ROOT)
    return raw, rng, model, data, plan


def conveyor_for(
    raw: dict[str, Any], rng: np.random.Generator, plan: SceneLayout
) -> belt.Conveyor:
    """A conveyor built from the shipped spawn configuration.

    Args:
        raw: The configuration.
        rng: The generator.
        plan: The resolved layout.

    Returns:
        The conveyor.
    """
    spawn = raw["spawn"]
    return belt.Conveyor(
        plan,
        rng,
        config.require_range(spawn, "spacing_meters", "spawn"),
        config.require_range(spawn, "lateral_offset_meters", "spawn"),
        config.require_range(spawn, "drop_height_meters", "spawn"),
        entry_margin=float(spawn["entry_margin_meters"]),
    )


def test_objects_reach_belt_speed_and_enter_the_window() -> None:
    """Objects reach belt speed and enter the window."""
    pytest.importorskip("mujoco")
    import mujoco

    raw = config.load(CONFIG)
    rng = np.random.default_rng(7)
    model, data, plan = scene.build(raw, rng, ROOT)
    spawn = raw["spawn"]
    conveyor = belt.Conveyor(
        plan,
        rng,
        config.require_range(spawn, "spacing_meters", "spawn"),
        config.require_range(spawn, "lateral_offset_meters", "spawn"),
        config.require_range(spawn, "drop_height_meters", "spawn"),
        entry_margin=float(spawn["entry_margin_meters"]),
    )
    for _ in range(12000):
        mujoco.mj_step(model, data)
        conveyor.step(model, data)
    assert conveyor.active, "nothing spawned"
    assert conveyor.entered_window(), "no object reached the arm"
    for item in conveyor.active:
        assert item.material_class.startswith("M-")
        assert item.channel.startswith("CH-")


def test_a_lifted_object_is_not_driven_by_the_belt() -> None:
    """A grasped object keeps its motion after it leaves the belt."""
    plan = layout(speed=0.3, offset=-0.34)
    assert belt._on_belt((0.0, 0.0, 0.90), plan)
    assert not belt._on_belt((0.0, 0.0, 0.951), plan)


def test_an_unpicked_object_is_not_removed() -> None:
    """A real line does not stop for a missed pick.

    The object leaves the belt when the belt carries it off the end, and for
    no other reason. Counting what is on the belt is not enough to say that:
    the feed doses by distance, so an object can reach the end and retire in
    the same window in which nothing new is due, and the count dips for a
    reason that has nothing to do with the line removing anything. This reads
    where each object was when it went instead.
    """
    pytest.importorskip("mujoco")
    import mujoco

    raw = config.load(CONFIG)
    rng = np.random.default_rng(2)
    model, data, plan = scene.build(raw, rng, ROOT)
    spawn = raw["spawn"]
    conveyor = belt.Conveyor(
        plan,
        rng,
        config.require_range(spawn, "spacing_meters", "spawn"),
        config.require_range(spawn, "lateral_offset_meters", "spawn"),
        config.require_range(spawn, "drop_height_meters", "spawn"),
        entry_margin=float(spawn["entry_margin_meters"]),
    )
    seen: dict[str, float] = {}
    left_from: dict[str, float] = {}
    end = plan.belt.length / 2.0
    for _ in range(13000):
        mujoco.mj_step(model, data)
        conveyor.step(model, data)
        riding = {item.index for item in conveyor.active}
        assert not (riding & set(conveyor._free)), (
            "a pool slot with an object on the belt was handed back as free"
        )
        for item in conveyor.active:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.name)
            address = model.jnt_qposadr[model.body_jntadr[body]]
            seen[item.name] = float(data.qpos[address])
        for gone in set(seen) - {item.name for item in conveyor.active}:
            left_from[gone] = seen[gone]
    assert left_from, "no object ever left the belt, so nothing was tested"
    for name, where in left_from.items():
        assert abs(where) > end - 0.05, (
            f"{name} left the belt from x={where:.3f}, in the middle of it"
        )


def test_the_window_covers_both_halves_of_the_belt() -> None:
    """The belt is centered on the origin, so the sweep must be.

    A sweep that starts at the origin sees only the downstream half, which the
    expert then halves again to obtain the edge an object leaves reach at. The
    two ends are found here by asking the same test the safety layer asks,
    rather than by a radius this test computes for itself.
    """
    plan = layout(speed=0.2, offset=-0.70)
    step = 0.002
    reachable = [
        x * step
        for x in range(
            int(-plan.belt.length / 2.0 / step), int(plan.belt.length / 2.0 / step) + 1
        )
        if armmod.reaches(
            (plan.arm_base[0], plan.arm_base[1]),
            x * step,
            0.0,
            armmod.ReachBounds(_REACH_MIN, _REACH_MAX, _TOOL_ABOVE),
        )
    ]
    assert min(reachable) < 0.0, "nothing upstream of the arm is reachable"
    span = max(reachable) - min(reachable)
    report = belt.reach_report(plan)
    assert report.window_length == pytest.approx(span, abs=2 * step)
    # The edges a figure draws are the ones the sweep found.
    assert report.window_edges[0] == pytest.approx(min(reachable), abs=2 * step)
    assert report.window_edges[1] == pytest.approx(max(reachable), abs=2 * step)


def test_the_feed_is_by_distance_rather_than_by_elapsed_time() -> None:
    """AC-RATE-01: the feed is by distance rather than by elapsed time.

    The property the whole loop rests on. Halving the belt speed has to
    halve the number of objects released in a given span of time; a feed by
    time would release the same number and only spread them further apart.
    """
    pytest.importorskip("mujoco")
    import mujoco

    released = {}
    for speed in (0.30, 0.15):
        raw, rng, model, data, plan = built(0)
        conveyor = conveyor_for(raw, rng, plan)
        conveyor.speed = speed
        for _ in range(int(40.0 / plan.timestep)):
            mujoco.mj_step(model, data)
            conveyor.step(model, data)
        released[speed] = len(conveyor.arrivals)
    assert released[0.30] >= 6, "too few objects for the ratio to mean anything"
    assert released[0.15] == pytest.approx(released[0.30] / 2, abs=1)


def test_a_recycled_slot_is_a_new_identity_ac_gt_06() -> None:
    """AC-GT-06: a respawned object is a new identity, not the slot it reuses.

    The regression this exists for: ground-truth markers were identified by
    pool slot, and a slot is reused as soon as the object in it leaves the
    belt. The task machine remembers which identities it has already served,
    so on the shipped line it served each of the pool's slots once and then
    sat idle for the rest of the run -- four visits in sixty seconds, with the
    remaining forty seconds spent parked while objects went past.
    """
    pytest.importorskip("mujoco")
    import mujoco

    raw = config.load(CONFIG)
    raw["spawn"]["pool_size"] = 2
    rng = np.random.default_rng(0)
    model, data, plan = scene.build(raw, rng, ROOT)
    conveyor = conveyor_for(raw, rng, plan)
    conveyor.speed = 0.35
    seen: dict[int, set[int]] = {}
    for _ in range(int(40.0 / plan.timestep)):
        mujoco.mj_step(model, data)
        conveyor.step(model, data)
        for item in conveyor.active:
            seen.setdefault(item.index, set()).add(item.serial)
    assert len(conveyor.arrivals) > plan.pool_size, "the pool was never spent"
    reused = {slot: serials for slot, serials in seen.items() if len(serials) > 1}
    assert reused, "no slot was ever reused, so nothing was tested"
    for slot, serials in reused.items():
        assert len(serials) > 1, f"slot {slot} reported one identity twice"


def test_an_object_on_the_belt_keeps_turning_ac_rate_11() -> None:
    """AC-RATE-11: the belt carries an object without pinning its rotation.

    The spin about the belt normal is left to physics and damped rather than
    pinned, and this exists because both halves were measured. Objects here
    turn at up to 5.4 radians per second on their way down the belt, which
    makes a grasp yaw claimed half a second earlier wrong by up to 40 degrees
    at the instant the jaws close -- and pinning is worse than the problem:
    with the spin set to zero every tick the contact becomes a reaction that
    tips the parcel over, and over the same 60 second run the pad-to-belt
    contact moved from −0.6 mm over 4 ticks to −2.8 mm over 54, the worst
    vertical lurch from 99 to 934 m/s², and the grasps held from one of seven
    to none. So the rotation is damped over a configured time constant, and a
    test that found it *stopped* dead within half a second would be describing
    the thing that was removed.
    """
    pytest.importorskip("mujoco")
    import mujoco

    raw, rng, model, data, plan = built(0)
    conveyor = conveyor_for(raw, rng, plan)
    conveyor.speed = 0.35
    for _ in range(int(12.0 / plan.timestep)):
        mujoco.mj_step(model, data)
        conveyor.step(model, data)
    assert conveyor.active, "nothing was on the belt to test"
    item = conveyor.active[0]
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.name)
    dof = model.jnt_dofadr[model.body_jntadr[body]]
    data.qvel[dof + 5] = 5.0
    for _ in range(int(0.5 / plan.timestep)):
        mujoco.mj_step(model, data)
        conveyor.step(model, data)
    spin = abs(float(data.qvel[dof + 5]))
    assert 0.05 < spin < 5.0, (
        f"the parcel was turning at {spin:.2f} rad/s half a second after 5.0: "
        f"zero means the belt is pinning it, five means nothing damps it"
    )


def test_a_slot_returns_to_the_pool_once_its_object_leaves() -> None:
    """AC-RATE-01: a slot returns to the pool once its object leaves.

    A compiled model cannot gain bodies at run time, so a line that never
    gives a slot back stops feeding after `pool_size` objects. A rate held
    for ninety seconds is a different claim from a rate held.
    """
    pytest.importorskip("mujoco")
    import mujoco

    raw, rng, model, data, plan = built(0)
    conveyor = conveyor_for(raw, rng, plan)
    conveyor.speed = 0.35
    for _ in range(int(240.0 / plan.timestep)):
        mujoco.mj_step(model, data)
        conveyor.step(model, data)
    assert conveyor.retired > 0, "nothing ever left the belt"
    assert len(conveyor.arrivals) > plan.pool_size, (
        "the line stopped feeding once the pool was spent"
    )
