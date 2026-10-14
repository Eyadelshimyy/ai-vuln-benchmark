"""The conveyor: spawning, carrying, and reachability.

MuJoCo has no conveyor primitive. Objects are free bodies whose horizontal
velocity is driven to the belt speed while they rest on the belt, leaving
vertical motion, rotation and contact to physics. This is a driven constraint
rather than a friction model, which is the honest description: objects do not
slip, and a real line's slip is a disturbance this world does not reproduce.

**The rotation is left to physics, and that was measured four ways.** With the
belt driving the centre velocity and the belt's friction acting under it, a
parcel turns at up to 5.4 radians per second on its way down the belt, which is
155 degrees over the half second between the capture that claims a grasp yaw and
the instant the jaws close. Every repair on this side of the seam was tried and
measured against the same 60 second run, and every one of them cost more than it
bought:

| Repair | Pad-to-belt contact | Worst vertical lurch | Grasps held |
| --- | --- | --- | --- |
| none, as shipped | −0.6 mm over 4 ticks | 99 m/s² | 1 of 7 |
| the spin pinned to zero each tick | **−2.8 mm, 54 ticks** | **934 m/s²** | none of 7 |
| the spin damped, 0.3 s constant | **−18.6 mm, 17 ticks** | 170 m/s² | none of 8 |

Pinning turns the belt's drag into a reaction that tips the parcel over.
Damping leaves the parcels lying flat, which puts more of them under a jaw that
then closes on a flat parcel's upper edge and is pulled down onto the belt. So
the rotation stays where it is and the yaw is carried forward on the control
side instead, which is written up in
[measurements.md](measurements.md#what-the-reported-run-turned-out-to-be).

An object that passes the reachable window is left alone. A real line does not
stop, so an unpicked object is a throughput loss rather than a fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clave.world import arm
from clave.world.config import Range
from clave.world.scene import PARKED_X, PARKED_Z, SceneLayout


@dataclass(frozen=True)
class ReachReport:
    """What the geometry implies about whether a pick is possible at all.

    Attributes:
        reach_min: Inner radius of the annulus the arm is trusted over.
        reach_max: Outer radius of that annulus, in meters.
        belt_offset: Lateral distance from the shoulder to the belt centerline.
        window_length: Belt length a centerline object spends inside the
            annulus, in meters, with the dead zone already deducted.
        window_edges: Where along the belt the window opens and closes, in
            world meters. Both are zero when the belt never enters reach.
        belt_speed: Belt speed in meters per second.
        time_budget: Seconds an object spends inside the window.
    """

    reach_min: float
    reach_max: float
    belt_offset: float
    window_length: float
    window_edges: tuple[float, float]
    belt_speed: float
    time_budget: float

    @property
    def reachable(self) -> bool:
        """Whether the belt passes within the manipulator's reach at all."""

        return self.window_length > 0.0


def within_reach(position: NDArray[np.float64], plan: SceneLayout) -> bool:
    """Whether the tool can reach a point.

    The arm is trusted over an annulus about its base, within a vertical band,
    both measured by sweeping the compiled model with the tool held vertical.
    The region is smaller than what the arm can actually serve, on purpose: a
    caller that trusts it is never surprised.

    The inner radius is not a defect to work around. An object inside it is
    carried out of it by the belt, so the dead zone costs pick time rather than
    coverage.

    Args:
        position: The object's position, in world meters.
        plan: The resolved scene layout, carrying the arm base and the
            workspace.

    Returns:
        Whether the tool can be placed on the object.
    """
    lowest, highest = plan.tool_above_base
    above = float(position[2]) - float(plan.arm_base[2])
    if not lowest <= above <= highest:
        return False
    return arm.reaches(
        (float(plan.arm_base[0]), float(plan.arm_base[1])),
        float(position[0]),
        float(position[1]),
        arm.ReachBounds(plan.reach_min, plan.reach_max, plan.tool_above_base),
    )


def reach_report(plan: SceneLayout) -> ReachReport:
    """Compute the reachable window and the per-object time budget.

    The window is the chord the reachable sphere cuts through the belt
    centerline. The budget is that chord divided by belt speed, and it is the
    number every perception latency has to fit inside.

    Args:
        plan: The resolved scene layout.

    Returns:
        The report. `window_length` is zero when the belt never enters reach.
    """
    offset = abs(plan.arm_base[1])
    # Swept rather than solved, so the window is measured through exactly the
    # test the safety layer applies. A closed-form chord would be correct for
    # this arm's annulus, and was wrong for the last one; measuring costs
    # milliseconds once and cannot drift.
    #
    # The sweep spans the belt's own extent. Objects ride from
    # -length/2 to +length/2, so a sweep starting at the origin measures the
    # downstream half and reports a window half the size of the one the arm
    # actually has.
    step = 0.002
    hits = 0
    first = last = 0.0
    seen = False
    x = -plan.belt.length / 2.0
    while x <= plan.belt.length / 2.0:
        # The belt centerline is y = 0; `offset` is how far the shoulder sits
        # from it, which the sweep sees through the shoulder position rather
        # than by being passed as a coordinate.
        if arm.reaches(
            (plan.arm_base[0], plan.arm_base[1]),
            x,
            0.0,
            arm.ReachBounds(plan.reach_min, plan.reach_max, plan.tool_above_base),
        ):
            hits += 1
            last = x
            if not seen:
                first = x
                seen = True
        x += step
    window = hits * step
    return ReachReport(
        reach_min=plan.reach_min,
        reach_max=plan.reach_max,
        belt_offset=offset,
        window_length=window,
        window_edges=(first, last),
        belt_speed=plan.belt.speed,
        time_budget=window / plan.belt.speed if plan.belt.speed else 0.0,
    )


def window_exit(plan: SceneLayout) -> float | None:
    """Return the belt coordinate at which the reachable window closes.

    The scripted expert ranks objects by how little time they have left, which
    is their distance to this coordinate, so a wrong value here teaches picks
    that cannot be made.

    It is the swept downstream edge rather than half the window's length. Those
    two agree only while the arm stands at the belt centre: move the arm and the
    window moves with it, leaving half the length naming a coordinate the arm
    cannot reach.

    Args:
        plan: The resolved scene layout.

    Returns:
        The downstream edge in world meters, or None when the belt never enters
        reach and there is no edge to name.
    """
    report = reach_report(plan)
    return report.window_edges[1] if report.reachable else None


@dataclass
class SpawnedObject:
    """One object riding the belt.

    Attributes:
        index: Pool slot, which is also the body index into the model. A slot
            is reused the moment the object in it leaves the belt, so this
            identifies the place rather than the object.
        serial: Which spawn this is, counted over the whole run. Slots recycle
            and this does not, so it is the only field here that names one
            object rather than one place. Anything that has to remember what
            it already did about an object -- which track the arm has served,
            which one a active plan is aiming at -- has to key on this. A
            run that keyed on the slot served each of the pool's slots once
            and then sat idle for the rest of the rollout, because every
            object after the first pass carried an identity already in the
            served set.
        object_name: Catalog name of the mesh or primitive that was spawned.
            Empty on a record written before the corpus carried it.
        name: Body name in the model.
        material_class: Taxonomy identifier, recorded at spawn so a consumer
            never has to infer it.
        channel: Default channel for that class.
        entered_window: Whether it has been seen inside the reachable window.
    """

    index: int
    name: str
    material_class: str
    channel: str
    serial: int = 0
    entered_window: bool = False
    object_name: str = ""


@dataclass
class Conveyor:
    """Drives the belt, spawns objects, and tracks reachability.

    Attributes:
        plan: The resolved scene layout.
        rng: Generator used for spawn timing and placement.
        spacing: Range the gap to the next object is drawn from, in metres of
            belt travel rather than in seconds. A metering feeder doses by
            distance, and it is what makes belt speed a throughput knob: the
            arrival rate is speed over spacing, so a controller with speed as
            its actuator has authority over it. Dosing by time leaves the
            rate at one over the interval whatever the belt does.
        lateral: Range the lateral placement is drawn from.
        drop: Range the drop height above the belt is drawn from.
        speed: What the belt is running at now, in meters per second. Held
            here rather than read from the layout because a feed controller
            moves it, and the layout is the line as configured rather than
            the line as it is running.
    """

    plan: SceneLayout
    rng: np.random.Generator
    spacing: Range
    lateral: Range
    drop: Range
    entry_margin: float = 0.0
    active: list[SpawnedObject] = field(default_factory=list)
    speed: float | None = None
    _free: list[int] = field(default_factory=list)
    _next_free: int = 0
    _spawned: int = 0
    _travelled: float = 0.0
    _retired: int = 0
    _due: float | None = None
    _arrived: list[float] = field(default_factory=list)

    @property
    def retired(self) -> int:
        """How many objects have run off the end of the belt."""
        return self._retired

    @property
    def running(self) -> float:
        """How fast the belt is running, in meters per second."""
        return self.plan.belt.speed if self.speed is None else self.speed

    @property
    def arrivals(self) -> tuple[float, ...]:
        """When each object entered the line, in simulated seconds."""
        return tuple(self._arrived)

    @property
    def report(self) -> ReachReport:
        """The reachability report for this layout."""
        return reach_report(self.plan)

    def _place(self, model: Any, data: Any, slot: int) -> SpawnedObject:
        """Put one pooled object at the belt entrance."""
        import mujoco

        template = self.plan.objects[slot % len(self.plan.objects)]
        name = f"object_{slot}"
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        address = model.jnt_qposadr[model.body_jntadr[body]]
        data.qpos[address : address + 3] = [
            -self.plan.belt.length / 2.0 + self.entry_margin,
            self.lateral.sample(self.rng),
            self.plan.belt.surface_height + self.drop.sample(self.rng),
        ]
        data.qpos[address + 3 : address + 7] = [1.0, 0.0, 0.0, 0.0]
        velocity = model.jnt_dofadr[model.body_jntadr[body]]
        data.qvel[velocity : velocity + 6] = 0.0
        self._spawned += 1
        spawned = SpawnedObject(
            index=slot,
            name=name,
            material_class=template.material_class,
            channel=template.channel,
            serial=self._spawned,
            object_name=template.name,
        )
        self.active.append(spawned)
        return spawned

    def _hold_parked(self, model: Any, data: Any) -> None:
        """Pin every unspawned pool slot in place.

        Parked slots rest on the floor clear of the belt, so the scene is
        stable without this. Pinning them anyway keeps a recorded rollout
        identical from one run to the next, since a body settling under gravity
        is one more thing that would have to converge the same way twice.
        """
        import mujoco

        parked = list(range(self._next_free, self.plan.pool_size)) + self._free
        for slot in parked:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"object_{slot}")
            address = model.jnt_qposadr[model.body_jntadr[body]]
            velocity = model.jnt_dofadr[model.body_jntadr[body]]
            data.qpos[address : address + 3] = [
                PARKED_X + 0.3 * slot,
                PARKED_X,
                PARKED_Z,
            ]
            data.qpos[address + 3 : address + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[velocity : velocity + 6] = 0.0

    def _claim(self) -> int | None:
        """Return a pool slot to spawn into, or None when the pool is full.

        Returns:
            A slot freed by an object that has left the belt, or the next
            never-used one, or None while every slot is occupied. None is a
            real condition rather than an error: it means the line is
            carrying as many objects as the pool allows, and the feed waits.
        """
        if self._free:
            return self._free.pop(0)
        if self._next_free < self.plan.pool_size:
            slot = self._next_free
            self._next_free += 1
            return slot
        return None

    def _recycle(self, model: Any, data: Any) -> None:
        """Return objects that have run off the end of the belt to the pool.

        A compiled MuJoCo model cannot gain bodies at run time, so a line
        that never gives a slot back stops feeding after `pool_size` objects
        however long it runs. That turns a rate the line is asked to hold
        into a rate it can hold for ninety seconds, which is not the same
        claim.

        Args:
            model: The compiled model.
            data: Its state, modified in place.
        """
        import mujoco

        past = self.plan.belt.length / 2.0
        staying = []
        for item in self.active:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.name)
            address = model.jnt_qposadr[model.body_jntadr[body]]
            x = float(data.qpos[address])
            z = float(data.qpos[address + 2])
            # Off either end, or fallen below the belt into a chute or the
            # floor. The upstream end is not a formality: a body that leaves
            # the belt backwards is one no arm can reach and one the belt will
            # not carry back, so leaving it in the pool is a slot held by an
            # object the line has lost. Measured before this was here: an
            # object parked 1.5 m upstream of the entrance stayed on the active
            # list for 668 ticks while its slot was handed back and spawned
            # into, which put two objects in one pool slot.
            gone = x > past or x < -past or z < 0.05
            if gone:
                self._free.append(item.index)
                self._retired += 1
            else:
                staying.append(item)
        self.active = staying

    def step(self, model: Any, data: Any) -> None:
        """Advance spawning and belt drive by one simulation step.

        Args:
            model: The compiled model.
            data: Its state, already stepped by the caller.
        """
        import mujoco

        self._hold_parked(model, data)

        # Feed by distance, not by elapsed time. The travel is integrated
        # from the speed the belt is actually running at, so a controller
        # that slows the belt genuinely slows the feed rather than only
        # spreading the same objects further apart.
        self._recycle(model, data)
        if self._due is None:
            self._due = self.spacing.sample(self.rng)
        self._travelled += self.running * self.plan.timestep
        if self._travelled >= self._due:
            # Claimed only once a spawn is actually due. Asking for a slot
            # every tick and discarding it burns the pool in one second.
            slot = self._claim()
            if slot is not None:
                self._place(model, data, slot)
                self._travelled -= self._due
                self._due = self.spacing.sample(self.rng)
                self._arrived.append(float(data.time))

        for item in self.active:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.name)
            address = model.jnt_qposadr[model.body_jntadr[body]]
            velocity = model.jnt_dofadr[model.body_jntadr[body]]
            position = data.qpos[address : address + 3]
            on_belt = _on_belt(position, self.plan)
            if on_belt:
                # Drive travel only. Vertical motion and rotation stay with
                # physics, so contacts with the chutes and the arm remain real.
                data.qvel[velocity] = self.running
            elif _on_takeaway(position, self.plan):
                # Drive sorted objects along the take-away conveyor away from the line.
                data.qvel[velocity + 1] = -self.plan.takeaway.speed
            if on_belt and within_reach(
                np.asarray(
                    (float(position[0]), float(position[1]), float(position[2])),
                    dtype=np.float64,
                ),
                self.plan,
            ):
                item.entered_window = True

    def entered_window(self) -> list[SpawnedObject]:
        """List the objects that have been inside the reachable window."""
        return [item for item in self.active if item.entered_window]


def _on_belt(position: Any, plan: SceneLayout) -> bool:
    """Return whether a body is still resting in the driven belt region."""
    surface = plan.belt.surface_height
    tolerance = plan.belt.height_tolerance
    return bool(
        abs(position[0]) <= plan.belt.length / 2.0
        and abs(position[1]) <= plan.belt.width / 2.0
        and surface - tolerance < position[2] <= surface + tolerance
    )


def _on_takeaway(position: Any, plan: SceneLayout) -> bool:
    """Return whether a body is resting on one of the take-away conveyors."""
    x, y, z = float(position[0]), float(position[1]), float(position[2])
    drive = plan.takeaway
    if not (drive.height_min <= z <= drive.height_max):
        return False
    for cx, cy, _ in plan.chutes.values():
        if abs(x - cx) <= drive.mouth_half_width and (
            cy - drive.past_mouth <= y <= cy + drive.toward_mouth
        ):
            return True
    return False
