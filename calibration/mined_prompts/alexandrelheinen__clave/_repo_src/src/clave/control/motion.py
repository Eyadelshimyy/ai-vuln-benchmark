"""The pose to command this tick, on a path the arm can follow.

A phase says where it wants the flange. It does not say how to get there, and
commanding the goal outright is how a controller asks for a jump the actuators
answer with a lunge. Motion is the layer in between: a straight line in task
space, which is what an industrial linear move is, walked under configured
speed and acceleration ceilings.

**The first argument is the reference, not the measurement.** Motion
integrates its own previous output, seeded once from where the flange stood.
Stepping from the measured flange instead makes the reference chase the plant
rather than lead it.

**The path is kept inside a region that is not convex.** The arm is trusted
over an annulus from world configuration, so a straight line between two
admitted poses can still pass through the hole around the base. Each stepped
pose is handed to a projection that pushes it back into the trusted region.

**A goal that rides the belt is aimed ahead of itself.** The goal carries the
instant it was seen, and the intercept is carried from there to now plus the
traverse. Belt speed is known here rather than estimated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from clave.control.settings import MotionSettings
from clave.control.task import Goal
from clave.control.trajectory import clip, distance, same, zeros
from clave.tracker.belt_frame import carry

NANOS_PER_SECOND = 1_000_000_000
"""Nanoseconds in a second, for the belt travel the intercept carries."""


@dataclass(frozen=True)
class Reference:
    """Where the reference stands and how fast it is going.

    Attributes:
        position: The pose commanded last tick.
        speed: How fast the reference was moving, in meters per second. Held
            across ticks because an acceleration bound is a statement about
            the change in this, and a caller that forgets it asks for a step
            in velocity every time it starts.
    """

    position: NDArray[np.float64]
    speed: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Reference):
            return NotImplemented
        return same(self.position, other.position) and self.speed == other.speed


@dataclass(frozen=True)
class Command:
    """The pose to command this tick.

    Attributes:
        position: Where the flange should be by the end of the tick.
        speed: How fast the reference is now moving, to be fed back next
            tick.
        velocity: Which way it is moving and how fast, as a vector. The
            servo needs the direction as well as the magnitude, because it
            leads the plant by a time rather than by a distance.
        yaw: What the tool should be turned to about the belt normal, or None
            to hold the rotation the arm already has.
        aim: The pose the step was aimed at, which is the goal itself unless
            the goal rides the belt. Reported so a run can show where the arm
            expected to meet an object rather than only where it went. None
            when the command was built directly rather than by [toward],
            which is how a caller asks for one pose and nothing else.
    """

    position: NDArray[np.float64]
    yaw: float | None
    speed: float = 0.0
    velocity: NDArray[np.float64] = field(default_factory=zeros)
    aim: NDArray[np.float64] | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Command):
            return NotImplemented
        if self.yaw != other.yaw or self.speed != other.speed:
            return False
        if not same(self.position, other.position):
            return False
        if not same(self.velocity, other.velocity):
            return False
        if self.aim is None or other.aim is None:
            return self.aim is other.aim
        return same(self.aim, other.aim)


def toward(
    reference: Reference,
    goal: Goal,
    timestep: float,
    limits: MotionSettings,
    belt_speed: float,
    at_nanos: int,
    keep_inside: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> Command:
    """Return the pose to command one tick along the way to a goal.

    Args:
        reference: Where the reference stands and how fast it is going. Seed it
            from the flange at rest once, at the start of a motion, and feed
            this function's own output back into it afterwards. Passing the
            measured flange every tick makes the reference chase the plant
            instead of leading it.
        goal: Where the phase wants it, what rotation it asked for, and
            whether it rides the belt.
        timestep: How long this tick lasts, in seconds.
        limits: The speed and acceleration ceilings.
        belt_speed: How fast the belt runs, in meters per second, used to aim
            ahead of a goal that rides it.
        at_nanos: Now, against which the goal's own instant says how stale
            the pose it carries has become.
        keep_inside: Pushes a pose back into the region the arm is trusted
            over, or None to step without that constraint. Supplied rather
            than computed here, so the region is the one `clave.world.arm`
            enforces and cannot drift from it.

    Returns:
        The command.
    """
    inside = keep_inside if keep_inside is not None else _unchanged
    aim = _intercept(
        reference.position,
        goal,
        limits.max_speed,
        belt_speed,
        at_nanos,
        limits.intercept_passes,
    )

    remaining = distance(reference.position, aim)
    speed = _speed(reference.speed, remaining, timestep, limits)
    reach = speed * timestep
    if remaining <= reach or remaining == 0.0:
        # Arriving does not licence an instant stop. Bleeding the last of the
        # speed off at the same rate everything else changes is what keeps
        # the bound true on the tick that matters most, which is the last
        # one.
        settling = max(0.0, reference.speed - limits.max_acceleration * timestep)
        return Command(position=inside(aim), speed=settling, yaw=goal.yaw, aim=aim)

    fraction = reach / remaining
    stepped = reference.position + (aim - reference.position) * fraction
    direction = (aim - reference.position) / remaining
    return Command(
        position=inside(stepped),
        speed=speed,
        velocity=direction * speed,
        yaw=goal.yaw,
        aim=aim,
    )


def _intercept(
    reference: NDArray[np.float64],
    goal: Goal,
    max_speed: float,
    belt_speed: float,
    at_nanos: int,
    passes: int,
) -> NDArray[np.float64]:
    """Return where to aim, ahead of a goal the belt is carrying.

    Args:
        reference: Where the reference stands.
        goal: The goal, which says whether it rides the belt and when its
            pose was seen.
        max_speed: The reference's speed ceiling, in meters per second.
        belt_speed: How fast the belt runs, in meters per second.
        at_nanos: Now.

    Returns:
        The goal itself when it stands still, and otherwise where the object
        will be at the instant the reference could get there, carried from
        when it was seen rather than from now.
    """
    if not goal.rides_belt or belt_speed <= 0.0 or max_speed <= 0.0:
        return goal.position
    aim = carry(goal.position, belt_speed, goal.observed_at_nanos, at_nanos)
    for _ in range(passes):
        seconds = distance(reference, aim) / max_speed
        aim = carry(
            goal.position,
            belt_speed,
            goal.observed_at_nanos,
            at_nanos + int(seconds * NANOS_PER_SECOND),
        )
    return aim


def _speed(
    speed: float, remaining: float, timestep: float, limits: MotionSettings
) -> float:
    """Return how fast the reference may move this tick.

    A trapezoidal profile: climb toward the ceiling while there is room to
    stop, and give the speed back once there is not. Arriving at speed is not
    arriving, because the reference overshoots and comes back, which reads on
    a plot as an arm that cannot settle.

    Args:
        speed: How fast the reference was going.
        remaining: How far it still has to go, in meters.
        timestep: How long this tick lasts, in seconds.
        limits: The speed and acceleration ceilings.

    Returns:
        The speed for this tick, never negative and never above the ceiling.
    """
    step = limits.max_acceleration * timestep
    # The stopping curve is read where the tick will end rather than where it
    # began. Reading it at the start samples a continuous curve one tick late,
    # and the profile then enters the braking leg above the bound. Looking
    # ahead keeps the speed on the curve through the tick.
    ahead = max(0.0, remaining - speed * timestep)
    stoppable = float(np.sqrt(max(0.0, 2.0 * limits.max_acceleration * ahead)))
    wanted = clip(speed + step, min(limits.max_speed, stoppable))
    # What is left of that residual is clamped away. On its own this floor is
    # harmful: without the look-ahead it holds the speed above the stopping
    # curve, the error compounds over the braking leg, and the profile arrives
    # fast and stops dead. With the look-ahead it binds only on the last
    # fraction of a percent, and the bound then holds exactly.
    return max(wanted, max(0.0, speed - step))


def _unchanged(pose: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the pose as given, for a caller that supplied no constraint.

    Args:
        pose: The pose.

    Returns:
        The same pose.
    """
    return pose
