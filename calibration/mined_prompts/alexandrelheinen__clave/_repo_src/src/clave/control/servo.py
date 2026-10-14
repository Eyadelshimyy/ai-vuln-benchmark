"""One commanded pose, turned into joint angles and written to the actuators.

The only module in the control path that compiles a model, and the only one
with no decisions in it. It takes the pose motion produced, asks
`clave.world.arm` for joint angles that put the flange there with the tool
vertical, and writes them.

**It leads the plant rather than following it.** These are position
actuators running a proportional-derivative loop, so the measured flange
trails a moving command by a lag proportional to the commanded speed.
Measured on the shipped arm by sweeping a target from 0.15 to 1.00 metres
per second, that lag divided by the speed is 32.8 to 38.6 milliseconds and
is otherwise flat, which is what makes it a time constant rather than a
distance. Commanding the pose the reference will hold one such constant
from now cancels most of it.

That is not cosmetic. With the jaw's clear opening at 85.2 mm, the widest
object left in the set leaves 8.7 mm of side clearance, and a flange
trailing by more than that closes the jaw onto the object rather than
around it.

It reports a refusal rather than swallowing one. `arm.step_toward` leaves the
command untouched for a target outside the trusted region, which is correct
for a controller and useless for a report: an arm that quietly stops moving
looks identical to an arm that arrived. So the reach is checked here and the
refusal travels back, which is what lets the task machine fault the track and
the run count it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clave.control.motion import Command
from clave.control.pick import nearer_parallel_yaw
from clave.world import arm as armmod


@dataclass(frozen=True)
class Step:
    """What one servo tick did.

    Attributes:
        joints: The joint angles commanded, in joint order.
        refusal: Why the pose was refused, or None when it was taken.
    """

    joints: NDArray[np.float64]
    refusal: str | None


def follow(
    model: Any,
    data: Any,
    arm: armmod.ArmIndices,
    command: Command,
    gain: float,
    max_joint_step: float | None = None,
    lead_seconds: float = 0.0,
    keep_inside: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> Step:
    """Command one pose and report whether the arm could take it.

    Args:
        model: The compiled model.
        data: Its state, with forward kinematics already current.
        arm: The arm indices.
        command: The pose to command.
        gain: Fraction of the solved step to command.
        max_joint_step: The furthest any one joint may be commanded to move,
            in radians, or None for no cap. This is what holds near a wrist
            singularity, where the damped solve still asks for a large joint
            motion to buy a small Cartesian one.
        lead_seconds: How far ahead of the commanded pose to aim, in
            seconds, to cancel the lag a position loop has against a moving
            command. Zero commands the pose as given.
        keep_inside: Pushes the led pose back into the region the arm is
            trusted over. Without it a lead near the edge of the annulus
            would be refused for the lead rather than for the pose, which
            would fault a target the arm can perfectly well serve.

    Returns:
        The step. A refused pose writes no actuator command, so the arm holds
        whatever it was last told.
    """
    led_position_world = command.position + command.velocity * lead_seconds
    if keep_inside is not None:
        led_position_world = keep_inside(led_position_world)
    target_position_world = np.asarray(led_position_world, dtype=np.float64)

    if not armmod.reachable(arm, target_position_world):
        # Hold, rather than write nothing. An uncommanded arm sags under
        # gravity, the sag puts the flange outside the trusted band, and from
        # there every pose is refused for a reason the controller caused. The
        # command written is where the arm already is, which is derived from
        # the arm and not from the pose that was refused.
        holding = armmod.joint_positions(model, data, arm)
        for slot, actuator in enumerate(arm.actuator_ids):
            data.ctrl[actuator] = holding[slot]
        return Step(joints=holding, refusal=_why(arm, target_position_world))

    # A goal that asked for no rotation gets the one the tool already holds,
    # which is how an unoriented footprint reaches the actuators without
    # anybody inventing an angle for it. A parallel jaw is the same grip at
    # yaw and yaw+π; take the nearer one so the wrist does not fold the long
    # way around (`AC-BEHAVE-03`).
    if command.yaw is None:
        yaw = armmod.tool_yaw(data, arm)
    else:
        yaw = nearer_parallel_yaw(float(command.yaw), armmod.tool_yaw(data, arm))
    return Step(
        joints=armmod.step_toward(
            model,
            data,
            arm,
            target_position_world,
            gain=gain,
            yaw=yaw,
            max_joint_step=max_joint_step,
        ),
        refusal=None,
    )


def _why(arm: armmod.ArmIndices, target_position_world: NDArray[np.float64]) -> str:
    """Return why a pose is outside the region the arm is trusted over.

    Args:
        arm: The arm indices.
        target_position_world: The pose that was refused.

    Returns:
        A sentence naming which bound it broke, so a fault says something a
        reader can act on rather than that something went wrong.
    """
    radius = float(
        np.linalg.norm(
            np.asarray(target_position_world, dtype=np.float64)[:2]
            - np.asarray(arm.base_position, dtype=np.float64)[:2]
        )
    )
    if not arm.reach.reach_min <= radius <= arm.reach.reach_max:
        return (
            f"{radius:.3f} m from the base, outside the "
            f"{arm.reach.reach_min:.2f} m to "
            f"{arm.reach.reach_max:.2f} m annulus"
        )
    lowest, highest = arm.reach.tool_above_base
    above = float(target_position_world[2]) - float(arm.base_position[2])
    return (
        f"{above:+.3f} m from the base, outside the trusted "
        f"{lowest:+.2f} m to {highest:+.2f} m band"
    )
