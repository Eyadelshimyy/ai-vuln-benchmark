"""The phases of one visit, and the pose each phase asks for.

A visit is what the arm does about one object: go to it, stay with it long
enough to prove it arrived, and move on. The machine says which phase it is
in and where it wants the flange; it decides nothing about the path there,
which is motion, and nothing about joint angles, which is the servo.

Under the motion-only profile the phases are standby, tracking, parking and
fault. That is what tuning arm speed against belt speed needs: the flange
rides at approach height the whole time, so nothing about a descent muddies
what a distance figure means.

**A full visit is planned, not stepped.** The motion-only path asks for a
pose and finds out when it arrives by arriving, which is enough to follow an
object and not enough to pick one up: a jaw has to reach an object at a
known instant, moving with it, having come straight down onto it. So the
full-visit path hands the whole visit to [clave.control.pick] the moment it
commits to a candidate, gets back a sequence of timed arcs, and thereafter
drives the arm from the clock rather than from proximity.

That makes the two profiles differ in how they are driven and not only in
how many phases they run, which is why a caller asks [TaskMachine.active]
which regime it is in rather than reading the phase.

**A plan owns which object and when; not where.** Re-deciding which object
a visit is about would turn an interception into a chase, and re-deciding
the arrival time would leave nothing able to schedule against it, so both
are fixed the moment the machine commits. Where the jaw closes is not. The
aim is refreshed on every new estimate until the jaw shuts, because the
belt model predicts travel along the belt and predicts a roll across it
only for as long as that roll lasts, and an object drifts out from under a
pose solved three seconds ago. Through the approach the arc is bent onto
the fresher estimate. Once the descent has started there is no approach
left to bend, and solving the visit again would send the arm back up, so
only the end of the descent moves. A correction that breaks a ceiling
leaves the plan already in hand.

**A refresh that cannot follow the object is answered, not ignored.** An
approach arc can only be bent so far, and an object that has fallen behind
its own prediction -- one dragging on the belt, or one that has stopped
drifting across it -- cannot be met by an arc already committed further
downstream. The refinement is refused, and a machine that reacts by running
the plan it already had descends onto bare belt while reporting a
millimetre arrival error against its own commands. So the freshest estimate
is compared with the pose the plan is aiming at: inside the configured
tolerance the refusal is noise and the plan flies, past it the visit is
solved again from the freshest estimate and may well choose a later
interception, and when no interception exists at all the visit is abandoned
with its reason recorded. Both cases are in the run's report, which is the
only way to tell a machine that gave up on an object from one that never
saw it.

A refusal is the one thing that tears a plan up, because a pose the solver
will not take is a pose the rest of the sequence was built on.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from clave.control.pick import (
    Plan,
    VisitTick,
    nearer_parallel_yaw,
    plan_pick,
    refine,
    retarget_descent,
)
from clave.control.selection import Candidate, Queue
from clave.control.settings import (
    CalibrationSettings,
    MotionSettings,
    Phase,
    Profile,
    TaskSettings,
)
from clave.control.story import StoryLog
from clave.control.trajectory import State, distance, same, where_carried, zeros
from clave.errors import ClaveError
from clave.world.effector import Effector

LOGGER = logging.getLogger(__name__)

NANOS_PER_SECOND = 1_000_000_000
"""Nanoseconds in a second, for the instant a goal records."""


class TaskError(ClaveError):
    """The task machine cannot run this configuration."""


@dataclass(frozen=True)
class Reaim:
    """What one refresh of a active plan found, and what was done about it.

    A visit is planned once and re-aimed every capture, and the interesting
    case is the one where the re-aim does not land: the freshest estimate says
    the object will not be where the plan is going. Recorded rather than
    logged and forgotten, because the figure that separates a run that
    abandoned a visit it could not serve from one that flew a stale plan onto
    bare belt is this drift, and no arrival error can show it.

    Attributes:
        at_seconds: When the refresh ran.
        track_id: Which track the visit was about.
        drift: How far the freshest estimate of the grasp pose stood from the
            pose the plan was aiming at, in meters, or None when the object
            was no longer among the candidates and there was nothing to
            measure against.
        refused: Whether the solver produced no corrected arc for the freshest
            estimate. An arc on a ceiling is refused rather than bent, so a
            refusal is the ordinary outcome of an object that has fallen
            behind its own prediction.
        action: What the machine did about it: "took" the corrected arc, "held"
            the plan it had, "solved" the visit again from the freshest
            estimate, or "abandoned" it and went for something else.
    """

    at_seconds: float
    track_id: int
    drift: float | None
    refused: bool
    action: str


def _aim_of(plan: Plan) -> NDArray[np.float64]:
    """Return the pose a plan descends onto, which is the pose it is aiming at.

    Args:
        plan: The plan.

    Returns:
        The end of its descent leg, in world frame meters. Everything after
        that end is hung off it, so this is where the jaws will close.
    """
    return next(
        leg.segment.end.position for leg in plan.legs if leg.phase is Phase.DESCEND
    )


@dataclass(frozen=True)
class Goal:
    """Where a phase wants the flange, and which phase asked.

    Attributes:
        phase: Which phase the arm is in.
        target_position_world: Where the flange should be, in world frame meters.
        target_yaw_world: What the tool should be turned to about the belt
            normal, or None to hold the rotation the arm already has.
        track_id: Which track this is about, or None when no track is.
        observed_at_nanos: When this pose was seen. A goal is decided once
            per capture and held for half a second while the belt keeps
            moving, so a consumer aiming at it has to know how old it is.
        rides_belt: Whether the belt is carrying this pose while the arm
            travels to it. True for an object and false for the park pose,
            which is what tells motion whether to aim ahead of it.
    """

    phase: Phase
    target_position_world: NDArray[np.float64]
    target_yaw_world: float | None = None
    track_id: int | None = None
    observed_at_nanos: int = 0
    rides_belt: bool = False

    def __init__(
        self,
        phase: Phase,
        target_position_world: NDArray[np.float64] | None = None,
        target_yaw_world: float | None = None,
        track_id: int | None = None,
        observed_at_nanos: int = 0,
        rides_belt: bool = False,
        *,
        position: NDArray[np.float64] | None = None,
        yaw: float | None = None,
    ) -> None:
        pos = target_position_world if target_position_world is not None else position
        if pos is None:
            raise TypeError("Goal requires target_position_world or position")
        y = target_yaw_world if target_yaw_world is not None else yaw
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "target_position_world", pos)
        object.__setattr__(self, "target_yaw_world", y)
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "observed_at_nanos", observed_at_nanos)
        object.__setattr__(self, "rides_belt", rides_belt)

    @property
    def position(self) -> NDArray[np.float64]:
        """Backwards compatibility alias for target_position_world."""
        return self.target_position_world

    @property
    def yaw(self) -> float | None:
        """Backwards compatibility alias for target_yaw_world."""
        return self.target_yaw_world

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Goal):
            return NotImplemented
        return (
            self.phase is other.phase
            and same(self.target_position_world, other.target_position_world)
            and self.target_yaw_world == other.target_yaw_world
            and self.track_id == other.track_id
            and self.observed_at_nanos == other.observed_at_nanos
            and self.rides_belt == other.rides_belt
        )


class TaskMachine:
    """Runs one visit at a time.

    Holds three things across ticks: which track it is serving, when the
    flange first came inside tolerance of it, and which tracks it has already
    served or faulted. Everything else it reads from the queue it is handed.
    """

    def __init__(
        self,
        settings: TaskSettings,
        calibration: CalibrationSettings,
        belt_surface_height_world: float | None = None,
        belt_speed: float = 0.0,
        motion: MotionSettings | None = None,
        admits: Callable[[NDArray[np.float64]], bool] | None = None,
        chutes: dict[str, NDArray[np.float64]] | None = None,
        belt_width: float = 0.50,
        belt_center_y: float = 0.0,
        belt_border_y: float | None = None,
        *,
        belt_surface: float | None = None,
        effector: Effector | None = None,
        story: StoryLog | None = None,
    ) -> None:
        """Hold the settings and the belt the approach height is measured from.

        Args:
            settings: The phases, the tolerances and the park pose.
            calibration: The offset between a marker's pose and the flange.
            belt_surface_height_world: Height of the belt surface, in meters.
            belt_speed: How fast the belt runs, in meters per second, which a
                planned visit needs to know where the object will be.
            motion: The speed and acceleration ceilings a planned arc has to
                respect. Required under the full-visit profile and unused
                under motion-only, which is stepped rather than planned.
            admits: Whether the arm is trusted at a pose, or None to plan
                without that check. Supplied rather than computed here, so
                the region a pick aims inside is the one the safety layer
                enforces and cannot drift from it.
            chutes: Where each channel's chute mouth stands. A visit ends
                over the one its object routes to; without them it ends at
                the retreat and the object goes back on the belt, which is
                a line with nowhere to put anything.
            belt_width: Width of the belt in meters.
            belt_center_y: Lateral center coordinate of the belt in meters.
            belt_border_y: Lateral position of the belt border on the chute side,
                or None to compute from belt_center_y - belt_width / 2.0.
            belt_surface: Legacy keyword alias for belt_surface_height_world.
            story: The debug narrative. One machine shares the log the run
                notes materials on, so a sentence can name the material. None
                builds a log that emits only when DEBUG is enabled.

        Raises:
            TaskError: If the profile plans arcs and was given neither the
                ceilings to plan them under nor a belt to plan them against.
        """
        surface = (
            belt_surface_height_world
            if belt_surface_height_world is not None
            else (0.0 if belt_surface is None else belt_surface)
        )
        if settings.profile is Profile.FULL_VISIT:
            if motion is None:
                raise TaskError(
                    "the full-visit profile plans arcs and needs the speed and "
                    "acceleration ceilings to plan them under"
                )
            if not belt_speed > 0.0:
                raise TaskError(
                    f"the full-visit profile intercepts a moving object, and a "
                    f"belt at {belt_speed} m/s carries nothing to intercept"
                )
            if effector is None:
                raise TaskError(
                    "the full-visit profile descends onto a belt and needs the "
                    "jaw's geometry to know how close it may come to one"
                )
        self._settings = settings
        self._calibration = calibration
        self._belt_surface_height_world = surface
        self._belt_speed = belt_speed
        self._motion = motion
        self._effector = effector
        self._admits = admits if admits is not None else _anywhere
        self._chutes = chutes or {}
        self._belt_border_y = (
            belt_border_y
            if belt_border_y is not None
            else (belt_center_y - belt_width / 2.0)
        )
        safe_clearance: float = (
            settings.safe_clearance
            if settings.safe_clearance is not None
            else settings.approach_height
        )
        self._safe_height_world = self._belt_surface_height_world + safe_clearance
        # The flange height below which the open jaw's lowest geometry is
        # inside the clearance it keeps above the belt. The shut jaw hangs
        # further, and the hold rises by that difference while the fingers
        # close. Zero when no effector was given, which only happens under
        # motion-only, where the flange never descends.
        self._grasp_floor_world = (
            0.0
            if effector is None
            else self._belt_surface_height_world + effector.open_flange_floor
        )
        self._jaw_below_flange_world = (
            0.0 if effector is None else effector.open_lowest_below_flange
        )
        self._shut_flange_floor_world = (
            0.0
            if effector is None
            else self._belt_surface_height_world + effector.flange_floor
        )
        self._serving: int | None = None
        self._arrived_at: float | None = None
        self._phase = Phase.STANDBY
        self._served: list[int] = []
        self._faults: list[tuple[int | None, str]] = []
        self._arrivals: list[float] = []
        self._plan: Plan | None = None
        self._plan_yaw: float | None = None
        self._last_yaw: float | None = None
        self._pick_error: float | None = None
        self._missed: list[int] = []
        self._refreshes: list[Reaim] = []
        self._abandoned: list[tuple[int, str]] = []
        self._story = story if story is not None else StoryLog()
        self._plan_refusal = ""
        self._told_reaim: tuple[int, str] | None = None

    @property
    def belt_surface(self) -> float:
        """Backwards compatibility alias for belt_surface_height_world."""
        return self._belt_surface_height_world

    @property
    def served(self) -> tuple[int, ...]:
        """Every track this machine finished a visit to, in order."""
        return tuple(self._served)

    @property
    def arrivals(self) -> tuple[float, ...]:
        """How far the flange was from its goal at the moment that decides it.

        One figure per completed visit, in meters, and the moment differs by
        profile because what the profile is for differs. A motion-only visit
        is measured at the end of its dwell, which is when it claims to have
        arrived. A planned visit is measured at the instant the jaw reaches
        the object, which is the only instant a pick is decided at.

        Either way the arm is judged against the pose it was asked for, not
        against the object: whether that pose was the right pose is the
        tracker's question and is measured separately.
        """
        return tuple(self._arrivals)

    @property
    def missed(self) -> tuple[int, ...]:
        """Every track no interception existed for, in order.

        Distinct from a fault. A fault is the solver or the envelope refusing
        a pose, which is a problem with the pose. This is the arm being too
        far away or the object too near the end of the belt, which is a
        problem with the timing and is the normal way a line drops work.
        """
        return tuple(self._missed)

    @property
    def refreshes(self) -> tuple[Reaim, ...]:
        """Every re-aim of a active plan, in order, with what came of it.

        One entry per capture a visit was active for, so a run of any length
        reports a bounded list: the figure a reader wants is the largest drift
        a plan was allowed to keep, and the actions that were taken instead of
        keeping it.
        """
        return tuple(self._refreshes)

    @property
    def abandoned(self) -> tuple[tuple[int, str], ...]:
        """Every visit given up before the descent began, and why.

        Distinct from a miss, which is an interception that was never planned:
        this is a plan that was active and that the freshest estimate
        contradicted. Recorded with its reason because that is the difference
        between an arm that stopped working and an arm that was told to.
        """
        return tuple(self._abandoned)

    @property
    def worst_aim_drift(self) -> float | None:
        """The largest aim error a active visit was found holding, in meters.

        None when no visit was ever refreshed, which is the honest answer for
        a run that never planned one.
        """
        drifts = [
            refresh.drift for refresh in self._refreshes if refresh.drift is not None
        ]
        return max(drifts) if drifts else None

    @property
    def belt_speed(self) -> float:
        """How fast the machine believes the belt is running.

        Settable, because a line whose feed rate is regulated moves its belt
        and an interception solved against the speed the run drew is wrong by
        however far the controller has trimmed it since.
        """
        return self._belt_speed

    @belt_speed.setter
    def belt_speed(self, metres_per_second: float) -> None:
        self._belt_speed = metres_per_second

    @property
    def active(self) -> bool:
        """Whether a planned visit is currently driving the arm.

        A caller reads this to know which regime it is in: while it is true
        the arm is driven from [TaskMachine.tick] and motion is out of
        the path, because the arcs already respect the ceilings motion
        would otherwise enforce.
        """
        return self._plan is not None

    @property
    def plan(self) -> Plan | None:
        """Return the active visit plan, if one is currently active."""
        return self._plan

    @property
    def faults(self) -> tuple[tuple[int | None, str], ...]:
        """Every refusal, with the track it was about.

        The track is None when a refusal arrived with nothing being served
        and nothing queued. That is recorded rather than dropped, because a
        refusal nobody counted is the one way the arm can stop working and
        leave no trace of why.
        """
        return tuple(self._faults)

    def step(
        self,
        queue: Queue,
        flange_position_world: NDArray[np.float64] | None = None,
        at_seconds: float = 0.0,
        refusal: str | None = None,
        reference_velocity: NDArray[np.float64] | None = None,
        *,
        flange: NDArray[np.float64] | None = None,
    ) -> Goal:
        """Decide what the arm is doing, and say where it wants the flange.

        Called at the cadence decisions are made at, which is slower than the
        cadence the arm is driven at. Under a planned visit the driving is
        [TaskMachine.tick]; this only commits to a candidate and then keeps
        out of the way until the plan runs out.

        Args:
            queue: The order selection produced.
            flange_position_world: Where the flange stands in world frame.
            at_seconds: Simulated time, which the dwell is measured against.
            refusal: Why the last commanded pose was refused, or None.
            reference_velocity: How fast the reference is moving, so a plan
                begins where the motion already is rather than asking for a
                step in velocity. Zero is right whenever the arm is at rest,
                which is where a visit normally starts.
            flange: Legacy keyword alias for flange_position_world.

        Returns:
            The goal for this tick.
        """
        flange_pos = (
            flange_position_world if flange_position_world is not None else flange
        )
        if flange_pos is None:
            raise TypeError("step requires flange_position_world or flange")
        at_nanos = int(at_seconds * NANOS_PER_SECOND)
        if reference_velocity is None:
            reference_velocity = zeros()
        if refusal is not None:
            return self._fault(queue, flange_pos, refusal, at_nanos)

        if self._plan is not None:
            self._reaim(queue, flange_pos, reference_velocity, at_seconds)
            if self._plan is not None:
                return self._held(at_nanos)
            # The visit was abandoned: the freshest estimate contradicted the
            # aim, so the plan is gone and the object it was about is recorded
            # as missed. Carrying on here rather than returning means the arm
            # reaches for the next candidate on this capture instead of
            # standing still for one, which on a line that keeps moving is the
            # difference between one visit lost and two.

        head = self._next(queue)
        if head is None:
            self._serving, self._arrived_at = None, None
            return self._rest(flange_pos, at_nanos)

        if self._settings.profile is Profile.FULL_VISIT:
            # A head with no interception is recorded as missed inside
            # `_commit` and must not park the arm while a later candidate
            # still has belt left (`AC-BEHAVE-01`).
            while True:
                missed_before = len(self._missed)
                goal = self._commit(
                    head, flange_pos, reference_velocity, at_seconds, at_nanos
                )
                if self._plan is not None:
                    return goal
                # `_commit` parks without marking a miss when the flange itself
                # is outside the trusted region; retrying the same head would
                # spin forever.
                if len(self._missed) == missed_before:
                    return goal
                head = self._next(queue)
                if head is None:
                    return goal

        if head.track_id != self._serving:
            self._serving, self._arrived_at = head.track_id, None
            self._phase = Phase.TRACK
            self._story.note(head.track_id, "", head.channel)
            self._story.tell(
                at_seconds,
                "the queue head is this object and the profile follows markers "
                "rather than planning a grasp",
                f"track {self._story.refer(head.track_id)} at approach height",
                "pending",
                channel=head.channel,
            )

        goal = self._track(head, at_nanos, self._phase)
        gap = distance(flange_pos, goal.position)
        if gap > self._settings.arrival_tolerance:
            # Not there yet, so the dwell has not started. A visit that
            # completes because time passed rather than because the arm
            # arrived makes every distance figure meaningless.
            self._arrived_at = None
            return goal

        if self._arrived_at is None:
            self._arrived_at = at_seconds
            return goal
        if at_seconds - self._arrived_at < self._settings.dwell_seconds:
            return goal

        # A stepped visit is one phase long: the arm goes to the marker and
        # moves on. Everything past tracking belongs to a plan, and a plan
        # never reaches here.
        self._arrivals.append(gap)
        self._served.append(head.track_id)
        self._story.tell(
            at_seconds,
            f"the flange held within tolerance for the dwell, {gap * 1000:.1f} mm "
            "from the marker",
            f"count the visit to {self._story.refer(head.track_id)} served and "
            "take the next object",
            "success",
            channel=head.channel,
        )
        self._serving, self._arrived_at = None, None
        self._phase = Phase.STANDBY
        return self.step(queue, flange_pos, at_seconds)

    def tick(
        self,
        at_seconds: float,
        flange_position_world: NDArray[np.float64] | None = None,
        *,
        flange: NDArray[np.float64] | None = None,
    ) -> VisitTick | None:
        """Drive one tick of a planned visit, or report that none is active.

        Called at the rate the arm is actually commanded at, which is the
        physics rate: a goal half a second old is still the right goal, and a
        joint command half a second old is a lurch.

        Args:
            at_seconds: Simulated time.
            flange_position_world: Where the flange stands, read only to
                measure how near it came to the object when the jaw closed.
            flange: Legacy keyword alias for flange_position_world.

        Returns:
            What to command, or None when no plan is active. None on the tick
            a plan runs out, which is also when the visit is recorded, so a
            caller that stops asking on None never misses the end.
        """
        flange_pos = (
            flange_position_world if flange_position_world is not None else flange
        )
        if flange_pos is None:
            raise TypeError("tick requires flange_position_world or flange")
        if self._plan is None:
            return None
        sampled = self._plan.at(at_seconds)
        if sampled is None:
            self._finish(flange_pos, at_seconds)
            return None
        phase, state, grip = sampled
        if phase is not self._phase:
            self._narrate_phase(phase, at_seconds)
        if phase is Phase.HOLD and self._pick_error is None:
            # The hold begins at the instant the jaw reaches the object, so
            # the first tick of it is the one a pick is decided on. Later
            # ticks are the carry, and measuring there would report how well
            # the arm rides the belt rather than how well it arrived.
            self._pick_error = distance(flange_pos, state.position)
        self._phase = phase
        yaw = self._plan.yaw_at(at_seconds)
        if yaw is None:
            yaw = self._plan_yaw
        self._last_yaw = yaw

        return VisitTick(
            phase=phase,
            position=state.position,
            velocity=state.velocity,
            yaw=yaw,
            grip=grip,
        )

    def _narrate_phase(self, phase: Phase, at_seconds: float) -> None:
        """Say that an active visit has entered another phase.

        Args:
            phase: The phase the clock just reached.
            at_seconds: Simulated time.
        """
        track = self._plan.track_id if self._plan is not None else self._serving
        who = self._story.refer(track)
        chute = self._story.channel_of(track)
        if phase is Phase.DESCEND:
            self._story.tell(
                at_seconds,
                "the approach is over and the plan says to come down",
                f"descend onto {who}, moving with the belt",
                "pending",
                channel=chute,
            )
        elif phase is Phase.HOLD:
            self._story.tell(
                at_seconds,
                "the descent has reached the grasp pose",
                f"close the jaw on {who}",
                "pending",
                channel=chute,
            )
        elif phase is Phase.RETREAT:
            self._story.tell(
                at_seconds,
                "the hold is over",
                f"lift clear of the belt with the jaw shut on {who}",
                "pending",
                channel=chute,
            )
        elif phase is Phase.DELIVER:
            if self._story.missed_the_grasp(track):
                self._story.tell(
                    at_seconds,
                    "the jaw closed with nothing in it and the plan still "
                    "has a release",
                    f"fly the release of {who} anyway",
                    "fail",
                    channel=chute,
                )
            else:
                self._story.tell(
                    at_seconds,
                    "the jaw is shut",
                    f"put {who} in its chute",
                    "pending",
                    channel=chute,
                )

    def _takes(self, pose: NDArray[np.float64]) -> bool:
        """Whether the arm may be commanded to a pose at all.

        Two questions, and both have to pass. The first is whether the arm is
        trusted over the pose, which is the safety layer's question and is
        answered by the region the caller enforces. The second is whether the
        jaw would come closer to the belt than the clearance it keeps, which is
        the one thing about a grasp pose a region cannot see: the pose is the
        flange, and the jaw hangs below it.

        A marker never asks for a pose that fails the second question, because
        the marker's own plane is clamped by the same clearance. This catches a
        plan that has drifted -- an aim re-solved against a moving estimate, or
        a configuration whose numbers no longer describe the jaw the model
        carries -- and it catches it before the servo pushes the pads into the
        belt.
        Args:
            pose: The flange pose in world frame meters.

        Returns:
            Whether it is inside the region and above the belt clearance.
        """
        return self._admits(pose) and pose[2] >= self._grasp_floor_world - 1e-9

    def _grasp_pose(self, head: Candidate) -> NDArray[np.float64]:
        """Return where the flange has to sit to grasp a candidate.

        Args:
            head: The candidate.

        Returns:
            The marker's flange pose with the calibration offset applied,
            which is the pose a plan aims at and the pose a stepped visit
            descends to. One place, so the two cannot disagree.
        """
        return head.flange_position_world + self._calibration.flange_offset

    def _narrate_reaim(
        self,
        at_seconds: float,
        track_id: int,
        action: str,
        because: str,
        doing: str,
    ) -> None:
        """Narrate a re-aim when the decision changed.

        ``_reaim`` runs once a capture. A capture that repeats the same
        action on the same object is the visit continuing, and the report
        already keeps every refresh. The narrative keeps the change.

        Args:
            at_seconds: Simulated time.
            track_id: The object the visit is about.
            action: ``took``, ``held``, ``solved``, or ``descent``.
            because: The cause, without a leading "because".
            doing: What the arm will do, without a leading "I will".
        """
        decision = (track_id, action)
        if decision == self._told_reaim:
            return
        self._told_reaim = decision
        self._story.tell(
            at_seconds,
            because,
            doing,
            "pending",
            channel=self._story.channel_of(track_id),
        )

    def _reaim(
        self,
        queue: Queue,
        flange: NDArray[np.float64],
        velocity: NDArray[np.float64],
        at_seconds: float,
    ) -> None:
        """Point the active plan at a fresher estimate of its object, or give it up.

        The arrival time does not move, so everything downstream of this is
        still scheduled against the same instant. What moves is where the
        approach ends, which is what stops a three second old prediction from
        deciding where a jaw closes -- as long as the approach arc can still be
        bent onto the fresh estimate.

        It cannot always. An object dragging on the belt, or one that has
        stopped drifting across it, falls *behind* the prediction the plan was
        built from, and no arc takes the arm back up the belt to meet it: the
        refinement is refused, and this used to hold the stale plan and fly it.
        On the run that prompted this that is what happened, twice, and the arm
        descended onto bare belt 338 mm and 231 mm from the object the visit
        was supposedly about.

        So a refusal is answered rather than ignored. Within the configured
        tolerance the plan is still aimed where the object is and the refusal
        is noise. Past it the visit is solved again from the freshest estimate,
        starting at the motion reference the plan already holds (AC-MOVE-68),
        which may well choose a later interception at a slower object's
        position. And if no interception exists at all, the visit is abandoned
        with its reason recorded and the object recorded as missed.

        Args:
            queue: The order selection produced, for the track being served.
            flange: Where the flange stands. Kept for the call site; a mid-
                mid-visit re-solve does not start from it.
            velocity: How fast the reference was moving when the caller last
                measured it. Kept for the call site; a mid-visit re-solve
                takes velocity from the plan sample instead.
            at_seconds: Simulated time.
        """
        del flange, velocity
        # The hold is the jaw already shut, and moving the aim then drags a
        # closed jaw across the belt. The descent is not: it is the last
        # fraction of a second, the object is still moving, and a correction
        # frozen at the start of it is a jaw that arrives where the object was.
        if self._phase not in (Phase.TRACK, Phase.DESCEND):
            return
        assert self._plan is not None
        assert self._motion is not None
        track_id = self._plan.track_id
        head = next(
            (item for item in queue.order if item.track_id == track_id),
            None,
        )
        if head is None:
            # Whatever this visit was about is no longer a candidate: it has
            # left the window, or the slot it rode in now holds something else.
            # There is no pose left to aim at, and a plan active at the last one
            # it saw is a plan descending onto bare belt.
            self._abandon(
                track_id,
                at_seconds,
                "the object is no longer a candidate",
                drift=None,
                refused=True,
            )
            return
        target = self._grasp_pose(head)
        if self._phase is Phase.DESCEND:
            # The approach is over. Bending it is what refine does, and once
            # the track legs have finished it answers None, which the rest of
            # this method reads as a refusal and may abandon the visit from.
            # A jaw that is already on the way down is a fraction of a second
            # from the object, and the answer to a correction that will not
            # fit is the plan already in hand.
            self._steer_descent(head, target, at_seconds)
            return
        aim = _aim_of(self._plan)
        drift = distance(self._fresh_aim(head, target, at_seconds), aim)
        refreshed = self._refinement(head, target, at_seconds)
        landed = (
            refreshed is not None
            and distance(self._fresh_aim(head, target, at_seconds), _aim_of(refreshed))
            <= self._settings.aim_tolerance
        )
        if (
            refreshed is not None
            and landed
            and self._takes(_aim_of(refreshed))
            and self._within_reach(refreshed)
        ):
            turned = self._yaw_at_pick(head, refreshed, at_seconds)
            self._refreshes.append(Reaim(at_seconds, track_id, drift, False, "took"))
            self._narrate_reaim(
                at_seconds,
                track_id,
                "took",
                f"the fresh estimate stands {drift * 1000:.0f} mm from the aim "
                "and the approach arc still fits",
                f"bend the approach and keep fetching {self._story.refer(track_id)}",
            )
            self._plan = refreshed.with_yaw(
                target_yaw=turned,
                initial_yaw=self._last_yaw,
            )
            self._plan_yaw = turned
            return
        if drift <= self._settings.aim_tolerance:
            # The freshest estimate is close enough to where the arm is already
            # going that there is nothing to correct, so the refusal costs
            # nothing and the visit flies as planned.
            self._refreshes.append(Reaim(at_seconds, track_id, drift, True, "held"))
            self._narrate_reaim(
                at_seconds,
                track_id,
                "held",
                f"the fresh estimate is {drift * 1000:.0f} mm from the aim, "
                "inside the tolerance",
                f"keep the plan I already have for {self._story.refer(track_id)}",
            )
            return
        # Re-solve from the motion reference, not the measured flange. The plan
        # advances under perfect tracking in the flat output; lag and collisions
        # are a control problem. Starting the new visit at the plant writes that
        # disturbance into the next projected trajectory (AC-MOVE-68).
        sampled = self._plan.at(at_seconds)
        if sampled is None:
            self._abandon(
                track_id,
                at_seconds,
                f"no interception from an aim {drift * 1000:.0f} mm out",
                drift=drift,
                refused=True,
            )
            return
        _, guided, _ = sampled
        solved = self._resolve(head, guided.position, guided.velocity, at_seconds)
        if solved is None:
            self._abandon(
                track_id,
                at_seconds,
                f"no interception from an aim {drift * 1000:.0f} mm out",
                drift=drift,
                refused=True,
            )
            return
        self._refreshes.append(Reaim(at_seconds, track_id, drift, True, "solved"))
        self._narrate_reaim(
            at_seconds,
            track_id,
            "solved",
            f"the fresh estimate is {drift * 1000:.0f} mm from the aim and the "
            "arc I had was refused",
            f"solve the visit again ({solved.duration:.3f} s) and keep fetching "
            f"{self._story.refer(track_id)}",
        )
        turned = self._yaw_at_pick(head, solved, at_seconds)
        self._plan = solved.with_yaw(
            target_yaw=turned,
            initial_yaw=self._last_yaw,
        )
        self._plan_yaw = turned
        LOGGER.debug(
            "re-solved pick visit for candidate %d: aim was %.0f mm out, new "
            "plan is %.3f s",
            track_id,
            drift * 1000,
            solved.duration,
        )

    def _steer_descent(
        self,
        head: Candidate,
        target: NDArray[np.float64],
        at_seconds: float,
    ) -> None:
        """Move the end of a descent onto the freshest estimate, or leave it.

        The approach is over, so there is no arc left to bend, and solving the
        visit again would send the arm back up the belt. The jaw has not shut,
        so the object is still moving and a descent frozen at its start arrives
        where the object was. A correction that breaks a ceiling, or that the
        arm is not trusted over, leaves the plan already in hand: abandoning
        from here drops an object the jaw is a fraction of a second from.

        Args:
            head: The candidate being served.
            target: The fresh grasp pose, in world frame meters.
            at_seconds: Simulated time.
        """
        assert self._plan is not None
        assert self._motion is not None
        transit_height_world = self._safe_height_world
        retreat_lift = max(
            self._settings.grasp_clearance, transit_height_world - target[2]
        )
        chute = self._chutes.get(head.channel)
        over = (
            np.asarray(
                (chute[0], chute[1], transit_height_world),
                dtype=np.float64,
            )
            if chute is not None
            else None
        )
        steered = retarget_descent(
            plan=self._plan,
            object_position=target,
            belt_velocity=self._object_velocity(head),
            approach_clearance_z=self._settings.grasp_clearance,
            approach_speed=self._settings.approach_speed,
            dwell_seconds=self._settings.dwell_seconds,
            max_speed=self._motion.max_speed,
            max_acceleration=self._motion.max_acceleration,
            at_seconds=at_seconds,
            drift_horizon=self._settings.drift_horizon,
            minimum_segment_seconds=self._motion.minimum_segment_seconds,
            closing_rise_seconds=self._settings.closing_rise_seconds,
            retreat_lift=retreat_lift,
            over=over,
            belt_border_y=self._belt_border_y,
            safe_height_world=transit_height_world,
            cross_speed=self._settings.approach_speed,
            clearance_flange_z=self._shut_flange_floor_world,
            segment_sample_count=self._motion.segment_sample_count,
            bisection_passes=self._motion.bisection_passes,
            minimum_delivery_seconds=self._motion.minimum_delivery_seconds,
            correction_steps=self._motion.correction_steps,
        )
        if (
            steered is None
            or not self._takes(_aim_of(steered))
            or not self._within_reach(steered)
        ):
            return
        drift = distance(self._fresh_aim(head, target, at_seconds), _aim_of(self._plan))
        turned = self._yaw_at_pick(head, steered, at_seconds)
        self._refreshes.append(Reaim(at_seconds, head.track_id, drift, False, "took"))
        self._narrate_reaim(
            at_seconds,
            head.track_id,
            "descent",
            f"the descent is underway and the fresh estimate stands "
            f"{drift * 1000:.0f} mm from the aim",
            f"move the end of the descent onto {self._story.refer(head.track_id)}",
        )
        self._plan = steered.with_yaw(
            target_yaw=turned,
            initial_yaw=self._last_yaw,
        )
        self._plan_yaw = turned

    def _yaw_at_pick(
        self, head: Candidate, plan: Plan, at_seconds: float
    ) -> float | None:
        """Return the tool yaw to command for a grasp the plan will make later.

        The marker's closing yaw is a claim made when the capture was settled,
        and the jaws close when the plan says they will -- up to half a second
        later, and longer when the visit was solved further ahead. An object on
        this belt turns while that time passes, so commanding the claimed yaw
        sends the tool to where the object was facing rather than where it will
        be facing. Measured over nine grabs: up to 5.4 radians per second, which
        is 155 degrees of staleness the tool arrived carrying, and 22 to 40
        degrees of error against the object's own axis at the instant the jaws
        shut.

        So the yaw is carried forward the same way the position is, with the one
        number that differs: this is a sum and not a scalar, so it wraps.

        Args:
            head: The candidate being served.
            plan: The plan whose arrival time fixes the instant.
            at_seconds: Simulated time.

        Returns:
            The yaw to command, in radians, or None when the marker claimed no
            axis to turn the tool to.
        """
        if head.closing_yaw_belt is None:
            return None
        if head.yaw_rate_belt is None:
            return head.closing_yaw_belt
        # A prediction is only worth making while it is a prediction. A spin of
        # 5.4 radians per second carried over a four second plan is twenty
        # radians, which is not an estimate of where the object will be facing
        # but a number drawn from a direction that has wrapped many times, and
        # commanding it swings the wrist across the belt: measured, the run that
        # did that without a bound put the jaw 106.6 mm inside the belt over 67
        # ticks. A jaw is symmetric about 90 degrees, so 45 is the furthest a
        # turn can be worth taking: past it the claim is no better than the
        # guess, and the claim is what a caller who has no rate at all gets.
        turn = head.yaw_rate_belt * (plan.pick_at - at_seconds)
        if abs(turn) > math.pi / 4.0:
            return head.closing_yaw_belt
        return (head.closing_yaw_belt + turn + math.pi) % (2.0 * math.pi) - math.pi

    def _within_reach(self, plan: Plan) -> bool:
        """Whether every pose along a plan's path is one the arm is trusted over.

        The descent's endpoint was checked and the path to it was not, and the
        path is where the failure lived: the carry rides with the object for the
        whole hold, and an object drifting across the belt takes the command
        with it. Measured on the run that prompted this, two visits in nine
        commanded poses 208 and 346 mm from any object and outside the region
        the arm is trusted over, the servo projected them back on the way to the
        joints, and the arm fell **50 to 80 mm behind its own command while the
        jaws closed on nothing**.

        Sampled rather than solved: the region is an annulus and a path between
        two admissible poses can cross the hole, so the question is what the
        worst point on each leg says. Five points per leg costs a handful of
        comparisons per capture and cannot miss a leg that leaves by more than
        the sample spacing.

        Args:
            plan: The plan to check.

        Returns:
            Whether every sampled pose is inside the region.
        """
        # Five poses per leg, because the region test is a call into the world
        # geometry and not an arithmetic expression: what this can miss is a leg
        # that leaves the annulus and returns within one spacing, and that is
        # the resolution the claim is made at. `np.linspace` rather than a
        # literal step so the count and the fractions cannot drift apart.
        fractions = np.linspace(0.0, 1.0, self._settings.leg_samples)
        for leg in plan.legs:
            for fraction in fractions:
                at = leg.segment.duration * float(fraction)
                if not self._admits(leg.segment.at(at).position):
                    return False
        return True

    def _object_velocity(self, head: Candidate) -> NDArray[np.float64]:
        """Return how the belt is carrying this candidate, with no vertical guess.

        One place, because three callers predict the object forward with it --
        the initial solve, the re-solve and the fresh aim the drift is measured
        against -- and a second copy of this clamp is how they come to disagree
        about which object they are aiming at.

        Args:
            head: The candidate.

        Returns:
            Its velocity in world frame, with travel up the belt floored at
            zero: a jaw cannot be aimed at an object that is behind where the
            plan already committed to meet it.
        """
        obj_vel = (
            head.velocity_world
            if head.velocity_world is not None
            else np.asarray((self._belt_speed, 0.0, 0.0), dtype=np.float64)
        )
        # A floor on one component only: `max` names that, and `np.clip` over a
        # per-component bound would hide which axis carries it.
        return np.asarray(
            (max(0.0, float(obj_vel[0])), float(obj_vel[1]), float(obj_vel[2])),
            dtype=np.float64,
        )

    def _fresh_aim(
        self,
        head: Candidate,
        target: NDArray[np.float64],
        at_seconds: float,
    ) -> NDArray[np.float64]:
        """Return where the freshest estimate puts the object at the pick instant.

        The same model the arcs are solved against, evaluated from this
        capture's estimate instead of the one the plan was committed on, at the
        arrival time the plan already fixed. The distance between this and what
        the plan is actually aiming at is the whole question a re-aim answers,
        and the reason a refused refinement must not be ignored: an object
        dragging on the belt moves this number by hundreds of millimetres while
        the plan's own arrival error stays at two.

        Args:
            head: The candidate being served.
            target: The fresh grasp pose, in world frame meters.
            at_seconds: Simulated time.

        Returns:
            The pose the jaws would meet the object at, in world frame meters.
        """
        assert self._plan is not None
        return where_carried(
            target,
            self._object_velocity(head),
            self._plan.pick_at - at_seconds,
            0.0,
            self._settings.drift_horizon,
        )

    def _refinement(
        self, head: Candidate, target: NDArray[np.float64], at_seconds: float
    ) -> Plan | None:
        """Return the active plan rebuilt around a fresh aim, or None.

        The same call [clave.control.pick.refine] has always been given, split
        out so the caller can tell a refinement that came back from one that
        did not. None is the ordinary answer for an object that has fallen
        behind the plan, and the caller has to know which it got.

        Args:
            head: The candidate being served.
            target: The fresh grasp pose, in world frame meters.
            at_seconds: Simulated time.

        Returns:
            The re-aimed plan, or None when there is nothing left to re-aim or
            the re-aimed arc breaks a ceiling.
        """
        assert self._plan is not None
        assert self._motion is not None
        transit_height_world = self._safe_height_world
        retreat_lift = max(
            self._settings.grasp_clearance, transit_height_world - target[2]
        )
        chute = self._chutes.get(head.channel)
        over = (
            np.asarray(
                (chute[0], chute[1], transit_height_world),
                dtype=np.float64,
            )
            if chute is not None
            else None
        )
        return refine(
            plan=self._plan,
            object_position=target,
            belt_velocity=self._object_velocity(head),
            z_offset=self._settings.grasp_clearance,
            approach_speed=self._settings.approach_speed,
            dwell_seconds=self._settings.dwell_seconds,
            max_speed=self._motion.max_speed,
            max_acceleration=self._motion.max_acceleration,
            at_seconds=at_seconds,
            over=over,
            retreat_lift=retreat_lift,
            belt_border_y=self._belt_border_y,
            safe_height_world=transit_height_world,
            cross_speed=self._settings.approach_speed,
            drift_horizon=self._settings.drift_horizon,
            minimum_segment_seconds=self._motion.minimum_segment_seconds,
            closing_rise_seconds=self._settings.closing_rise_seconds,
            clearance_flange_z=self._shut_flange_floor_world,
            segment_sample_count=self._motion.segment_sample_count,
            bisection_passes=self._motion.bisection_passes,
            minimum_delivery_seconds=self._motion.minimum_delivery_seconds,
            correction_steps=self._motion.correction_steps,
        )

    def _abandon(
        self,
        track_id: int,
        at_seconds: float,
        reason: str,
        drift: float | None,
        refused: bool,
    ) -> None:
        """Give up a visit whose aim the freshest estimate has contradicted.

        The plan is discarded rather than flown, because the one thing known
        about it is that it arrives where the object is not. The arm is left
        where it is with the jaw open -- a caller stops being handed a visit tick
        and holds position -- and the object is recorded as missed, so the next
        capture reaches for something else instead of re-planning the same
        hopeless visit forever.

        Args:
            track_id: The track the visit was about.
            at_seconds: When it was given up.
            reason: Why, in words, for the report.
            drift: How far out the aim was, or None when there was nothing left
                to measure it against.
            refused: Whether the solver had refused a correction.
        """
        LOGGER.warning(
            "abandoned pick visit for candidate %d at %.3f s: %s",
            track_id,
            at_seconds,
            reason,
        )
        self._story.tell(
            at_seconds,
            reason,
            f"abandon {self._story.refer(track_id)} and look for the next object",
            "fail",
            channel=self._story.channel_of(track_id),
        )
        self._refreshes.append(Reaim(at_seconds, track_id, drift, refused, "abandoned"))
        self._abandoned.append((track_id, reason))
        if track_id not in self._missed:
            self._missed.append(track_id)
        self._plan, self._plan_yaw, self._pick_error = None, None, None
        self._serving, self._arrived_at = None, None
        self._told_reaim = None
        self._phase = Phase.STANDBY

    def _commit(
        self,
        head: Candidate,
        flange: NDArray[np.float64],
        velocity: NDArray[np.float64],
        at_seconds: float,
        at_nanos: int,
    ) -> Goal:
        """Plan a whole visit to a candidate, or give it up.

        Args:
            head: The candidate to serve.
            flange: Where the flange stands.
            velocity: How fast the reference is moving, so the first arc
                begins where the motion already is.
            at_seconds: Simulated time the plan starts.
            at_nanos: The same instant, for the goal it returns.

        Returns:
            A goal naming the approach point, or the park pose when no
            interception exists. An object the arm cannot reach in the belt
            it has left is recorded as missed and skipped, because chasing it
            costs the objects behind it.
        """
        # The constructor refused this profile without ceilings to plan under.
        assert self._motion is not None
        if not self._admits(flange):
            # A plan begins where the arm is, so an arm outside the region it
            # is trusted over produces a plan whose very first pose the servo
            # refuses, and the refusal tears up the plan that caused it, and
            # the next capture builds the same one again. Measured on the
            # shipped line that latched: one excursion became 46 faults and
            # the arm never moved again.
            #
            # Going home is the recovery, because the park pose is inside the
            # region by construction and the servo will take it. The
            # candidate is not recorded as missed: nothing is wrong with it.
            self._story.note(head.track_id, "", head.channel)
            self._story.tell(
                at_seconds,
                "the flange is outside the region the arm is trusted over",
                f"go to the park pose instead of committing to "
                f"{self._story.refer(head.track_id)}",
                "pending",
                channel=head.channel,
            )
            self._serving = None
            return self._rest(flange, at_nanos)
        plan = self._resolve(head, flange, velocity, at_seconds)
        if plan is None:
            self._missed.append(head.track_id)
            self._story.note(head.track_id, "", head.channel)
            self._story.tell(
                at_seconds,
                self._plan_refusal or "no interception exists",
                f"skip {self._story.refer(head.track_id)} rather than chase it",
                "fail",
                channel=head.channel,
            )
            self._serving = None
            return self._rest(flange, at_nanos)
        target_yaw = self._yaw_at_pick(head, plan, at_seconds)
        initial_yaw = self._last_yaw if self._last_yaw is not None else target_yaw
        if target_yaw is not None and initial_yaw is not None:
            target_yaw = nearer_parallel_yaw(target_yaw, initial_yaw)

        self._plan = plan.with_yaw(target_yaw=target_yaw, initial_yaw=initial_yaw)
        self._plan_yaw = target_yaw
        self._serving = head.track_id
        self._pick_error = None
        self._told_reaim = None
        self._phase = Phase.TRACK
        self._story.note(head.track_id, "", head.channel)
        self._story.tell(
            at_seconds,
            "the queue offered this object and an interception fits the belt "
            "it has left",
            f"fly a {plan.duration:.3f} s visit ({_visit_manner(plan)}) to fetch "
            f"{self._story.refer(head.track_id)}",
            "pending",
            channel=head.channel,
        )
        LOGGER.debug(
            "committed pick plan for candidate %d: duration=%.3f s, legs=%d, chute=%s",
            head.track_id,
            plan.duration,
            len(plan.legs),
            head.channel,
        )
        return Goal(
            phase=Phase.TRACK,
            rides_belt=True,
            target_position_world=plan.legs[0].segment.end.position,
            target_yaw_world=target_yaw,
            track_id=head.track_id,
            observed_at_nanos=at_nanos,
        )

    def _resolve(
        self,
        head: Candidate,
        flange: NDArray[np.float64],
        velocity: NDArray[np.float64],
        at_seconds: float,
    ) -> Plan | None:
        """Solve one whole visit for a candidate, or report that none exists.

        Split out of the commit path because a visit is solved more than once:
        first when the arm commits to a candidate, and again when the freshest
        estimate says the object has moved off the pose that first solve was
        aiming at by more than the tolerance. Both callers want the same answer
        from the same numbers, and a second copy of this search would be a
        second place for the never-fly-a-stale-plan rule to be forgotten.

        Args:
            head: The candidate to serve.
            flange: Where the first arc of the plan begins. On the first
                commit this is the measured flange; on a mid-visit re-solve
                it is the motion reference sampled from the plan already in
                hand (AC-MOVE-68).
            velocity: How fast that start is moving.
            at_seconds: Simulated time the plan starts.

        Returns:
            The whole visit, or None when no interception inside what the
            object has left on the belt respects both ceilings, or when every
            one that does asks for a pose the jaw may not take. None is the
            honest answer for an object the arm cannot reach in the belt it has
            left.
        """
        assert self._motion is not None
        target = self._grasp_pose(head)
        transit_height_world = self._safe_height_world
        retreat_lift = max(
            self._settings.grasp_clearance, transit_height_world - target[2]
        )
        chute = self._chutes.get(head.channel)
        over = (
            np.asarray(
                (chute[0], chute[1], transit_height_world),
                dtype=np.float64,
            )
            if chute is not None
            else None
        )
        # An interception is worth planning only inside what the object has
        # left on the belt, whichever of the two limits binds first.
        clamped_vel = self._object_velocity(head)
        speed_x = max(0.01, clamped_vel[0])
        leaving = head.distance_before_leaving / speed_x
        # The margin buys room to re-aim and it buys it by intercepting
        # further downstream, which can put the pick past the edge of the
        # annulus. Where it does, the soonest interception is taken instead:
        # an arc that cannot be corrected still beats one the arm cannot fly.
        plan = None
        jaw_in_the_belt: float | None = None
        for margin in (self._settings.interception_margin, 1.0):
            attempt = plan_pick(
                flange=State(position=flange, velocity=velocity, acceleration=zeros()),
                track_id=head.track_id,
                object_position=target,
                belt_velocity=clamped_vel,
                z_offset=self._settings.grasp_clearance,
                approach_speed=self._settings.approach_speed,
                dwell_seconds=self._settings.dwell_seconds,
                max_speed=self._motion.max_speed,
                max_acceleration=self._motion.max_acceleration,
                latest=min(self._settings.interception_limit, leaving),
                at_seconds=at_seconds,
                margin=margin,
                over=over,
                retreat_lift=retreat_lift,
                belt_border_y=self._belt_border_y,
                safe_height_world=transit_height_world,
                cross_speed=self._settings.approach_speed,
                drift_horizon=self._settings.drift_horizon,
                minimum_segment_seconds=self._motion.minimum_segment_seconds,
                closing_rise_seconds=self._settings.closing_rise_seconds,
                clearance_flange_z=self._shut_flange_floor_world,
                segment_sample_count=self._motion.segment_sample_count,
                bisection_passes=self._motion.bisection_passes,
                minimum_delivery_seconds=self._motion.minimum_delivery_seconds,
                correction_steps=self._motion.correction_steps,
            )
            if attempt is not None:
                descent_leg = next(
                    leg for leg in attempt.legs if leg.phase is Phase.DESCEND
                )
                if self._takes(descent_leg.segment.end.position) and self._within_reach(
                    attempt
                ):
                    plan = attempt
                    break
                if descent_leg.segment.end.position[2] < self._grasp_floor_world:
                    jaw_in_the_belt = descent_leg.segment.end.position[2]
        if plan is None:
            if jaw_in_the_belt is not None:
                self._plan_refusal = (
                    "the grasp pose would leave the jaw inside the belt "
                    f"(lowest geometry at z={jaw_in_the_belt:.4f} m, "
                    f"floor {self._grasp_floor_world:.4f} m)"
                )
                LOGGER.debug(
                    "pick planning refused candidate %d: its grasp pose leaves "
                    "the jaw's lowest geometry inside the %.3f m the jaw keeps "
                    "above the belt at z=%.4f m (floor %.4f m)",
                    head.track_id,
                    self._grasp_floor_world - self._jaw_below_flange_world,
                    jaw_in_the_belt,
                    self._grasp_floor_world,
                )
            else:
                self._plan_refusal = (
                    f"no interception exists in the {leaving:.3f} s of belt it has left"
                )
                LOGGER.debug(
                    "pick planning failed for candidate %d (leaving=%.3f s); "
                    "marked missed",
                    head.track_id,
                    leaving,
                )
        else:
            self._plan_refusal = ""
        return plan

    def _held(self, at_nanos: int) -> Goal:
        """Report the visit already active, without re-deciding it.

        Args:
            at_nanos: Now.

        Returns:
            A goal naming the phase and the pose the plan is currently
            asking for. It exists so a caller can report what the arm is
            doing; nothing consumes it to steer, because the plan does that.
        """
        assert self._plan is not None
        elapsed = max(0.0, at_nanos / NANOS_PER_SECOND - self._plan.started_at)
        sampled = self._plan.at(self._plan.started_at + elapsed)
        position = (
            self._plan.legs[-1].segment.end.position
            if sampled is None
            else sampled[1].position
        )
        return Goal(
            phase=self._phase,
            rides_belt=True,
            target_position_world=position,
            target_yaw_world=self._plan_yaw,
            track_id=self._plan.track_id,
            observed_at_nanos=at_nanos,
        )

    def _finish(self, flange: NDArray[np.float64], at_seconds: float) -> None:
        """Close a planned visit that has run out of arcs.

        Args:
            flange: Where the flange stands, used only if the hold was
                shorter than a tick and so was never sampled. That cannot
                happen at the shipped dwell and tick, and falling back beats
                recording a figure nobody measured.
            at_seconds: When the plan ran out. The lift that says whether the
                jaw kept the object is measured by the caller, so this line
                is pending rather than a verdict.
        """
        assert self._plan is not None
        last = self._plan.legs[-1].segment.end.position
        error = (
            self._pick_error if self._pick_error is not None else distance(flange, last)
        )
        self._arrivals.append(error)
        self._served.append(self._plan.track_id)
        self._story.tell(
            at_seconds,
            f"the plan ran out with the flange {error * 1000:.1f} mm from the "
            "pose it was aiming at",
            f"open the jaw and wait for the lift to say whether "
            f"{self._story.refer(self._plan.track_id)} was held",
            "pending",
            channel=self._story.channel_of(self._plan.track_id),
        )
        LOGGER.debug(
            "completed pick visit for candidate %d: pick_error=%.4f m",
            self._plan.track_id,
            error,
        )
        self._plan = None
        self._plan_yaw = None
        self._pick_error = None
        self._serving = None
        self._arrived_at = None
        self._told_reaim = None
        self._phase = Phase.STANDBY

    def _next(self, queue: Queue) -> Candidate | None:
        """Return the first candidate worth serving, or None.

        Args:
            queue: The order selection produced.

        Returns:
            The head, skipping anything already served, faulted or missed. A
            pose the solver refused will be refused again, so retrying it
            forever is an arm that stops working on the first bad pose, and
            an object no interception existed for only gets further away.
        """
        refused = {track_id for track_id, _ in self._faults if track_id is not None}
        done = set(self._served) | refused | set(self._missed)
        return next((item for item in queue.order if item.track_id not in done), None)

    def _track(self, head: Candidate, at_nanos: int, phase: Phase) -> Goal:
        """Return the goal one phase of a visit asks for.

        Args:
            head: The candidate being served.
            at_nanos: When the candidate's pose was seen.
            phase: Which phase of the visit this is.

        Returns:
            The goal, carrying the calibration offset. Only the height
            changes between phases: the descent follows the object along the
            belt exactly as the approach did, because the belt does not stop
            while the flange comes down.
        """
        offset = self._calibration.flange_offset
        above = (
            head.flange_position_world[2] - self._belt_surface_height_world
            if phase is Phase.DESCEND
            else self._settings.approach_height
        )
        return Goal(
            phase=phase,
            rides_belt=True,
            target_position_world=(
                np.asarray(
                    [
                        head.flange_position_world[0],
                        head.flange_position_world[1],
                        self._belt_surface_height_world + above,
                    ],
                    dtype=np.float64,
                )
                + offset
            ),
            target_yaw_world=head.closing_yaw_belt,
            track_id=head.track_id,
            observed_at_nanos=at_nanos,
        )

    def _rest(self, flange: NDArray[np.float64], at_nanos: int) -> Goal:
        """Return the goal for an arm with nothing to serve.

        Args:
            flange: Where the flange stands.
            at_nanos: Now.

        Returns:
            The park pose, under the phase that says whether the arm is still
            travelling to it.
        """
        park = self._settings.park_position_world
        arrived = distance(flange, park) <= self._settings.arrival_tolerance
        return Goal(
            phase=Phase.STANDBY if arrived else Phase.PARK,
            target_position_world=park,
            target_yaw_world=self._last_yaw,
            track_id=None,
            observed_at_nanos=at_nanos,
        )

    def _fault(
        self,
        queue: Queue,
        flange: NDArray[np.float64],
        refusal: str,
        at_nanos: int,
    ) -> Goal:
        """Record a refusal and hold the arm where it is.

        A refusal is about the pose commanded on the previous tick, so the
        track it belongs to is normally the one being served. On the first
        tick of a run there is no such track, and the refusal is attributed
        to the head the machine would have served instead. Only when nothing
        is queued either does it go down against no track, which is still
        recorded: a refusal nobody counted is the one way the arm can stop
        working and leave no trace of why.

        Args:
            queue: The order selection produced, read only for its head.
            flange: Where the flange stands.
            refusal: Why the pose was refused.
            at_nanos: Now.

        Returns:
            A goal asking for no motion, so a refused pose moves nothing.
        """
        self._plan, self._plan_yaw, self._pick_error = None, None, None
        faulted = self._serving
        if faulted is None:
            head = self._next(queue)
            faulted = None if head is None else head.track_id
        LOGGER.warning(
            "task machine faulted: refusal=%s (candidate=%s)",
            refusal,
            faulted,
        )
        who = self._story.refer(faulted)
        self._story.tell(
            at_nanos / NANOS_PER_SECOND,
            f"the last commanded pose was refused ({refusal})",
            f"hold the arm where it is and drop {who}",
            "fail",
            channel=self._story.channel_of(faulted),
        )
        already = {track_id for track_id, _ in self._faults}
        if faulted is None or faulted not in already:
            self._faults.append((faulted, refusal))
        self._serving, self._arrived_at = None, None
        self._phase = Phase.STANDBY
        return Goal(
            phase=Phase.FAULT,
            target_position_world=flange,
            target_yaw_world=None,
            track_id=faulted,
            observed_at_nanos=at_nanos,
        )


def _visit_manner(plan: Plan) -> str:
    """Return the phases of a plan, once each, in the order they are flown.

    Args:
        plan: The visit.

    Returns:
        A comma-separated list of phase names. A delivery that is two arcs
        is named once, because the manner is the phase and not the arc.
    """
    names: list[str] = []
    for leg in plan.legs:
        name = leg.phase.value
        if not names or names[-1] != name:
            names.append(name)
    return ", ".join(names)


def _anywhere(pose: NDArray[np.float64]) -> bool:
    """Admit any pose, for a caller that supplied no region.

    Args:
        pose: The pose.

    Returns:
        True.
    """
    del pose
    return True
