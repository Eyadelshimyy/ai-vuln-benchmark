"""Estimating where an object is going, rather than assuming it.

The tracker used to carry a position and nothing else. A new detection
replaced it outright, and between detections the estimate was carried along
the belt at the configured belt speed, in the travel axis only. That model
makes two claims: that every object travels at exactly the belt's speed,
and that lateral drift is exactly zero.

The first is very nearly true and the second is false. Measured on the
shipped line, an object rolling or settling drifts sideways by a mean of
10 mm over half a second and 41 mm over 2.5 seconds, against a jaw whose
narrowest side clearance is 8.7 mm. Downstream of the sensing gate nothing
observes it again, so that drift is the error the pick inherits.

**An alpha-beta filter, which is the steady-state Kalman filter for a
constant-velocity model.** Two gains per axis instead of a covariance to
propagate: `alpha` is how much of the residual corrects the position and
`beta` how much corrects the velocity. Writing it this way keeps the tuning
legible, since each gain is a number between zero and one with a meaning a
reader can hold, and it costs nothing a four-state covariance would buy on
a problem with no cross-axis coupling.

**The two axes are not symmetric and are not tuned as if they were.** The
belt drives travel rigidly, so the velocity there is known before any
observation arrives and the filter should barely move it. Across the belt
nothing drives anything, so the velocity there has to be learned entirely
from successive observations. One pair of gains for both axes would either
refuse to learn the drift or let belt-speed noise wander the travel
estimate.

**What this does not do.** The drift it learns is a constant velocity, and
the drift it is learning from is not: the increments measured over growing
horizons fall away, because an object that is rolling is also settling. So
an estimate fitted inside the gate and extrapolated two seconds past it
over-predicts. Better than assuming zero is a low bar, and clearing it is
not the same as being right.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Gains:
    """How hard a filter corrects toward what it just saw.

    Attributes:
        alpha: Fraction of the position residual applied to the position.
            One replaces the estimate with the observation, which is what
            the tracker did before this existed; zero ignores the
            observation entirely.
        beta: Fraction of the residual, divided by the elapsed time, applied
            to the velocity. Zero freezes the velocity at whatever it was
            initialised to, which for the travel axis is the belt speed.
    """

    alpha: float
    beta: float


@dataclass(frozen=True)
class Estimate:
    """Where one object is and how fast it is going.

    Attributes:
        position: In belt frame meters, along travel and across it.
        velocity: In meters per second, the same way round.
        at_seconds: The instant the estimate describes.
    """

    position: tuple[float, float]
    velocity: tuple[float, float]
    at_seconds: float

    def at(self, at_seconds: float) -> tuple[float, float]:
        """Return where the object is predicted to be.

        Args:
            at_seconds: The instant wanted. May precede the estimate, in
                which case the object is carried upstream.

        Returns:
            The predicted position.
        """
        elapsed = at_seconds - self.at_seconds
        return (
            self.position[0] + self.velocity[0] * elapsed,
            self.position[1] + self.velocity[1] * elapsed,
        )


def begin(
    position: tuple[float, float], belt_speed: float, at_seconds: float
) -> Estimate:
    """Open an estimate on a first observation.

    The travel velocity starts at the belt's, because the belt drives that
    axis and pretending otherwise would spend the first several observations
    rediscovering a number the configuration already states. The lateral
    velocity starts at zero, because nothing drives that axis and zero is
    the honest prior rather than a claim.

    Args:
        position: Where it was seen, along travel and across it.
        belt_speed: Belt speed in meters per second.
        at_seconds: When it was seen.

    Returns:
        The estimate.
    """
    return Estimate(
        position=position, velocity=(belt_speed, 0.0), at_seconds=at_seconds
    )


def fold(
    estimate: Estimate,
    position: tuple[float, float],
    at_seconds: float,
    along: Gains,
    across: Gains,
) -> Estimate:
    """Correct an estimate toward one new observation.

    Args:
        estimate: What was believed before.
        position: Where the object was just seen.
        at_seconds: When it was seen.
        along: Gains for the belt travel axis.
        across: Gains for the lateral axis.

    Returns:
        The corrected estimate. An observation at or before the instant the
        estimate already describes is folded at position only, because a
        velocity correction divides by the elapsed time and two readings at
        one instant say nothing about a velocity.
    """
    elapsed = at_seconds - estimate.at_seconds
    predicted = estimate.at(at_seconds)
    corrected: list[float] = []
    moved: list[float] = []
    for axis, gains in ((0, along), (1, across)):
        residual = position[axis] - predicted[axis]
        corrected.append(predicted[axis] + gains.alpha * residual)
        moved.append(
            estimate.velocity[axis] + gains.beta * residual / elapsed
            if elapsed > 0.0
            else estimate.velocity[axis]
        )
    return replace(
        estimate,
        position=(corrected[0], corrected[1]),
        velocity=(moved[0], moved[1]),
        at_seconds=max(at_seconds, estimate.at_seconds),
    )
