"""Holding the line at a feed rate by trimming belt speed.

A line whose throughput is whatever the spawn timer happened to produce
cannot be reasoned about. This gives it a setpoint in objects per second, a
measurement of what it is achieving, and a proportional-integral controller
between the two. `docs/requirements/line-throughput.md` carries the spec.

**Belt speed only regulates a rate if the feed is by distance.** With
objects released every `T` seconds the arrival rate is `1/T` whatever the
belt does: speeding up spreads the same objects over more metres. With
objects released every `s` metres of belt travel, which is what a metering
feeder does on a real line, the rate is `v / s` and speed is the throughput
knob directly. That is why [clave.world.belt] feeds by spacing.

The loop is not trivial even so, because `s` is drawn per object from a
range. The realised rate wanders around `v` over the mean spacing, and this
trims `v` to hold it.

**The measurement reads the simulator.** Counting arrivals is ground truth
and says so here rather than quietly: the tracker's own estimate is
unusable downstream of the sensing gate, so a loop built on it would
regulate the tracker's opinion rather than the line.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from clave.world.config import WorldConfigError, require


@dataclass(frozen=True)
class FeedSettings:
    """What the feed controller reads.

    Attributes:
        rate: Objects per second the line is asked to carry.
        window_seconds: How long a span the measured rate is counted over.
        proportional_gain: Meters per second of belt speed per object per
            second of error. A positional gain: the term is added to the
            speed the run started at, not to the speed commanded last tick.
        integral_gain: Meters per second per object, since it multiplies an
            error integrated over time.
    """

    rate: float
    window_seconds: float
    proportional_gain: float
    integral_gain: float

    @classmethod
    def load(cls, raw: dict[str, Any]) -> FeedSettings:
        """Read the feed block of a world configuration.

        Args:
            raw: The parsed world configuration.

        Returns:
            The settings it describes.

        Raises:
            WorldConfigError: If the block or a key is absent, or if the rate
                or the window is not above zero. A rate at or below zero
                describes a line that carries nothing.
        """
        block = require(raw, "feed")
        return cls(
            rate=_positive(block, "rate_objects_per_second"),
            window_seconds=_positive(block, "window_seconds"),
            proportional_gain=float(require(block, "proportional_gain", "feed")),
            integral_gain=float(require(block, "integral_gain", "feed")),
        )


@dataclass
class RateController:
    """Trims belt speed to hold the line at its feed rate.

    Attributes:
        settings: The setpoint, the window and the gains.
        lowest: Slowest the belt drive runs, in meters per second.
        highest: Fastest it runs.
        speed: What the belt is commanded to now. Seeded with the speed the
            run drew, which also becomes the operating point the
            proportional term is measured from.
    """

    settings: FeedSettings
    lowest: float
    highest: float
    speed: float
    _base: float | None = None
    _arrivals: deque[float] = field(default_factory=deque)
    _accumulated: float = 0.0
    _measured: float = 0.0

    @property
    def measured(self) -> float:
        """The rate the line is achieving, in objects per second."""
        return self._measured

    @property
    def saturated(self) -> bool:
        """Whether the commanded speed is sitting on one of its limits."""
        return self.speed <= self.lowest or self.speed >= self.highest

    def arrived(self, at_seconds: float) -> None:
        """Record one object entering the line.

        Args:
            at_seconds: When it entered.
        """
        self._arrivals.append(at_seconds)

    def update(self, at_seconds: float, timestep: float) -> float:
        """Advance the loop one tick and return the speed to command.

        Args:
            at_seconds: Simulated time.
            timestep: How long this tick lasts, in seconds.

        Returns:
            The commanded belt speed, inside the configured range.
        """
        opened = at_seconds - self.settings.window_seconds
        while self._arrivals and self._arrivals[0] < opened:
            self._arrivals.popleft()
        # Divided by the window rather than by how long the run has lasted,
        # so the figure is a rate the line is achieving now and not an
        # average over a start-up transient it has long left behind. Early in
        # a run the window is not yet full, which reads as a low rate and is
        # the truth: the line has not yet delivered that many objects.
        filled = min(at_seconds, self.settings.window_seconds)
        self._measured = len(self._arrivals) / filled if filled > 0.0 else 0.0

        # Positional form, around the speed the run started at. The velocity
        # form, adding the terms to the current command each tick, makes the
        # proportional term an integrator of its own: at the physics rate it
        # accumulates five hundred times a second, and the loop slams to a
        # drive limit on any error at all and sits there.
        if self._base is None:
            self._base = self.speed
        error = self.settings.rate - self._measured
        wanted = (
            self._base
            + self.settings.proportional_gain * error
            + self.settings.integral_gain * self._accumulated
        )
        bounded = min(max(wanted, self.lowest), self.highest)
        # Anti-windup by conditional integration. Accumulating while the
        # command is pinned to a limit buys nothing, because the extra term
        # cannot reach the plant, and it costs an overshoot on the way back
        # when the condition clears.
        pinned_high = bounded >= self.highest and error > 0.0
        pinned_low = bounded <= self.lowest and error < 0.0
        if not (pinned_high or pinned_low):
            self._accumulated += error * timestep
        self.speed = bounded
        return bounded


def _positive(block: dict[str, Any], key: str) -> float:
    """Read a value that has to be above zero.

    Args:
        block: The `feed` block.
        key: The key required.

    Returns:
        The value.

    Raises:
        WorldConfigError: If the key is absent or the value is not positive.
    """
    value = float(require(block, key, "feed"))
    if not value > 0.0:
        raise WorldConfigError(
            f"feed.{key} is {value}, and a line carrying nothing is not a line"
        )
    return value
