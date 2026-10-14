"""Folding one reading into a belief, and the rules that make it trustworthy.

Three decisions here are load bearing, and each is a property of a signature
rather than of a comment.

**Conflict never resolves by source.** `weight_of` takes a confidence and an
elapsed time, and no function in this module accepts a `source_id`. Source
priority is not forbidden by a rule that somebody could forget; it is
unrepresentable, because there is nothing to prioritize by. A sensor is trusted
for what it reported and how sure it was, so adding one cannot silently reorder
the others.

**Elapsed time is clamped at zero.** A weight of the form `k ** elapsed`
inverts when elapsed is negative, so a reading older than the belief it joins
would outweigh a newer one. The contract anticipates two cameras firing 8 ms
apart, which is exactly when that happens, so the clamp is a rule here.

**Nothing ever reaches zero.** A class every reading so far dismissed would be
dismissed forever, because zero is unrecoverable under multiplication. The floor
is configuration rather than a constant here, for the same reason the world's
numbers are.

The fold table is an explicit dict rather than single dispatch, so what happens
to a payload nobody wrote a rule for has exactly one address to read.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from clave.errors import ClaveError
from clave.taxonomy import MATERIAL_CLASSES
from clave.tracker.evidence import (
    Code,
    Detection,
    Evidence,
    GroundTruth,
    Height,
    Material,
    Payload,
)
from clave.tracker.motion import Gains
from clave.world.config import Range, load, require

REJECT = "reject"
"""The twelfth outcome, beside the eleven taxonomy classes.

Whether it is a predicted class or an absence of confidence is a question
`docs/perception-contract.md` declines to settle. A slot is the reading taken
here, and the contract is what should be amended if that turns out wrong.
"""

OUTCOMES: tuple[str, ...] = tuple(entry.id for entry in MATERIAL_CLASSES) + (REJECT,)
"""Every outcome a posterior spans, in taxonomy order with reject last."""


class FusionError(ClaveError):
    """Evidence cannot be folded into a belief as it stands."""


@dataclass(frozen=True)
class Posterior:
    """What the tracker currently believes an object is made of.

    Attributes:
        weights: One probability per outcome, in `OUTCOMES` order.
    """

    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        """Refuse anything that is not a distribution over the outcomes.

        Raises:
            FusionError: If the wrong number of outcomes is spanned, or the
                weights do not sum to one.
        """
        if len(self.weights) != len(OUTCOMES):
            raise FusionError(
                f"a belief spans {len(OUTCOMES)} outcomes and this one spans "
                f"{len(self.weights)}"
            )
        total = math.fsum(self.weights)
        if abs(total - 1.0) > 1e-9:
            raise FusionError(f"a belief sums to one, and this one sums to {total}")

    @classmethod
    def uniform(cls) -> Posterior:
        """Return the belief held before any evidence arrives."""
        return cls(weights=(1.0 / len(OUTCOMES),) * len(OUTCOMES))

    @property
    def most_likely(self) -> tuple[str, float]:
        """Return the leading outcome and how much weight it holds."""
        best = max(range(len(OUTCOMES)), key=lambda index: self.weights[index])
        return OUTCOMES[best], self.weights[best]

    def fold(
        self, likelihood: Sequence[float], weight: float, floor: float
    ) -> Posterior:
        """Return this belief updated by one likelihood.

        The likelihood is raised to the weight before multiplying, so a weak
        reading moves the belief less than a strong one without needing a
        separate rule for how much.

        Args:
            likelihood: One value per outcome, in `OUTCOMES` order.
            weight: How much the reading counts, from 0.0 upward.
            floor: The smallest weight any outcome may hold afterwards.

        Returns:
            The updated belief.

        Raises:
            FusionError: If the likelihood spans the wrong outcomes, or the
                update leaves no probability anywhere.
        """
        if len(likelihood) != len(OUTCOMES):
            raise FusionError(
                f"a likelihood spans {len(OUTCOMES)} outcomes and this one "
                f"spans {len(likelihood)}"
            )
        if weight <= 0.0:
            return self
        raised = [
            prior * (max(value, floor) ** weight)
            for prior, value in zip(self.weights, likelihood, strict=True)
        ]
        total = math.fsum(raised)
        if total <= 0.0 or not math.isfinite(total):
            raise FusionError(
                "the update leaves no probability anywhere, so the reading and "
                "the belief share no outcome either could hold"
            )
        return Posterior(
            weights=_floored(tuple(value / total for value in raised), floor)
        )


def _floored(weights: tuple[float, ...], floor: float) -> tuple[float, ...]:
    """Return the distribution with every outcome at or above the floor.

    Lifting each value to the floor and then renormalizing does not work: the
    division puts the lifted ones back under it, which is how the first version
    of this landed a floor of 0.0005 at 0.000497. So the floor is reserved
    first and only what is left over is distributed, which makes the guarantee
    exact rather than approximate.

    Args:
        weights: A distribution.
        floor: The smallest weight any outcome may hold.

    Returns:
        A distribution summing to one in which nothing sits below the floor,
        and whose ordering is unchanged.
    """
    reserved = floor * len(weights)
    spare = 1.0 - reserved
    excess = [max(0.0, value - floor) for value in weights]
    total = math.fsum(excess)
    if total <= 0.0:
        return (1.0 / len(weights),) * len(weights)
    return tuple(floor + spare * value / total for value in excess)


@dataclass(frozen=True)
class FusionSettings:
    """Everything the fold reads before it starts.

    Attributes:
        posterior_floor: The smallest weight any outcome may hold.
        recency_half_life: How long a reading takes to lose half its weight.
        densities: Bulk density band per material class, in kilograms per cubic
            meter. A class with no object in the world has no entry, and the
            mass of an object in one is unknown rather than estimated.
        along: Filter gains for the belt travel axis.
        across: Filter gains for the lateral axis.
        retire_after: How long a track may go unobserved before it is
            dropped, in seconds. An object nobody has seen for that long is
            not an object whose position is known, and carrying it forward
            produces a confident record about a fiction.
    """

    posterior_floor: float
    recency_half_life: float
    densities: dict[str, Range]
    along: Gains
    across: Gains
    retire_after: float

    @classmethod
    def load(cls, path: Path, world: Path) -> FusionSettings:
        """Read the settings, taking the density bands from the world.

        The world already declares a bulk density range per object, so the band
        for a class is the union of the ranges its objects declare. A second
        table here would give the world and the tracker two answers to one
        question, and the first time they disagreed nobody would know which was
        right.

        Args:
            path: The fusion configuration.
            world: The world configuration the densities come from.

        Returns:
            The settings.

        Raises:
            FusionError: If the floor or the half life describe nothing usable.
        """
        raw = yaml.safe_load(path.read_text())
        floor = float(require(require(raw, "posterior"), "floor", "posterior"))
        half_life = float(
            require(require(raw, "recency"), "half_life_seconds", "recency")
        )
        if not 0.0 < floor < 1.0 / len(OUTCOMES):
            raise FusionError(
                f"a posterior floor of {floor} leaves no room for a belief "
                f"across {len(OUTCOMES)} outcomes"
            )
        if not math.isfinite(half_life) or half_life <= 0.0:
            raise FusionError(
                f"a half life of {half_life} describes no decay, so no reading "
                f"would ever age"
            )
        motion = require(raw, "motion")
        return cls(
            posterior_floor=floor,
            recency_half_life=half_life,
            densities=_densities_of(load(world)),
            along=_gains(require(motion, "along", "motion"), "motion.along"),
            across=_gains(require(motion, "across", "motion"), "motion.across"),
            retire_after=_retirement(raw),
        )


def _retirement(raw: dict[str, Any]) -> float:
    """Read how long a track may go unobserved before it is dropped.

    Args:
        raw: The parsed fusion configuration.

    Returns:
        The bound in seconds.

    Raises:
        FusionError: If the key is absent or not above zero. A bound at or
            below zero retires a track the instant it opens.
    """
    value = float(
        require(require(raw, "lifecycle"), "retire_after_seconds", "lifecycle")
    )
    if not value > 0.0:
        raise FusionError(
            f"a retirement bound of {value} s drops a track before its first "
            f"observation has been read"
        )
    return value


def _gains(block: dict[str, Any], path: str) -> Gains:
    """Read one axis's filter gains.

    Args:
        block: The axis block.
        path: Dotted path, for the message.

    Returns:
        The gains.

    Raises:
        FusionError: If either gain is absent or outside the unit range. A
            gain above one over-corrects past the observation and one below
            zero corrects away from it, and neither describes a filter.
    """
    alpha = float(require(block, "alpha", path))
    beta = float(require(block, "beta", path))
    for name, value in (("alpha", alpha), ("beta", beta)):
        if not 0.0 <= value <= 1.0:
            raise FusionError(
                f"{path}.{name} is {value}, and a gain outside zero to one "
                f"corrects past the observation or away from it"
            )
    return Gains(alpha=alpha, beta=beta)


def _densities_of(world: dict[str, object]) -> dict[str, Range]:
    """Return the bulk density band per material class the world declares.

    Args:
        world: The loaded world configuration.

    Returns:
        One range per class that has at least one object. A class with none is
        absent, which is what makes its mass unknown rather than estimated.
    """
    bands: dict[str, Range] = {}
    for entry in require(world, "objects"):
        class_id = str(require(entry, "material_class", "objects"))
        low, high = (
            float(value) for value in require(entry, "density_kg_per_m3", "objects")
        )
        known = bands.get(class_id)
        bands[class_id] = (
            Range(low, high)
            if known is None
            else Range(min(known.low, low), max(known.high, high))
        )
    return bands


def weight_of(confidence: float, elapsed_seconds: float, half_life: float) -> float:
    """Return how much one reading counts.

    Two arguments and no third. There is no `source_id` here, which forbids
    source priority: a rule that cannot be expressed cannot be forgotten.

    Elapsed time is clamped at zero. A reading older than the belief it joins
    has aged no less than a fresh one, and without the clamp the exponent turns
    negative and the stale reading outweighs everything.

    Args:
        confidence: The adapter's own certainty, from 0.0 to 1.0.
        elapsed_seconds: How long ago the reading was taken, relative to the
            instant it is being folded at. Negative means it arrived out of
            order.
        half_life: How long a reading takes to lose half its weight.

    Returns:
        The weight, never above the confidence it started with.
    """
    aged = max(0.0, elapsed_seconds)
    return float(confidence * 0.5 ** (aged / half_life))


def code_prior(components: Sequence[str], floor: float) -> tuple[float, ...]:
    """Return the likelihood a bill of materials implies.

    A GTIN does not name a material. A product resolves to a packaging bill of
    materials, and a jar returning glass, plastic and aluminum says that one of
    them carries the label rather than which. So the weight spreads evenly over
    the components named, and where the code resolves to a single rigid
    component that prior is close to decisive.

    Args:
        components: Taxonomy identifiers the packaging is made of.
        floor: The weight every other outcome keeps.

    Returns:
        One value per outcome, in `OUTCOMES` order.
    """
    named = {component for component in components if component in OUTCOMES}
    if not named:
        return (1.0,) * len(OUTCOMES)
    share = 1.0 / len(named)
    return tuple(share if outcome in named else floor for outcome in OUTCOMES)


@dataclass(frozen=True)
class Fold:
    """What folding one reading did.

    Attributes:
        posterior: The belief afterwards, unchanged when nothing was learned.
        recognized: Whether any rule knew what to do with the payload. A
            consumer that does not know a variant keeps working, and this is
            how it finds out it did not know.
    """

    posterior: Posterior
    recognized: bool


Resolver = Callable[[str], Sequence[str]]
"""Turns decoded digits into the components the packaging is made of.

Supplied by the caller rather than imported, so this module needs no catalog and
the catalog needs no fusion. The default resolves nothing, which is also what an
unknown GTIN does.
"""


def _no_components(digits: str) -> Sequence[str]:
    """Resolve nothing, which is what an unknown GTIN resolves to."""
    del digits
    return ()


def fold(
    posterior: Posterior,
    evidence: Evidence,
    at_nanos: int,
    settings: FusionSettings,
    resolve: Resolver = _no_components,
) -> Fold:
    """Fold one reading into a belief.

    Args:
        posterior: What is believed now.
        evidence: The reading to fold.
        at_nanos: The instant to fold it at, which is what its age is measured
            against.
        settings: The floor and the half life.
        resolve: Turns a decoded symbol into packaging components.

    Returns:
        The belief afterwards, and whether any rule recognized the payload.
    """
    rule = FOLDS.get(type(evidence.payload))
    if rule is None:
        # The one miss branch. A payload nobody wrote a rule for leaves the
        # belief alone and says so, rather than raising: a consumer that does
        # not know a variant is supposed to keep working.
        return Fold(posterior=posterior, recognized=False)
    elapsed = (at_nanos - evidence.observed_at_nanos) / 1_000_000_000
    weight = weight_of(evidence.confidence, elapsed, settings.recency_half_life)
    return Fold(
        posterior=rule(posterior, evidence.payload, weight, settings, resolve),
        recognized=True,
    )


def _fold_geometry(
    posterior: Posterior,
    payload: Payload,
    weight: float,
    settings: FusionSettings,
    resolve: Resolver,
) -> Posterior:
    """Leave the belief alone, because geometry measures no material.

    A detection says where an object is and a height says how tall it is.
    Neither says what it is made of, and a rule that invented an opinion here
    would be inventing evidence.
    """
    del payload, weight, settings, resolve
    return posterior


def _fold_material(
    posterior: Posterior,
    payload: Payload,
    weight: float,
    settings: FusionSettings,
    resolve: Resolver,
) -> Posterior:
    """Fold a classifier's distribution straight in."""
    del resolve
    assert isinstance(payload, Material)
    return posterior.fold(payload.posterior, weight, settings.posterior_floor)


def _fold_code(
    posterior: Posterior,
    payload: Payload,
    weight: float,
    settings: FusionSettings,
    resolve: Resolver,
) -> Posterior:
    """Fold a bill of materials in, without letting it overrule what is seen.

    The contract says a code never overrides a confident visual observation.
    That is enforced by shrinking the code's weight as the belief sharpens: a
    belief already at 0.95 leaves a code almost nothing to push with, and a
    belief near uniform leaves it almost all of its confidence.
    """
    assert isinstance(payload, Code)
    components = resolve(payload.digits)
    if not components:
        return posterior
    _, leading = posterior.most_likely
    return posterior.fold(
        code_prior(components, settings.posterior_floor),
        weight * (1.0 - leading),
        settings.posterior_floor,
    )


def _fold_ground_truth(
    posterior: Posterior,
    payload: Payload,
    weight: float,
    settings: FusionSettings,
    resolve: Resolver,
) -> Posterior:
    """Fold the simulator's label in through the path a sensor would use."""
    del resolve
    assert isinstance(payload, GroundTruth)
    floor = settings.posterior_floor
    likelihood = tuple(
        1.0 if outcome == payload.material_class else floor for outcome in OUTCOMES
    )
    return posterior.fold(likelihood, weight, floor)


FOLDS: dict[type, Callable[..., Posterior]] = {
    Detection: _fold_geometry,
    Height: _fold_geometry,
    Material: _fold_material,
    Code: _fold_code,
    GroundTruth: _fold_ground_truth,
}
"""Which rule folds which payload.

An explicit dict rather than single dispatch, chosen so that "what happens to a
payload nobody wrote a rule for" has one address to read instead of being spread
across as many modules as register a rule.
"""


def mass_band(
    material_class: str,
    volume: float,
    settings: FusionSettings,
    packaging_mass: float | None = None,
) -> Range | None:
    """Return what the object might weigh, as a band.

    Density times footprint volume is weak. A bottle empty or half full differs
    tenfold, and that difference decides whether a suction cup holds, so a
    single number would invite a consumer to trust something nobody measured.

    Args:
        material_class: Taxonomy identifier of the form `M-NN`.
        volume: The footprint volume in cubic meters.
        settings: Carries the density band per class.
        packaging_mass: The mass a resolved GTIN reported, in kilograms, or
            None. Where one resolved it replaces the estimate, which is the one
            case the band is allowed to collapse.

    Returns:
        The band in kilograms, or None when nothing in the world carries this
        class and there is therefore no density to reason from.
    """
    if packaging_mass is not None:
        # A published packaging mass is a measurement of the empty container,
        # so the band stays open upward for whatever it still holds.
        return Range(packaging_mass, packaging_mass * 1.1)
    band = settings.densities.get(material_class)
    if band is None or volume <= 0.0:
        return None
    return Range(band.low * volume, band.high * volume)
