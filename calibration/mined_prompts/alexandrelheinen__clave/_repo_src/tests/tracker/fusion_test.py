"""Folding evidence into a belief, and the two rules that make it trustworthy.



Two tests here are the reason the module is shaped the way it is. The
source-swap test folds two disagreeing readings, then folds the same two with
their provenance exchanged, and demands the same answer. The out-of-order test
folds a reading older than the belief it is folded into and demands it does not
outweigh a newer one, which is the case a weight of the form `k ** elapsed`
inverts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import clave.tracker.evidence as evidence_module
from clave.taxonomy import MATERIAL_CLASSES
from clave.tracker.belt_frame import Footprint
from clave.tracker.evidence import (
    Code,
    Detection,
    Evidence,
    GroundTruth,
    Height,
    Material,
    PixelMask,
    Role,
)
from clave.tracker.fusion import (
    OUTCOMES,
    REJECT,
    FusionSettings,
    Posterior,
    code_prior,
    fold,
    mass_band,
    weight_of,
)

ROOT = Path(__file__).resolve().parents[2]
OUTCOME_COUNT = len(MATERIAL_CLASSES) + 1
SECOND = 1_000_000_000


def settings() -> FusionSettings:
    """The shipped fusion settings, with densities from the shipped world."""
    return FusionSettings.load(
        ROOT / "configs" / "perception" / "fusion.yml",
        ROOT / "configs" / "world" / "sorting_line.yml",
    )


def sharp(class_id: str, mass: float = 0.95) -> tuple[float, ...]:
    """A likelihood putting most of its weight on one class."""
    spread = (1.0 - mass) / (OUTCOME_COUNT - 1)
    order = [entry.id for entry in MATERIAL_CLASSES] + [REJECT]
    return tuple(mass if name == class_id else spread for name in order)


def reading(
    payload: object, role: Role, source: str, at: int = SECOND, confidence: float = 0.8
) -> Evidence:
    """One reading, from one named source, at one instant."""
    return Evidence(
        source_id=source,
        role=role,
        observed_at_nanos=at,
        confidence=confidence,
        payload=payload,  # type: ignore[arg-type]
    )


def a_detection() -> Detection:
    """A box on the belt, which says nothing about material."""
    return Detection(
        footprint=Footprint(
            center=np.asarray((-1.0, 0.0, 0.93), dtype=np.float64),
            major_extent=0.10,
            minor_extent=0.06,
            yaw=0.0,
        ),
        mask=PixelMask(width=64, height=64, runs=((10, 20, 10),)),
        height=0.06,
    )


def test_a_fresh_posterior_prefers_nothing() -> None:
    """Before any evidence, every outcome is equally open."""
    uniform = Posterior.uniform()
    assert len(uniform.weights) == OUTCOME_COUNT
    assert uniform.weights == pytest.approx((1.0 / OUTCOME_COUNT,) * OUTCOME_COUNT)


def test_a_posterior_stays_a_distribution_however_it_is_folded() -> None:
    """It sums to one and no class ever reaches zero."""
    belief = Posterior.uniform()
    for _ in range(20):
        belief = belief.fold(
            sharp("M-06"), weight=1.0, floor=settings().posterior_floor
        )
    assert sum(belief.weights) == pytest.approx(1.0)
    assert min(belief.weights) >= settings().posterior_floor
    assert belief.most_likely[0] == "M-06"


def test_a_class_starved_of_evidence_can_still_be_argued_back() -> None:
    """A class starved of evidence can still be argued back.

    A zero is unrecoverable under multiplication, so a class every reading so
    far has dismissed would be dismissed forever. The floor is what keeps the
    next reading able to speak.
    """
    floor = settings().posterior_floor
    belief = Posterior.uniform()
    for _ in range(40):
        belief = belief.fold(sharp("M-06", mass=0.99), weight=1.0, floor=floor)
    assert belief.most_likely[0] == "M-06"
    for _ in range(80):
        belief = belief.fold(sharp("M-09", mass=0.99), weight=1.0, floor=floor)
    assert belief.most_likely[0] == "M-09"


def test_weight_falls_with_age_and_rises_with_confidence() -> None:
    """The two things a weight may depend on, and nothing else."""
    assert weight_of(0.9, 0.0, half_life=2.0) == pytest.approx(0.9)
    assert weight_of(0.9, 2.0, half_life=2.0) == pytest.approx(0.45)
    assert weight_of(0.45, 0.0, half_life=2.0) == pytest.approx(0.45)
    assert weight_of(0.9, 4.0, half_life=2.0) < weight_of(0.9, 2.0, half_life=2.0)


def test_a_reading_older_than_the_belief_does_not_outweigh_a_newer_one() -> None:
    """A reading older than the belief does not outweigh a newer one.

    This is the case `k ** elapsed` inverts: a negative exponent makes the
    weight exceed one, so the stale reading wins. Every test in the designs
    this module replaced folded forward, so none of them would have caught it.
    """
    stale = weight_of(0.9, -5.0, half_life=2.0)
    fresh = weight_of(0.9, 0.0, half_life=2.0)
    assert stale <= fresh
    assert stale <= 0.9


def test_no_fusion_function_accepts_a_source() -> None:
    """No fusion function accepts a source.

    Source priority is not forbidden by a rule, it is unrepresentable: no
    function here takes a `source_id`, so there is nothing to prioritize by.
    """
    import inspect

    from clave.tracker import fusion

    for name, function in inspect.getmembers(fusion, inspect.isfunction):
        if function.__module__ != fusion.__name__:
            continue
        parameters = set(inspect.signature(function).parameters)
        assert "source_id" not in parameters, name
        assert "source" not in parameters, name


def test_exchanging_provenance_changes_no_belief() -> None:
    """Exchanging provenance changes no belief.

    The source-swap proof. Two disagreeing readings are folded, then the same
    two are folded again with their `source_id` and `role` exchanged. A system
    that trusted a sensor for being that sensor would answer differently.
    """
    config = settings()
    first = reading(Material(sharp("M-06")), Role.SPECTRAL, "gate_wide", confidence=0.8)
    second = reading(
        Material(sharp("M-09")), Role.SPECTRAL, "gate_code_left", confidence=0.6
    )

    forward = fold(Posterior.uniform(), first, SECOND, config).posterior
    forward = fold(forward, second, SECOND, config).posterior

    swapped_first = reading(
        Material(sharp("M-06")), Role.SPECTRAL, "gate_code_left", confidence=0.8
    )
    swapped_second = reading(
        Material(sharp("M-09")), Role.SPECTRAL, "gate_wide", confidence=0.6
    )
    reverse = fold(Posterior.uniform(), swapped_first, SECOND, config).posterior
    reverse = fold(reverse, swapped_second, SECOND, config).posterior

    assert forward.weights == pytest.approx(reverse.weights, abs=1e-12)


def test_a_payload_no_rule_knows_reports_the_miss_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A payload no rule knows reports the miss and changes nothing.

    The realistic shape of this is a half-finished addition: a sixth variant
    reaches the union and the role table, and whoever added it forgets the
    fusion rule. The envelope accepts it, fusion does not recognize it, and the
    belief is left alone rather than an exception being raised, because a
    consumer that does not know a variant is supposed to keep working.

    The fold table is an explicit dict with one miss branch precisely so that
    this behaviour has a single address to read.
    """

    class Spectral:
        """A sixth variant, as a later step would add."""

    monkeypatch.setitem(evidence_module.ROLE_OF, Spectral, Role.SPECTRAL)

    config = settings()
    before = Posterior.uniform()
    result = fold(
        before, reading(Spectral(), Role.SPECTRAL, "gate_wide"), SECOND, config
    )
    assert not result.recognized
    assert result.posterior.weights == pytest.approx(before.weights)


def test_a_detection_is_recognized_and_argues_no_material() -> None:
    """A rule exists; it has nothing to say about material."""
    config = settings()
    before = Posterior.uniform()
    result = fold(
        before, reading(a_detection(), Role.DETECTION, "gate_wide"), SECOND, config
    )
    assert result.recognized
    assert result.posterior.weights == pytest.approx(before.weights)


def test_a_height_is_recognized_and_argues_no_material() -> None:
    """Same for depth, which measures geometry rather than stuff."""
    config = settings()
    before = Posterior.uniform()
    result = fold(
        before,
        reading(Height(top_surface=0.96), Role.DEPTH, "gate_wide"),
        SECOND,
        config,
    )
    assert result.recognized
    assert result.posterior.weights == pytest.approx(before.weights)


def test_the_simulator_label_moves_the_belief_to_its_class() -> None:
    """Ground truth travels the fusion path like anything else."""
    config = settings()
    label = GroundTruth(
        object_id=3,
        material_class="M-06",
        position=np.asarray((-1.0, 0.0, 0.93), dtype=np.float64),
    )
    result = fold(
        Posterior.uniform(),
        reading(label, Role.GROUND_TRUTH, "simulator", confidence=1.0),
        SECOND,
        config,
    )
    assert result.recognized
    assert result.posterior.most_likely[0] == "M-06"


def test_a_code_prior_spreads_over_the_components_it_named() -> None:
    """A code prior spreads over the components it named.

    A GTIN resolves to a packaging bill of materials, so a jar returning glass,
    plastic and aluminum says one of them carries the label and not which.
    """
    prior = code_prior(("M-07", "M-04", "M-05"), floor=settings().posterior_floor)
    order = [entry.id for entry in MATERIAL_CLASSES] + [REJECT]
    named = [prior[order.index(name)] for name in ("M-07", "M-04", "M-05")]
    assert named == pytest.approx([named[0]] * 3)
    assert prior[order.index("M-06")] < named[0]


@pytest.mark.parametrize("held", [0.80, 0.90, 0.95])
def test_a_code_never_overrides_a_confident_visual_reading(held: float) -> None:
    """A code never overrides a confident visual reading.

    The rule the contract states outright. A belief already sure of steel is
    not talked out of it by a label that could be on a glass jar.

    The belief is built directly rather than folded up to, because the regime
    matters and folding overshoots it. Past about 0.99 the belief is so extreme
    that no code moves it whatever the rule says, so a test written up there
    passes with the rule deleted. These three all flip to M-07 if the cap on
    the code's weight is removed, which was checked.
    """
    config = settings()
    belief = Posterior(
        tuple(held if o == "M-06" else (1.0 - held) / 11 for o in OUTCOMES)
    )
    code = reading(
        Code(symbology="UPC_A", digits="037600138727"), Role.CODE, "gate_code_left"
    )
    after = fold(belief, code, SECOND, config, resolve=lambda _: ("M-07",)).posterior
    assert after.most_likely[0] == "M-06"


def test_a_code_does_move_a_belief_that_is_not_yet_sure() -> None:
    """A code does move a belief that is not yet sure.

    The other half. A code that could never move anything would be a code
    nobody needed to read.
    """
    config = settings()
    code = reading(
        Code(symbology="UPC_A", digits="037600138727"), Role.CODE, "gate_code_left"
    )
    after = fold(
        Posterior.uniform(), code, SECOND, config, resolve=lambda _: ("M-07",)
    ).posterior
    assert after.most_likely[0] == "M-07"


def test_a_code_that_resolves_to_nothing_argues_nothing() -> None:
    """An unknown GTIN yields no bill of materials, never a guess."""
    config = settings()
    before = Posterior.uniform()
    code = reading(
        Code(symbology="UPC_A", digits="037600138727"), Role.CODE, "gate_code_left"
    )
    result = fold(before, code, SECOND, config, resolve=lambda _: ())
    assert result.recognized
    assert result.posterior.weights == pytest.approx(before.weights)


def test_the_mass_band_comes_from_the_world_and_not_from_a_second_table() -> None:
    """The mass band comes from the world and not from a second table.

    configs/world/sorting_line.yml already declares a bulk density per object,
    so the band for a class is the union of its objects' ranges. A second table
    here would give the world and the tracker two answers.
    """
    config = settings()
    assert config.densities["M-06"].low == pytest.approx(180.0)
    assert config.densities["M-06"].high == pytest.approx(320.0)
    assert config.densities["M-02"].low == pytest.approx(70.0)
    assert config.densities["M-02"].high == pytest.approx(140.0)


def test_mass_is_a_band_rather_than_a_number() -> None:
    """Mass is a band rather than a number.

    Density times footprint volume is weak: a bottle empty or half full differs
    tenfold, and a single number invites a consumer to trust it.
    """
    config = settings()
    band = mass_band("M-06", volume=0.10 * 0.06 * 0.06, settings=config)
    assert band is not None
    assert band.low < band.high
    assert band.low == pytest.approx(180.0 * 0.00036)
    assert band.high == pytest.approx(320.0 * 0.00036)


def test_a_class_with_no_object_has_no_mass_rather_than_a_guess() -> None:
    """A class with no object has no mass rather than a guess.

    Seven of eleven classes have no object. Reporting a band for one of them
    would be reporting a number nothing measured.
    """
    assert mass_band("M-01", volume=0.001, settings=settings()) is None


def test_a_resolved_packaging_mass_narrows_the_band_to_itself() -> None:
    """A resolved packaging mass narrows the band to itself.

    Where a GTIN resolved, the packaging mass replaces the estimate. That is
    the one case the band is allowed to collapse.
    """
    config = settings()
    estimated = mass_band("M-06", volume=0.10 * 0.06 * 0.06, settings=config)
    narrowed = mass_band(
        "M-06", volume=0.10 * 0.06 * 0.06, settings=config, packaging_mass=0.410
    )
    assert narrowed is not None and estimated is not None
    assert narrowed.low <= 0.410 <= narrowed.high
    assert (narrowed.high - narrowed.low) < (estimated.high - estimated.low)
