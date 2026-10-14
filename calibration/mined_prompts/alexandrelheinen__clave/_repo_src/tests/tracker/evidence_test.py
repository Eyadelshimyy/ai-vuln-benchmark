"""The evidence envelope and its five payloads.



Two rules give the union its value, and both are tested here rather than
described. A variant states what was measured and never what should be done, so
no payload may carry a channel, a grasp or an instruction. And the envelope's
role has to agree with the payload it carries, because a `Code` arriving under
the detection role is a wiring fault that would otherwise fuse silently.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from clave.taxonomy import MATERIAL_CLASSES
from clave.tracker.belt_frame import Footprint
from clave.tracker.evidence import (
    Code,
    Detection,
    Evidence,
    EvidenceError,
    GroundTruth,
    Height,
    Material,
    Payload,
    PixelMask,
    Role,
)

OUTCOME_COUNT = len(MATERIAL_CLASSES) + 1
"""Eleven taxonomy classes plus reject, which is what a posterior spans."""


def a_footprint() -> Footprint:
    """A box resting on the belt surface."""
    return Footprint(
        center=np.asarray((-1.0, 0.0, 0.93), dtype=np.float64),
        major_extent=0.10,
        minor_extent=0.06,
        yaw=0.0,
    )


def a_mask() -> PixelMask:
    """Two rows of ten pixels, as a segmenter would report them."""
    return PixelMask(width=64, height=64, runs=((10, 20, 10), (11, 20, 10)))


def a_detection() -> Detection:
    """What the wide camera reports for one object."""
    return Detection(footprint=a_footprint(), mask=a_mask(), height=0.06)


def wrap(payload: Payload, role: Role, **overrides: object) -> Evidence:
    """An envelope around one payload."""
    fields: dict[str, object] = {
        "source_id": "gate_wide",
        "role": role,
        "observed_at_nanos": 1_000,
        "confidence": 0.9,
        "payload": payload,
    }
    fields.update(overrides)
    return Evidence(**fields)  # type: ignore[arg-type]


def test_the_envelope_carries_provenance_a_clock_and_one_payload() -> None:
    """The envelope carries provenance a clock and one payload."""
    evidence = wrap(a_detection(), Role.DETECTION)
    assert evidence.source_id == "gate_wide"
    assert evidence.role is Role.DETECTION
    assert evidence.observed_at_nanos == 1_000
    assert evidence.confidence == pytest.approx(0.9)
    assert isinstance(evidence.payload, Detection)


def test_the_union_admits_exactly_the_five_specified_variants() -> None:
    """A sixth variant is a contract change, not an import."""
    assert set(Payload.__args__) == {Detection, Material, Code, Height, GroundTruth}


def test_no_payload_states_what_should_be_done() -> None:
    """No payload states what should be done.

    A variant states what was measured. The moment one carries a channel or a
    grasp, an adapter has started deciding, and the routing policy and the
    effector stop owning the things they own.
    """
    forbidden = {
        "channel",
        "grasp",
        "grasp_point",
        "grasp_width",
        "grasp_axis",
        "action",
        "command",
        "pick",
        "target",
    }
    for variant in Payload.__args__:
        names = {field.name for field in dataclasses.fields(variant)}
        assert not (names & forbidden), (
            f"{variant.__name__} carries {names & forbidden}"
        )


def test_a_payload_arriving_under_the_wrong_role_is_refused() -> None:
    """A payload arriving under the wrong role is refused.

    The role is what fusion dispatches on, so a code read filed as a detection
    would be folded by the wrong rule and nothing would say so.
    """
    with pytest.raises(EvidenceError, match="role"):
        wrap(a_detection(), Role.CODE)
    with pytest.raises(EvidenceError, match="role"):
        wrap(Height(top_surface=0.96), Role.DETECTION)


def test_every_variant_has_a_role_it_belongs_to() -> None:
    """No variant is orphaned from the enum."""
    pairs = [
        (a_detection(), Role.DETECTION),
        (Material(posterior=(1.0 / OUTCOME_COUNT,) * OUTCOME_COUNT), Role.SPECTRAL),
        (Code(symbology="UPC_A", digits="037600138727", quad=()), Role.CODE),
        (Height(top_surface=0.96), Role.DEPTH),
        (
            GroundTruth(
                object_id=3,
                material_class="M-06",
                position=np.asarray((-1.0, 0.0, 0.93), dtype=np.float64),
            ),
            Role.GROUND_TRUTH,
        ),
    ]
    assert {type(payload) for payload, _ in pairs} == set(Payload.__args__)
    for payload, role in pairs:
        assert wrap(payload, role).payload is payload  # type: ignore[arg-type]


def test_a_confidence_outside_the_unit_interval_is_refused() -> None:
    """The envelope states a certainty, so it has to be one."""
    for bad in (-0.01, 1.01, float("nan")):
        with pytest.raises(EvidenceError, match="confidence"):
            wrap(a_detection(), Role.DETECTION, confidence=bad)


def test_a_material_posterior_spans_the_taxonomy_plus_reject() -> None:
    """Eleven classes and reject, and it sums to one."""
    with pytest.raises(EvidenceError, match="outcomes"):
        Material(posterior=(0.5, 0.5))
    with pytest.raises(EvidenceError, match="sum"):
        Material(posterior=(1.0,) * OUTCOME_COUNT)
    Material(posterior=(1.0 / OUTCOME_COUNT,) * OUTCOME_COUNT)


def test_a_ground_truth_label_names_a_class_the_taxonomy_defines() -> None:
    """The simulator's label is still a taxonomy label."""
    with pytest.raises(EvidenceError, match="taxonomy"):
        GroundTruth(
            object_id=1,
            material_class="M-99",
            position=np.asarray((0.0, 0.0, 0.93), dtype=np.float64),
        )


def test_a_code_whose_check_digit_fails_is_not_a_code() -> None:
    """A code whose check digit fails is not a code.

    The decoder is task 6; the refusal belongs to the payload, so a Code that
    exists is a Code that checked out whatever produced it.
    """
    with pytest.raises(EvidenceError, match="check digit"):
        Code(symbology="UPC_A", digits="037600138728", quad=())


def test_a_mask_counts_its_own_pixels_and_states_its_bounds() -> None:
    """A mask counts its own pixels and states its bounds.

    Run-length triples rather than an array, so an adapter's whole input
    surface is a literal a test can write.
    """
    mask = a_mask()
    assert mask.pixel_count == 20
    assert mask.bounds == (20, 10, 29, 11)


def test_a_mask_run_outside_its_own_frame_is_refused() -> None:
    """A mask describes pixels of a frame that exists."""
    with pytest.raises(EvidenceError, match="outside"):
        PixelMask(width=8, height=8, runs=((2, 6, 10),))
    with pytest.raises(EvidenceError, match="outside"):
        PixelMask(width=8, height=8, runs=((9, 0, 2),))


def test_an_empty_mask_is_refused_rather_than_reported_as_an_object() -> None:
    """No pixels is not a detection."""
    with pytest.raises(EvidenceError, match="no pixels"):
        PixelMask(width=8, height=8, runs=())
