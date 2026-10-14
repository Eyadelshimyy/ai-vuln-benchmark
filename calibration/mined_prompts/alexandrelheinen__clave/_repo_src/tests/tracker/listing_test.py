"""Writing down everything the tracker believes.

The rule this file keeps honest is that the listing shows what the record
carries and nothing else. A listing that invents a value argues for something
the system does not know, which is the failure the no-fabricated-evidence rule
guards against everywhere else in this repository.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from clave.tracker.belt_frame import Footprint
from clave.tracker.evidence import Role
from clave.tracker.listing import described_fields
from clave.tracker.track import WasteObject
from clave.world.config import Range

ROOT = Path(__file__).resolve().parents[2]


def a_record(track_id: int = 1, **overrides: object) -> WasteObject:
    """One settled record."""
    fields: dict[str, object] = {
        "track_id": track_id,
        "observed_at_nanos": 1_000_000_000,
        "valid_until_nanos": 4_000_000_000,
        "footprint": Footprint(
            center=np.asarray((-1.0, 0.0, 0.93), dtype=np.float64),
            major_extent=0.10,
            minor_extent=0.06,
            yaw=0.3,
        ),
        "height": 0.06,
        "material": "M-06",
        "material_confidence": 0.94,
        "density": Range(180.0, 320.0),
        "mass": Range(0.065, 0.115),
        "codes": ("037600138727",),
        "evidence": frozenset({Role.DETECTION, Role.GROUND_TRUTH}),
        "simulated": frozenset({"material"}),
    }
    fields.update(overrides)
    return WasteObject(**fields)  # type: ignore[arg-type]


def test_every_field_of_the_record_is_described() -> None:
    """Every field of the record is described.

    The contract is the record's own shape. Adding a field to `WasteObject`
    and not to the listing fails here, which is what stops the listing quietly
    showing less than the tracker knows.
    """
    stored = {field.name for field in dataclasses.fields(WasteObject)}
    computed = {"grasp_point", "grasp_axis", "grasp_width", "surface_normal"}
    assert described_fields(a_record()).keys() >= stored | computed


def test_a_simulated_field_is_marked_as_one() -> None:
    """A simulated field is marked as one.

    A reader auditing perception has to see which values the simulator
    supplied, because a material folded from `GroundTruth` is the simulator's
    label.
    """
    described = described_fields(a_record(simulated=frozenset({"material"})))
    assert "simulated" in described["material"].lower()
    clean = described_fields(a_record(simulated=frozenset()))
    assert "simulated" not in clean["material"].lower()


def test_an_unoriented_footprint_claims_no_yaw() -> None:
    """An unoriented footprint claims no yaw.

    A can is circular in plan. Reporting a confident yaw of zero is a claim
    the record declined to make.
    """
    round_box = Footprint(
        center=np.asarray((-1.0, 0.0, 0.93), dtype=np.float64),
        major_extent=0.07,
        minor_extent=0.07,
        yaw=0.0,
        oriented=False,
    )
    described = described_fields(a_record(footprint=round_box))
    assert "unoriented" in described["footprint"].lower()


def test_an_unmeasured_height_says_so_rather_than_reading_zero() -> None:
    """An unmeasured height says so rather than reading zero.

    No sensor on this line estimates depth, so this is the ordinary case
    rather than the edge one.
    """
    assert described_fields(a_record(height=None))["height"] == "unmeasured"


def test_the_listing_draws_nothing() -> None:
    """The listing draws nothing.

    Where the tracker's belief is shown in the world it is scene geometry, so
    nothing here may reach a renderer, a drawing library, or the published
    encoder.
    """
    source = (ROOT / "src" / "clave" / "tracker" / "listing.py").read_text()
    for forbidden in ("cv2", "mujoco", "clave.demo", "open_recorder"):
        assert forbidden not in source, forbidden
