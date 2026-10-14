"""Reading a published decision the way an outside consumer would.

These tests read the committed golden vectors rather than a payload built here,
which is the same check a consumer in another language runs before it ever sees
a live decision.
"""

from pathlib import Path

import pytest

from clave.ros.decisions import CONTRACT_VERSION, DecisionError, decode, read_vector

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "crates" / "clave-decision" / "contract" / "vectors" / "v1"


def test_the_nominal_vector_decodes_into_every_documented_field() -> None:
    """Every value here is quoted from the vector directory's own README."""
    decision = decode(read_vector(VECTORS / "nominal.hex"))
    assert decision.version == CONTRACT_VERSION
    assert decision.object_id == 4815162342
    assert decision.material_class == "M-01"
    assert decision.channel == 3
    assert decision.point == pytest.approx((0.412, -0.085, 0.031))
    assert decision.yaw_radians == pytest.approx(1.047)
    assert decision.reference_time_nanos == 9_100_000_000
    assert decision.window_start_nanos == 9_000_000_000
    assert decision.window_end_nanos == 9_250_000_000
    assert decision.confidence == pytest.approx(0.94, abs=1e-6)


def test_a_rejected_decision_reads_as_an_ordinary_decision() -> None:
    """The reject channel is a channel, and the class it carries is the real one."""
    decision = decode(read_vector(VECTORS / "rejected.hex"))
    assert decision.material_class == "M-11"
    assert decision.channel == 0
    assert decision.confidence == pytest.approx(0.41, abs=1e-6)


def test_the_window_duration_comes_from_the_window_rather_than_the_pose() -> None:
    """A subscriber uses this to tell an actionable decision from a stale one."""
    decision = decode(read_vector(VECTORS / "nominal.hex"))
    assert decision.window_seconds == pytest.approx(0.25)


def test_a_version_this_build_does_not_implement_is_refused_by_version() -> None:
    """The contract says reject whole and read no field past the version."""
    with pytest.raises(DecisionError, match="version 2"):
        decode(read_vector(VECTORS / "unknown-version.hex"))


def test_bytes_that_are_not_a_decision_are_reported_rather_than_raised_blindly() -> (
    None
):
    """One bad datagram is not a reason to stop a conveyor."""
    for payload in (b"", b"not cbor at all", b"\xa1\x64test\x01"):
        with pytest.raises(DecisionError):
            decode(payload)


def test_every_material_class_on_the_wire_maps_to_a_taxonomy_identifier() -> None:
    """The wire carries a name; the taxonomy owns what it means."""
    from clave.ros.decisions import WIRE_CLASSES
    from clave.taxonomy import MATERIAL_CLASSES

    assert len(WIRE_CLASSES) == len(MATERIAL_CLASSES)
    assert set(WIRE_CLASSES.values()) == {entry.id for entry in MATERIAL_CLASSES}
