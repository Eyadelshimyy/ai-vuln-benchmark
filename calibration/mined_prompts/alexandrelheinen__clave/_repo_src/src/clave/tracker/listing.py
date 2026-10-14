"""One settled record, rendered for reading.

The tracker's belief is a dataclass, and a dataclass printed raw is a wall of
floats. This turns one record into field-name to readable-string pairs, so a
run can write down what it believed beside the frames it rendered.

Nothing here draws. Where the record is shown in the world, it is shown as
geometry in the scene, which is [clave.tracker.markers]. A value painted onto
a captured frame argues for something the simulator never produced, and
`agents/claude.md` forbids exactly that.

The record's own shape is the contract: a field added to `WasteObject` and not
described here fails a test rather than quietly going unseen.
"""

from __future__ import annotations

from clave.tracker.track import WasteObject


def described_fields(record: WasteObject) -> dict[str, str]:
    """Return every property of one record, rendered for reading.

    Every field the record stores and every one it computes. The record's own
    shape is the contract, so a field added to `WasteObject` and not here fails
    a test rather than quietly going unseen.

    Args:
        record: The settled record.

    Returns:
        Field name to the string drawn for it.
    """
    simulated = record.simulated

    def mark(name: str, text: str) -> str:
        """Flag a value the simulator supplied rather than a sensor measured."""
        return f"{text}   [SIMULATED]" if name in simulated else text

    box = record.footprint
    orientation = (
        f"yaw {box.yaw:+.3f} rad" if box.oriented else "unoriented, no yaw claimed"
    )
    return {
        "track_id": str(record.track_id),
        "observed_at_nanos": f"{record.observed_at_nanos} ns",
        "valid_until_nanos": (
            f"{record.valid_until_nanos} ns "
            f"(+{(record.valid_until_nanos - record.observed_at_nanos) / 1e9:.2f} s)"
        ),
        "footprint": (
            f"centre ({box.center[0]:.3f}, {box.center[1]:.3f}, "
            f"{box.center[2]:.3f}) m  "
            f"major {box.major_extent:.3f} m  minor {box.minor_extent:.3f} m  "
            f"{orientation}"
        ),
        "height": (
            f"{record.height:.3f} m" if record.height is not None else "unmeasured"
        ),
        "material": mark(
            "material", f"{record.material} at {record.material_confidence:.3f}"
        ),
        "material_confidence": f"{record.material_confidence:.3f}",
        "density": (
            mark(
                "density",
                f"{record.density.low:.3f} to {record.density.high:.3f} kg/m3",
            )
            if record.density is not None
            else "no class density, nothing in the world carries this class"
        ),
        "mass": (
            mark("mass", f"{record.mass.low:.3f} to {record.mass.high:.3f} kg")
            if record.mass is not None
            else "unknown, no density or no height"
        ),
        "codes": ", ".join(record.codes) if record.codes else "none read",
        "evidence": (
            ", ".join(sorted(role.value for role in record.evidence))
            if record.evidence
            else "none"
        ),
        "simulated": (
            ", ".join(sorted(record.simulated)) if record.simulated else "nothing"
        ),
        "grasp_point": (
            f"({record.grasp_point[0]:.3f}, {record.grasp_point[1]:.3f}, "
            f"{record.grasp_point[2]:.3f}) m"
        ),
        "grasp_axis": f"{record.grasp_axis:+.3f} rad",
        "grasp_width": f"{record.grasp_width:.3f} m",
        "surface_normal": (
            f"({record.surface_normal[0]:.1f}, {record.surface_normal[1]:.1f}, "
            f"{record.surface_normal[2]:.1f})"
        ),
    }
