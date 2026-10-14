"""Complexity regression for generated-item testlet pilot handoffs."""

from __future__ import annotations

from pathlib import Path
import runpy

from fast_mlsirm.rubric import TestletPilotDesign, build_testlet_pilot_design
import fast_mlsirm.rubric.testlet_pilot as testlet_pilot

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_testlet_pilot_design.py"))
)
_testlet_records = _FIXTURES["_testlet_records"]


class _CountForbiddenTuple(tuple):
    """Tuple that fails if validation regresses to repeated linear scans."""

    def count(self, value: object) -> int:
        """Reject use of the quadratic ``tuple.count`` validation pattern."""
        raise AssertionError(f"tuple.count must not be called for {value!r}")


def test_testlet_membership_validation_avoids_repeated_tuple_scans() -> None:
    """Testlet group sizes are computed without one full scan per group."""
    design = build_testlet_pilot_design(_testlet_records())
    binary_design = design.binary_design
    object.__setattr__(
        binary_design,
        "item_factor_ids",
        _CountForbiddenTuple(binary_design.item_factor_ids),
    )

    rebuilt = TestletPilotDesign(
        binary_design=binary_design,
        schema_version=binary_design.schema_version,
        _design_token=testlet_pilot._TESTLET_DESIGN_TOKEN,
    )

    assert rebuilt.item_testlet_ids == (0, 0, 1)
