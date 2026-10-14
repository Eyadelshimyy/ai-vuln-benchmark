"""Regression contract for pytest collection of the public testlet pilot type."""

from fast_mlsirm.rubric.testlet_pilot import TestletPilotDesign


def test_public_testlet_pilot_design_is_not_a_pytest_test_class() -> None:
    """The public ``TestletPilotDesign`` type must opt out of pytest discovery."""
    assert TestletPilotDesign.__test__ is False
