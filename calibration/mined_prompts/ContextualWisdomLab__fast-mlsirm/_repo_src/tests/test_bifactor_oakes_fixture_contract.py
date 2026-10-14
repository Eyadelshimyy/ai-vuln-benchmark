"""Validate the generated bifactor Oakes oracle as strict JSON."""

from __future__ import annotations

import json
from pathlib import Path


OAKES_FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "bifactor_grm_stage3_oakes"
    / "mirt_oakes_fixture.json"
)


def test_bifactor_oakes_fixture_is_strict_json() -> None:
    """Require generated metadata vectors to remain valid JSON strings."""
    fixture_payload = json.loads(OAKES_FIXTURE_PATH.read_text(encoding="utf-8"))

    assert len(fixture_payload["mirt_vcov_dimnames"]) == 40
    assert all(
        isinstance(parameter_name, str)
        for parameter_name in fixture_payload["mirt_vcov_dimnames"]
    )
    assert len(fixture_payload["mirt_vcov_order_par"]) == 40
    assert all(
        isinstance(parameter_name, str)
        for parameter_name in fixture_payload["mirt_vcov_order_par"]
    )
