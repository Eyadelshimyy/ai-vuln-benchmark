"""Canonical whole-artifact budget contracts for assessment specifications."""

from __future__ import annotations

from pathlib import Path
import runpy

from fast_mlsirm.scoring import canonical_json

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_contract_fixtures.py"))
)
assessment = _FIXTURES["assessment"]


def test_valid_metadata_budget_remains_addressable_inside_assessment() -> None:
    """A dense valid metadata tree must fit the complete artifact node budget."""
    dense_metadata = {
        "dense_values": [
            [outer_index * 64 + inner_index for inner_index in range(64)]
            for outer_index in range(14)
        ]
    }

    specification = assessment(metadata=dense_metadata)

    assert len(specification.assessment_fingerprint) == 64
    assert canonical_json(specification).startswith("{")
