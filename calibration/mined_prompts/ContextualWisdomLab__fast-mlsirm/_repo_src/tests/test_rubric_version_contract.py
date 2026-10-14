"""Regression contracts for rubric-governance version provenance."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fast_mlsirm.rubric import (
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    build_generation_contract,
    compile_item_blueprints,
)


def _rubric(version: str = "1.2.3") -> RubricSpecification:
    """Return a minimal versioned rubric fixture."""
    return RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which claims are supported.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(0, "unsupported", "No support.", ("unsupported claim",)),
            RubricLevel(1, "full_support", "Full support.", ("supported claim",)),
        ),
        task_families=("claim_verification",),
        evidence_requirements=("Quote the supporting source span.",),
        rubric_version=version,
    )


def test_compiler_copies_governance_version_into_blueprint_identity():
    """A blueprint and its full fingerprint retain the exact rubric revision."""
    blueprint = compile_item_blueprints(_rubric())[0]
    assert blueprint.rubric_version == "1.2.3"
    assert blueprint.to_dict()["rubric_version"] == "1.2.3"
    assert len(blueprint.blueprint_fingerprint) == 64


def test_generation_contract_rejects_a_blueprint_from_another_revision():
    """A display-compatible blueprint cannot be replayed under a new rubric version."""
    rubric = _rubric()
    blueprint = compile_item_blueprints(rubric)[0]
    with pytest.raises(ValueError, match="rubric_version"):
        build_generation_contract(
            rubric,
            replace(blueprint, rubric_version="1.2.4"),
        )


def test_rubric_revision_changes_all_content_addressed_identities():
    """A governance revision invalidates rubric, blueprint, and contract hashes."""
    first = _rubric("1.2.3")
    second = _rubric("1.2.4")
    first_blueprint = compile_item_blueprints(first)[0]
    second_blueprint = compile_item_blueprints(second)[0]
    first_contract = build_generation_contract(first, first_blueprint)
    second_contract = build_generation_contract(second, second_blueprint)

    assert first.fingerprint != second.fingerprint
    assert first_blueprint.blueprint_fingerprint != second_blueprint.blueprint_fingerprint
    assert first_contract["contract_fingerprint"] != second_contract["contract_fingerprint"]
