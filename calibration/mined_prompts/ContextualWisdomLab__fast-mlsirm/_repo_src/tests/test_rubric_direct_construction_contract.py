"""Regression contracts for directly constructed rubric blueprints."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fast_mlsirm.rubric import (
    BlueprintPlan,
    DifficultyBand,
    EvidenceMode,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    build_generation_contract,
    compile_item_blueprints,
)


def _rubric() -> RubricSpecification:
    """Return a compact versioned rubric with one declared task family."""
    return RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which substantive claims are supported.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(0, "unsupported", "No support.", ("unsupported claim",)),
            RubricLevel(1, "partial_support", "Partial support.", ("mixed support",)),
            RubricLevel(2, "full_support", "Full support.", ("complete support",)),
        ),
        task_families=("claim_verification",),
        evidence_requirements=("Quote the supporting source span.",),
        rubric_version="1.2.3",
    )


def _blueprint():
    """Compile one canonical blueprint through the supported factory path."""
    return compile_item_blueprints(
        _rubric(),
        BlueprintPlan(
            difficulty_bands=(DifficultyBand.MEDIUM,),
            evidence_modes=(EvidenceMode.SINGLE_SOURCE,),
            items_per_cell=1,
            seed=20260803,
        ),
    )[0]


def test_generation_contract_rejects_undeclared_task_family() -> None:
    """Direct construction cannot add a task family absent from the rubric."""
    rubric = _rubric()
    forged = replace(_blueprint(), task_family="different_task")

    with pytest.raises(ValueError, match="task_family"):
        build_generation_contract(rubric, forged)


def test_generation_contract_rejects_forged_blueprint_display_id() -> None:
    """A syntactically valid display id must still derive from the full digest."""
    rubric = _rubric()
    forged = replace(_blueprint(), blueprint_id="item_blueprint_forged")

    with pytest.raises(ValueError, match="blueprint_id"):
        build_generation_contract(rubric, forged)


def test_compiler_created_blueprint_remains_contract_compatible() -> None:
    """The supported compiler path still yields an auditable generation contract."""
    rubric = _rubric()
    blueprint = compile_item_blueprints(rubric)[0]

    contract = build_generation_contract(rubric, blueprint)

    assert contract["blueprint"]["blueprint_id"] == blueprint.blueprint_id
    assert contract["blueprint"]["blueprint_fingerprint"] == (
        blueprint.blueprint_fingerprint
    )
