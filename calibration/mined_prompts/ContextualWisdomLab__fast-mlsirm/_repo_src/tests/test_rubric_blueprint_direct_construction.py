"""Direct-construction invariants for rubric blueprint provenance."""

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
    """Return a minimal versioned rubric with one declared task family."""
    return RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which claims are supported.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(0, "unsupported", "No support.", ("unsupported claim",)),
            RubricLevel(1, "supported", "Supported.", ("supported claim",)),
        ),
        task_families=("claim_verification",),
        evidence_requirements=("Quote the supporting source span.",),
        rubric_version="1.0.0",
    )


def _blueprint():
    """Return one compiler-produced canonical blueprint."""
    return compile_item_blueprints(
        _rubric(),
        BlueprintPlan(
            difficulty_bands=(DifficultyBand.MEDIUM,),
            evidence_modes=(EvidenceMode.SINGLE_SOURCE,),
            items_per_cell=1,
            seed=7,
        ),
    )[0]


def test_generation_contract_rejects_undeclared_direct_task_family():
    """A syntactically valid but undeclared task family cannot enter a contract."""
    forged = replace(_blueprint(), task_family="undeclared_task")
    with pytest.raises(ValueError, match="task_family"):
        build_generation_contract(_rubric(), forged)


def test_generation_contract_rejects_noncanonical_blueprint_display_id():
    """A display identifier must remain derived from the full blueprint digest."""
    forged = replace(_blueprint(), blueprint_id="item_blueprint_forged")
    with pytest.raises(ValueError, match="blueprint_id"):
        build_generation_contract(_rubric(), forged)
