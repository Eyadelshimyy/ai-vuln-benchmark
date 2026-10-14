"""Strict JSON-schema contracts for rubric-centered item generation."""

from __future__ import annotations

from fast_mlsirm.rubric import (
    BlueprintPlan,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    build_generation_contract,
    compile_item_blueprints,
)


def _contract() -> dict:
    """Return a generation contract with three ordered score levels."""
    rubric = RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which claims are supported by evidence.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(0, "unsupported", "No support.", ("unsupported claim",)),
            RubricLevel(1, "partial_support", "Partial support.", ("mixed support",)),
            RubricLevel(2, "full_support", "Full support.", ("complete support",)),
        ),
        task_families=("claim_verification",),
        evidence_requirements=("Quote supporting evidence.",),
    )
    blueprint = compile_item_blueprints(rubric, BlueprintPlan())[0]
    return build_generation_contract(rubric, blueprint)


def test_score_level_arrays_require_each_score_once_in_order():
    """Fixed-length arrays cannot duplicate one score while omitting another."""
    schema = _contract()["output_schema"]
    assert schema["$schema"].endswith("draft/2020-12/schema")
    for field in ("scoring_guide", "rubric_alignment"):
        field_schema = schema["properties"][field]
        assert field_schema["items"] is False
        assert field_schema["minItems"] == 3
        assert field_schema["maxItems"] == 3
        assert [
            entry["properties"]["score"]["const"]
            for entry in field_schema["prefixItems"]
        ] == [0, 1, 2]


def test_generated_text_and_collections_are_resource_bounded():
    """Provider output contracts impose explicit text and collection limits."""
    properties = _contract()["output_schema"]["properties"]
    assert properties["stem"]["maxLength"] == 8_192
    assert properties["stimulus"]["maxItems"] == 32
    assert properties["stimulus"]["items"]["minLength"] == 1
    assert properties["options"]["minItems"] == 0
    assert properties["options"]["maxItems"] == 0
    assert properties["source_attributions"]["maxItems"] == 32
    assert properties["safety_notes"]["maxItems"] == 32
    assert properties["safety_notes"]["items"]["minLength"] == 1
    assert properties["answer_key"]["properties"]["rationale"]["maxLength"] == 8_192


def test_authoring_instructions_expose_score_order_contract():
    """The natural-language contract mirrors the machine-enforced order."""
    instructions = _contract()["authoring_instructions"]
    assert any("ascending rubric-score order" in item for item in instructions)
    assert any("unique option_id" in item for item in instructions)
