"""Enterprise audit and strict-schema contract for rubric item authoring."""

from __future__ import annotations

import re

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


def _levels() -> tuple[RubricLevel, ...]:
    """Return a valid three-level ordinal scale."""
    return (
        RubricLevel(0, "unsupported", "No support.", ("unsupported claim",)),
        RubricLevel(1, "partial_support", "Partial support.", ("mixed support",)),
        RubricLevel(2, "full_support", "Full support.", ("complete support",)),
    )


def _rubric(
    response_format: ResponseFormat = ResponseFormat.ORDINAL_RATING,
) -> RubricSpecification:
    """Return a versioned enterprise rubric fixture."""
    return RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which claims are supported.",
        response_format=response_format,
        levels=_levels(),
        task_families=("claim_verification",),
        evidence_requirements=("Quote the supporting source span.",),
        locale="en-US",
        rubric_version="1.2.3",
    )


def _blueprint(response_format: ResponseFormat = ResponseFormat.ORDINAL_RATING):
    """Compile one deterministic blueprint."""
    return compile_item_blueprints(
        _rubric(response_format),
        BlueprintPlan(
            difficulty_bands=(DifficultyBand.MEDIUM,),
            evidence_modes=(EvidenceMode.SINGLE_SOURCE,),
            items_per_cell=1,
            seed=20260803,
        ),
    )[0]


def test_rubric_revision_is_distinct_from_serialization_schema_version():
    """Human governance revision and wire-schema evolution are independently traceable."""
    rubric = _rubric()
    assert rubric.schema_version == "1.0"
    assert rubric.rubric_version == "1.2.3"
    assert rubric.to_dict()["rubric_version"] == "1.2.3"


@pytest.mark.parametrize("version", ["1", "1.2", "v1.2.3", "01.2.3", "1.02.3"])
def test_rubric_revision_requires_canonical_semantic_version(version):
    """Rubric revisions reject ambiguous or non-canonical version strings."""
    with pytest.raises(ValueError, match=r"rubric_version.*semantic version"):
        RubricSpecification(
            rubric_id="faithfulness_rubric",
            construct_id="evidence_grounding",
            construct_definition="Degree to which claims are supported.",
            response_format=ResponseFormat.ORDINAL_RATING,
            levels=_levels(),
            task_families=("claim_verification",),
            evidence_requirements=("Quote the supporting source span.",),
            rubric_version=version,
        )


def test_blueprint_exposes_full_sha256_and_migration_safe_128_bit_handle():
    """Legacy display ids coexist with a stronger public audit handle and full digest."""
    rubric = _rubric()
    blueprint = _blueprint()
    contract = build_generation_contract(rubric, blueprint)
    assert re.fullmatch(r"[0-9a-f]{64}", blueprint.blueprint_fingerprint)
    assert blueprint.to_dict()["blueprint_fingerprint"] == (
        blueprint.blueprint_fingerprint
    )
    assert blueprint.blueprint_id == (
        f"item_blueprint_{blueprint.blueprint_fingerprint[:16]}"
    )
    assert contract["blueprint"]["blueprint_handle"] == (
        f"item_blueprint_{blueprint.blueprint_fingerprint[:32]}"
    )


def test_generation_contract_exposes_full_sha256_and_128_bit_handle():
    """Generation provenance remains collision-resistant without breaking display ids."""
    rubric = _rubric()
    blueprint = compile_item_blueprints(rubric)[0]
    contract = build_generation_contract(rubric, blueprint)
    fingerprint = contract["contract_fingerprint"]
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
    assert contract["contract_id"] == f"generation_contract_{fingerprint[:16]}"
    assert contract["contract_handle"] == f"generation_contract_{fingerprint[:32]}"
    assert contract["blueprint"]["blueprint_fingerprint"] == (
        blueprint.blueprint_fingerprint
    )


def test_structured_output_schema_declares_json_schema_draft_2020_12():
    """Provider adapters can select the intended validation dialect unambiguously."""
    contract = build_generation_contract(_rubric(), _blueprint())
    assert contract["output_schema"]["$schema"] == (
        "https://json-schema.org/draft/2020-12/schema"
    )


def test_structured_output_echoes_immutable_rubric_and_blueprint_provenance():
    """Wrong-blueprint replay fails structural validation before semantic checks."""
    rubric = _rubric()
    blueprint = compile_item_blueprints(rubric)[0]
    contract = build_generation_contract(rubric, blueprint)
    schema = contract["output_schema"]
    provenance = {
        "blueprint_id": blueprint.blueprint_id,
        "blueprint_handle": contract["blueprint"]["blueprint_handle"],
        "blueprint_fingerprint": blueprint.blueprint_fingerprint,
        "rubric_id": rubric.rubric_id,
        "rubric_version": rubric.rubric_version,
        "rubric_fingerprint": rubric.fingerprint,
    }
    assert schema["allOf"] == [{"required": list(provenance)}]
    for field, expected in provenance.items():
        assert schema["properties"][field] == {"const": expected}


@pytest.mark.parametrize(
    ("response_format", "key_field", "expected"),
    [
        (
            ResponseFormat.CONSTRUCTED_RESPONSE,
            "reference_response",
            {"type": "string", "minLength": 1, "maxLength": 8_192},
        ),
        (
            ResponseFormat.SELECTED_RESPONSE,
            "option_ids",
            {
                "type": "array",
                "minItems": 1,
                "maxItems": 32,
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                },
            },
        ),
        (ResponseFormat.BINARY_JUDGMENT, "value", {"type": "boolean"}),
        (
            ResponseFormat.ORDINAL_RATING,
            "score",
            {"type": "integer", "enum": [0, 1, 2]},
        ),
        (
            ResponseFormat.PAIRWISE_COMPARISON,
            "preferred_option_id",
            {
                "oneOf": [
                    {"type": "string", "minLength": 1, "maxLength": 128},
                    {"type": "null"},
                ]
            },
        ),
    ],
)
def test_answer_key_schema_is_closed_for_each_response_format(
    response_format,
    key_field,
    expected,
):
    """Each typed key remains closed, bounded, and independently explainable."""
    rubric = _rubric(response_format)
    blueprint = _blueprint(response_format)
    answer_key = build_generation_contract(rubric, blueprint)["output_schema"][
        "properties"
    ]["answer_key"]
    assert answer_key["type"] == "object"
    assert answer_key["additionalProperties"] is False
    assert key_field in answer_key["required"]
    assert "rationale" in answer_key["required"]
    assert answer_key["properties"][key_field] == expected
    assert answer_key["properties"]["rationale"]["maxLength"] == 8_192
