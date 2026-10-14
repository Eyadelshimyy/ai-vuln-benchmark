"""Resource-budget contract for untrusted rubric-provider JSON."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.rubric import (
    BlueprintPlan,
    CandidateValidationError,
    EvidenceMode,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    build_generation_request,
    compile_item_blueprints,
    parse_generated_item_candidate,
)


def _closed_book_request():
    """Return a valid request needed to reach the JSON trust boundary."""
    rubric = RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Grounded response quality.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(0, "unsupported", "No support.", ("unsupported",)),
            RubricLevel(1, "supported", "Supported.", ("supported",)),
        ),
        task_families=("claim_verification",),
        evidence_requirements=("Use the declared task evidence.",),
    )
    blueprint = compile_item_blueprints(
        rubric,
        BlueprintPlan(evidence_modes=(EvidenceMode.CLOSED_BOOK,)),
    )[0]
    return build_generation_request(rubric, blueprint)


def test_provider_json_node_count_is_bounded_before_field_validation():
    """Many shallow containers cannot bypass the raw-size and depth controls."""
    raw_json = json.dumps(
        {"nodes": [[] for _ in range(50_001)]},
        separators=(",", ":"),
    )
    assert len(raw_json) < 262_144
    with pytest.raises(CandidateValidationError) as error:
        parse_generated_item_candidate(raw_json, _closed_book_request())
    assert error.value.code == "json_node_budget"
