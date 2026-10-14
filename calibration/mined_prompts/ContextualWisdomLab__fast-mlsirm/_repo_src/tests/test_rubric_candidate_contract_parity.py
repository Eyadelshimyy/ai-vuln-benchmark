"""Cross-layer parity between generation contracts and candidate validation."""

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
    SourceDocument,
    build_generation_request,
    compile_item_blueprints,
    parse_generated_item_candidate,
)


def _request(response_format: ResponseFormat):
    """Build a single-source request for one response format."""
    rubric = RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which claims are supported by evidence.",
        response_format=response_format,
        levels=(
            RubricLevel(0, "unsupported", "No support.", ("unsupported",)),
            RubricLevel(1, "partial_support", "Partial support.", ("partial",)),
            RubricLevel(2, "full_support", "Full support.", ("complete",)),
        ),
        task_families=("claim_verification",),
        evidence_requirements=("Quote supporting evidence.",),
        rubric_version="1.2.3",
    )
    blueprint = compile_item_blueprints(
        rubric,
        BlueprintPlan(evidence_modes=(EvidenceMode.SINGLE_SOURCE,)),
    )[0]
    source = SourceDocument(
        source_id="policy_source",
        content="The policy requires every substantive claim to cite evidence.",
    )
    return build_generation_request(rubric, blueprint, (source,))


def _payload(request) -> dict:
    """Return an item conforming to the request's exact output schema."""
    response_format = request.blueprint.response_format
    options: list[dict[str, str]] = []
    if response_format is ResponseFormat.CONSTRUCTED_RESPONSE:
        answer_key = {
            "reference_response": "The response should cite the policy.",
            "accepted_variants": ["Cite the supplied policy."],
            "rationale": "The construct requires source-grounded support.",
        }
    elif response_format is ResponseFormat.SELECTED_RESPONSE:
        options = [
            {"option_id": "option_alpha", "text": "Supported"},
            {"option_id": "option_beta", "text": "Unsupported"},
        ]
        answer_key = {
            "option_ids": ["option_alpha"],
            "rationale": "Only option alpha is source-supported.",
        }
    elif response_format is ResponseFormat.BINARY_JUDGMENT:
        answer_key = {
            "value": True,
            "rationale": "The cited statement occurs in the source.",
        }
    elif response_format is ResponseFormat.ORDINAL_RATING:
        answer_key = {
            "score": 2,
            "rationale": "All substantive claims are supported.",
        }
    else:
        options = [
            {"option_id": "response_alpha", "text": "Response A"},
            {"option_id": "response_beta", "text": "Response B"},
        ]
        answer_key = {
            "outcome": "left_option",
            "preferred_option_id": "response_alpha",
            "rationale": "Response A is better grounded.",
        }
    contract = request.contract
    return {
        "blueprint_id": request.blueprint.blueprint_id,
        "blueprint_handle": contract["blueprint"]["blueprint_handle"],
        "blueprint_fingerprint": request.blueprint.blueprint_fingerprint,
        "rubric_id": request.blueprint.rubric_id,
        "rubric_version": request.blueprint.rubric_version,
        "rubric_fingerprint": request.blueprint.rubric_fingerprint,
        "item_id": "generated_item_001",
        "stem": "Evaluate the response against the supplied evidence.",
        "stimulus": ["Claims require evidence."],
        "response_format": response_format.value,
        "options": options,
        "answer_key": answer_key,
        "scoring_guide": [
            {"score": 0, "evidence": "No support.", "rationale": "Unsupported."},
            {"score": 1, "evidence": "Some support.", "rationale": "Partial."},
            {"score": 2, "evidence": "Full support.", "rationale": "Complete."},
        ],
        "rubric_alignment": [
            {"score": 0, "observable_indicators": ["unsupported"]},
            {"score": 1, "observable_indicators": ["partial"]},
            {"score": 2, "observable_indicators": ["complete"]},
        ],
        "source_attributions": [
            {
                "source_id": "policy_source",
                "evidence_span": "requires every substantive claim to cite evidence",
            }
        ],
        "safety_notes": [],
    }


@pytest.mark.parametrize("response_format", tuple(ResponseFormat))
def test_every_contract_conforming_answer_key_crosses_candidate_boundary(response_format):
    """The parser accepts exactly the typed answer-key shape emitted by its contract."""
    request = _request(response_format)
    payload = _payload(request)
    candidate = parse_generated_item_candidate(
        json.dumps(payload, ensure_ascii=False),
        request,
    )
    assert candidate.to_dict()["answer_key"] == payload["answer_key"]
    assert candidate.blueprint_handle == payload["blueprint_handle"]
    assert candidate.contract_fingerprint == request.contract_fingerprint


def test_contract_schema_and_parser_require_the_same_provenance_fields():
    """Every immutable schema constant is required and checked by the parser."""
    request = _request(ResponseFormat.ORDINAL_RATING)
    schema = request.contract["output_schema"]
    provenance = set(schema["allOf"][0]["required"])
    assert provenance == {
        "blueprint_id",
        "blueprint_handle",
        "blueprint_fingerprint",
        "rubric_id",
        "rubric_version",
        "rubric_fingerprint",
    }
    for field in sorted(provenance):
        payload = _payload(request)
        payload[field] = "0" * 64 if field.endswith("fingerprint") else "wrong_value"
        with pytest.raises(CandidateValidationError) as error:
            parse_generated_item_candidate(json.dumps(payload), request)
        assert error.value.code == "provenance_mismatch"


def test_selected_answer_key_rejects_undeclared_option_ids():
    """Typed selected-response keys may reference only supplied option identifiers."""
    request = _request(ResponseFormat.SELECTED_RESPONSE)
    payload = _payload(request)
    payload["answer_key"]["option_ids"] = ["missing_option"]
    with pytest.raises(CandidateValidationError) as error:
        parse_generated_item_candidate(json.dumps(payload), request)
    assert error.value.code == "invalid_answer_key"
