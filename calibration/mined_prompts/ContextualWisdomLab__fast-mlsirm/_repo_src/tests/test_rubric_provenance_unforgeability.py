"""End-to-end provenance mutation contracts for generated rubric items."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from fast_mlsirm.rubric import (
    BlueprintPlan,
    DifficultyBand,
    EvidenceMode,
    GenerationExecution,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    SourceDocument,
    build_generation_request,
    compile_item_blueprints,
    parse_generated_item_candidate,
)


def _request():
    """Return one content-addressed ordinal generation request."""
    rubric = RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which claims are supported.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(0, "unsupported", "No support.", ("unsupported",)),
            RubricLevel(1, "supported", "Supported.", ("supported",)),
        ),
        task_families=("claim_verification",),
        evidence_requirements=("Quote the supporting source span.",),
        rubric_version="1.0.0",
    )
    blueprint = compile_item_blueprints(
        rubric,
        BlueprintPlan(
            difficulty_bands=(DifficultyBand.MEDIUM,),
            evidence_modes=(EvidenceMode.SINGLE_SOURCE,),
            items_per_cell=1,
            seed=17,
        ),
    )[0]
    source = SourceDocument(
        source_id="policy_source",
        content="The policy requires every substantive claim to cite evidence.",
    )
    return build_generation_request(rubric, blueprint, (source,))


def _payload(request) -> dict[str, object]:
    """Return one valid provider payload bound to ``request``."""
    contract = request.contract
    return {
        "blueprint_id": request.blueprint.blueprint_id,
        "blueprint_handle": contract["blueprint"]["blueprint_handle"],
        "blueprint_fingerprint": request.blueprint.blueprint_fingerprint,
        "rubric_id": request.blueprint.rubric_id,
        "rubric_version": request.blueprint.rubric_version,
        "rubric_fingerprint": request.blueprint.rubric_fingerprint,
        "item_id": "generated_item_001",
        "stem": "Rate whether the response is supported.",
        "stimulus": ["Claims require evidence."],
        "response_format": "ordinal_rating",
        "options": [],
        "answer_key": {
            "score": 1,
            "rationale": "The source explicitly supports the response.",
        },
        "scoring_guide": [
            {"score": 0, "evidence": "No cited support.", "rationale": "Unsupported."},
            {"score": 1, "evidence": "Exact cited support.", "rationale": "Supported."},
        ],
        "rubric_alignment": [
            {"score": 0, "observable_indicators": ["unsupported"]},
            {"score": 1, "observable_indicators": ["supported"]},
        ],
        "source_attributions": [
            {
                "source_id": "policy_source",
                "evidence_span": "requires every substantive claim to cite evidence",
            }
        ],
        "safety_notes": [],
    }


def _candidate(request):
    """Parse one candidate through the supported validation boundary."""
    return parse_generated_item_candidate(
        json.dumps(_payload(request), ensure_ascii=False),
        request,
    )


def _sha256_json(value: object) -> str:
    """Return the repository's compact sorted canonical JSON digest."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_request_recomputes_contract_fingerprint_from_canonical_contract_body():
    """Changing one schema byte while retaining claimed digests must fail closed."""
    request = _request()
    forged_contract = request.contract
    forged_contract["output_schema"]["properties"]["stem"]["maxLength"] += 1
    with pytest.raises(ValueError, match="contract_fingerprint"):
        replace(
            request,
            contract_json=json.dumps(
                forged_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )


def test_direct_candidate_construction_cannot_change_validated_provenance():
    """A parser-created candidate cannot be replaced with a forged request proof."""
    candidate = _candidate(_request())
    with pytest.raises(ValueError, match="candidate|provenance|request_fingerprint"):
        replace(candidate, request_fingerprint="0" * 64)


def test_execution_rejects_forged_candidate_even_with_recomputed_display_id():
    """Rehashing a forged candidate and execution cannot manufacture audit proof."""
    request = _request()
    candidate = _candidate(request)
    forged_candidate = object.__new__(candidate.__class__)
    for field_name, field_value in candidate.__dict__.items():
        object.__setattr__(forged_candidate, field_name, field_value)
    object.__setattr__(forged_candidate, "request_fingerprint", "0" * 64)

    with pytest.raises(ValueError, match="provenance does not match validated content"):
        _ = forged_candidate.candidate_fingerprint

    attacker_candidate_fingerprint = (
        forged_candidate._computed_candidate_fingerprint()
    )
    raw_response_digest = hashlib.sha256(b"provider response").hexdigest()
    execution_payload = {
        "schema_version": candidate.schema_version,
        "request_id": candidate.request_id,
        "request_fingerprint": forged_candidate.request_fingerprint,
        "contract_id": candidate.contract_id,
        "contract_fingerprint": candidate.contract_fingerprint,
        "provider_id": "fixture_provider",
        "model_id": "fixture_model",
        "candidate_fingerprint": attacker_candidate_fingerprint,
        "raw_response_digest": raw_response_digest,
    }
    forged_execution_id = f"generation_execution_{_sha256_json(execution_payload)[:16]}"

    with pytest.raises(ValueError, match="candidate|provenance|request_fingerprint"):
        GenerationExecution(
            execution_id=forged_execution_id,
            request_id=request.request_id,
            contract_id=request.contract_id,
            provider_id="fixture_provider",
            model_id="fixture_model",
            candidate=forged_candidate,
            raw_response_digest=raw_response_digest,
        )
