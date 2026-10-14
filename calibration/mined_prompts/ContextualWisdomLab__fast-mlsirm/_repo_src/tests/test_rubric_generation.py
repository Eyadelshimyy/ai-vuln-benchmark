"""Behavioral contracts for governed rubric item generation."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.rubric import (
    BlueprintPlan,
    CandidateValidationError,
    DifficultyBand,
    EvidenceMode,
    GeneratedItemCandidate,
    GenerationProviderError,
    ItemGenerationProvider,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    SourceDocument,
    StaticFixtureProvider,
    build_generation_request,
    compile_item_blueprints,
    execute_generation,
    parse_generated_item_candidate,
)


def _rubric(response_format=ResponseFormat.ORDINAL_RATING):
    """Return a reusable three-level groundedness rubric."""
    return RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which substantive claims are supported.",
        response_format=response_format,
        levels=(
            RubricLevel(0, "unsupported", "No support.", ("unsupported claim",)),
            RubricLevel(1, "partial_support", "Partial support.", ("mixed support",)),
            RubricLevel(2, "full_support", "Full support.", ("complete support",)),
        ),
        task_families=("claim_verification",),
        evidence_requirements=("Quote the supporting source span.",),
        prohibited_patterns=("Do not invent support.",),
        locale="en-US",
    )


def _source(source_id="policy_source", content=None):
    """Return one valid source document."""
    return SourceDocument(
        source_id,
        content
        or "The policy requires every substantive claim to cite evidence.",
        "text/plain",
        "en-US",
    )


def _request(response_format=ResponseFormat.ORDINAL_RATING, mode=EvidenceMode.SINGLE_SOURCE):
    """Build one request with evidence-mode-valid sources."""
    rubric = _rubric(response_format)
    blueprint = compile_item_blueprints(
        rubric,
        BlueprintPlan(
            difficulty_bands=(DifficultyBand.MEDIUM,),
            evidence_modes=(mode,),
            items_per_cell=1,
            seed=7,
        ),
    )[0]
    if mode is EvidenceMode.CLOSED_BOOK:
        sources = ()
    elif mode in {EvidenceMode.MULTI_SOURCE, EvidenceMode.ADVERSARIAL_CONTEXT}:
        sources = (_source(), _source("secondary_source", "Corroborating evidence."))
    else:
        sources = (_source(),)
    return build_generation_request(rubric, blueprint, sources)


def _provenance(request):
    """Return exact constants required by the generation contract."""
    return {
        "blueprint_id": request.blueprint.blueprint_id,
        "blueprint_handle": request.contract["blueprint"]["blueprint_handle"],
        "blueprint_fingerprint": request.blueprint.blueprint_fingerprint,
        "rubric_id": request.blueprint.rubric_id,
        "rubric_version": request.blueprint.rubric_version,
        "rubric_fingerprint": request.blueprint.rubric_fingerprint,
    }


def _payload(request, *, closed_book=False):
    """Return one valid candidate for the request's response format."""
    response_format = request.blueprint.response_format
    options = []
    if response_format is ResponseFormat.CONSTRUCTED_RESPONSE:
        answer_key = {
            "reference_response": "A supported response cites the policy.",
            "accepted_variants": ["Cite the supplied policy."],
            "rationale": "The rubric requires grounding.",
        }
    elif response_format is ResponseFormat.SELECTED_RESPONSE:
        options = [
            {"option_id": "option_alpha", "text": "Supported"},
            {"option_id": "option_beta", "text": "Unsupported"},
        ]
        answer_key = {
            "option_ids": ["option_alpha"],
            "rationale": "Only option alpha is supported.",
        }
    elif response_format is ResponseFormat.BINARY_JUDGMENT:
        answer_key = {"value": True, "rationale": "The source supports the claim."}
    elif response_format is ResponseFormat.ORDINAL_RATING:
        answer_key = {"score": 2, "rationale": "All claims are supported."}
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
    return {
        **_provenance(request),
        "item_id": "generated_item_001",
        "stem": "Judge whether the response is source-supported.",
        "stimulus": ["The response says claims require evidence."],
        "response_format": response_format.value,
        "options": options,
        "answer_key": answer_key,
        "scoring_guide": [
            {"score": 0, "evidence": "No support.", "rationale": "Unsupported."},
            {"score": 1, "evidence": "Some support.", "rationale": "Partial."},
            {"score": 2, "evidence": "Full support.", "rationale": "Complete."},
        ],
        "rubric_alignment": [
            {"score": 0, "observable_indicators": ["unsupported claim"]},
            {"score": 1, "observable_indicators": ["mixed support"]},
            {"score": 2, "observable_indicators": ["complete support"]},
        ],
        "source_attributions": []
        if closed_book
        else [
            {
                "source_id": "policy_source",
                "evidence_span": "requires every substantive claim to cite evidence",
            }
        ],
        "safety_notes": [],
    }


def _raw(request, *, closed_book=False):
    """Return canonical test JSON for one request."""
    return json.dumps(_payload(request, closed_book=closed_book), ensure_ascii=False)


def test_source_and_request_provenance_is_deterministic_and_redacted():
    """Exact content changes identities without leaking source text into metadata."""
    first = _request()
    second = _request()
    assert first == second
    assert first.request_id == second.request_id
    assert len(first.sources[0].content_digest) == 64
    assert first.sources[0].content not in json.dumps(first.to_metadata_dict())

    rubric = _rubric()
    blueprint = compile_item_blueprints(
        rubric,
        BlueprintPlan(evidence_modes=(EvidenceMode.SINGLE_SOURCE,)),
    )[0]
    changed = build_generation_request(
        rubric,
        blueprint,
        (_source(content="A changed source requires two citations."),),
    )
    assert changed.request_id != first.request_id


@pytest.mark.parametrize("mode", tuple(EvidenceMode))
def test_each_evidence_mode_accepts_its_declared_source_cardinality(mode):
    """Every evidence mode can be represented by a valid request."""
    assert _request(mode=mode).blueprint.evidence_mode is mode


@pytest.mark.parametrize("response_format", tuple(ResponseFormat))
def test_every_response_format_crosses_the_same_candidate_boundary(response_format):
    """Typed answer-key formats preserve provenance and deterministic fingerprints."""
    request = _request(response_format)
    candidate = parse_generated_item_candidate(_raw(request), request)
    assert isinstance(candidate, GeneratedItemCandidate)
    assert candidate.response_format is response_format
    assert len(candidate.candidate_fingerprint) == 64
    for field, expected in _provenance(request).items():
        assert getattr(candidate, field) == expected


def test_closed_book_candidate_has_no_source_attribution():
    """Closed-book candidates remain source-free across the parser boundary."""
    request = _request(ResponseFormat.CONSTRUCTED_RESPONSE, EvidenceMode.CLOSED_BOOK)
    candidate = parse_generated_item_candidate(_raw(request, closed_book=True), request)
    assert candidate.source_attributions == ()


def test_fixture_provider_executes_once_and_returns_redacted_provenance():
    """The offline adapter is called once and raw content stays out of results."""
    request = _request()
    provider = StaticFixtureProvider(
        provider_id="fixture_provider",
        model_id="fixture_model",
        response_text=_raw(request),
    )
    assert isinstance(provider, ItemGenerationProvider)
    execution = execute_generation(provider, request)
    assert provider.call_count == 1
    assert execution.request_id == request.request_id
    assert len(execution.raw_response_digest) == 64
    assert _source().content not in json.dumps(execution.to_dict())


def test_duplicate_keys_and_nonfinite_numbers_are_rejected_without_echo():
    """Unsafe JSON interoperability cases use stable redacted codes."""
    request = _request()
    duplicate = _raw(request)[:-1] + ',"item_id":"other_item"}'
    with pytest.raises(CandidateValidationError) as error:
        parse_generated_item_candidate(duplicate, request)
    assert error.value.code == "duplicate_json_key"
    assert "other_item" not in str(error.value)

    nonfinite = _raw(request).replace('"score": 2', '"score": NaN', 1)
    with pytest.raises(CandidateValidationError) as error:
        parse_generated_item_candidate(nonfinite, request)
    assert error.value.code == "nonfinite_json_number"


def test_provider_failures_are_redacted():
    """Provider diagnostics cannot escape the trust boundary."""
    request = _request()

    class FailingProvider:
        provider_id = "failing_provider"
        model_id = "failing_model"

        def generate(self, request):
            raise RuntimeError("secret provider diagnostic")

    with pytest.raises(GenerationProviderError) as error:
        execute_generation(FailingProvider(), request)
    assert error.value.code == "provider_failure"
    assert "secret" not in str(error.value)
