"""Failure-path tests for governed rubric item generation."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from fast_mlsirm.rubric import (
    BlueprintPlan,
    CandidateValidationError,
    DifficultyBand,
    EvidenceMode,
    GenerationRequest,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    SourceDocument,
    build_generation_request,
    compile_item_blueprints,
    execute_generation,
    parse_generated_item_candidate,
)


def _rubric(response_format=ResponseFormat.ORDINAL_RATING):
    """Return a compact three-level rubric."""
    return RubricSpecification(
        "faithfulness_rubric",
        "evidence_grounding",
        "Grounded response quality.",
        response_format,
        (
            RubricLevel(0, "unsupported", "No support.", ("unsupported",)),
            RubricLevel(1, "partial_support", "Partial support.", ("partial",)),
            RubricLevel(2, "full_support", "Full support.", ("complete",)),
        ),
        ("claim_verification",),
        ("Quote evidence.",),
        ("Do not invent support.",),
        "en-US",
        "1.2.3",
    )


def _source(source_id="policy_source", content=None):
    """Return a valid source for mutation tests."""
    return SourceDocument(
        source_id,
        content
        or "The policy requires every substantive claim to cite evidence.",
        "text/plain",
        "en-US",
    )


def _sources(mode: EvidenceMode):
    """Return a valid source packet for one evidence mode."""
    if mode is EvidenceMode.CLOSED_BOOK:
        return ()
    if mode in {EvidenceMode.MULTI_SOURCE, EvidenceMode.ADVERSARIAL_CONTEXT}:
        return (_source(), _source("secondary_source", "Second evidence source."))
    return (_source(),)


def _request(
    response_format=ResponseFormat.ORDINAL_RATING,
    mode=EvidenceMode.SINGLE_SOURCE,
):
    """Return one valid request for mutation tests."""
    rubric = _rubric(response_format)
    blueprint = compile_item_blueprints(
        rubric,
        BlueprintPlan((DifficultyBand.MEDIUM,), (mode,), 1, 5),
    )[0]
    return build_generation_request(rubric, blueprint, _sources(mode))


def _options(response_format: ResponseFormat):
    """Return valid response options when the format requires them."""
    if response_format is ResponseFormat.SELECTED_RESPONSE:
        return [
            {"option_id": "option_alpha", "text": "Supported"},
            {"option_id": "option_beta", "text": "Unsupported"},
        ]
    if response_format is ResponseFormat.PAIRWISE_COMPARISON:
        return [
            {"option_id": "response_alpha", "text": "Response A"},
            {"option_id": "response_beta", "text": "Response B"},
        ]
    return []


def _answer_key(response_format: ResponseFormat):
    """Return one valid typed answer-key object."""
    if response_format is ResponseFormat.CONSTRUCTED_RESPONSE:
        return {
            "reference_response": "Supported response.",
            "accepted_variants": ["Evidence-grounded response."],
            "rationale": "The source supports the response.",
        }
    if response_format is ResponseFormat.SELECTED_RESPONSE:
        return {"option_ids": ["option_alpha"], "rationale": "Alpha is correct."}
    if response_format is ResponseFormat.BINARY_JUDGMENT:
        return {"value": True, "rationale": "The statement is supported."}
    if response_format is ResponseFormat.ORDINAL_RATING:
        return {"score": 2, "rationale": "The response is fully supported."}
    return {
        "outcome": "left_option",
        "preferred_option_id": "response_alpha",
        "rationale": "The left response is better grounded.",
    }


def _payload(request, *, closed_book=False):
    """Return one contract-ordered candidate object with immutable provenance."""
    contract = request.contract
    response_format = request.blueprint.response_format
    attributions = []
    if not closed_book:
        attributions = [
            {
                "source_id": "policy_source",
                "evidence_span": "requires every substantive claim to cite evidence",
            }
        ]
        if request.blueprint.evidence_mode is EvidenceMode.MULTI_SOURCE:
            attributions.append(
                {
                    "source_id": "secondary_source",
                    "evidence_span": "Second evidence source",
                }
            )
    return {
        "blueprint_id": request.blueprint.blueprint_id,
        "blueprint_handle": contract["blueprint"]["blueprint_handle"],
        "blueprint_fingerprint": request.blueprint.blueprint_fingerprint,
        "rubric_id": request.blueprint.rubric_id,
        "rubric_version": request.blueprint.rubric_version,
        "rubric_fingerprint": request.blueprint.rubric_fingerprint,
        "item_id": "generated_item_001",
        "stem": "Judge the evidence support.",
        "stimulus": ["Claims require evidence."],
        "response_format": response_format.value,
        "options": _options(response_format),
        "answer_key": _answer_key(response_format),
        "scoring_guide": [
            {"score": 0, "evidence": "No.", "rationale": "Unsupported."},
            {"score": 1, "evidence": "Some.", "rationale": "Partial."},
            {"score": 2, "evidence": "Full.", "rationale": "Complete."},
        ],
        "rubric_alignment": [
            {"score": 0, "observable_indicators": ["unsupported"]},
            {"score": 1, "observable_indicators": ["partial"]},
            {"score": 2, "observable_indicators": ["complete"]},
        ],
        "source_attributions": attributions,
        "safety_notes": [],
    }


def _parse(payload, request):
    """Parse one payload through the public JSON trust boundary."""
    return parse_generated_item_candidate(
        json.dumps(payload, ensure_ascii=False),
        request,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_id": "source"}, "two-or-more-token"),
        ({"source_id": 7}, "source_id must be a string"),
        ({"content": ""}, "content must not be empty"),
        ({"content": 7}, "content must be a string"),
        ({"content": "x" * 262145}, "content must contain at most"),
        ({"media_type": "application/octet-stream"}, "media_type must be one of"),
        ({"locale": "bad_locale"}, "locale must be a BCP 47-style tag"),
        ({"schema_version": "2.0"}, "schema_version must be '1.0'"),
    ],
)
def test_source_document_rejects_invalid_fields(kwargs, message):
    """Every caller-controlled source field fails closed."""
    values = {
        "source_id": "policy_source",
        "content": "Evidence text.",
        "media_type": "text/plain",
        "locale": "en-US",
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SourceDocument(**values)


@pytest.mark.parametrize(
    ("mode", "sources"),
    [
        (EvidenceMode.CLOSED_BOOK, (_source(),)),
        (EvidenceMode.SINGLE_SOURCE, ()),
        (EvidenceMode.SINGLE_SOURCE, (_source(), _source("secondary_source"))),
        (EvidenceMode.MULTI_SOURCE, (_source(),)),
        (EvidenceMode.ADVERSARIAL_CONTEXT, (_source(),)),
        (EvidenceMode.UNANSWERABLE, ()),
    ],
)
def test_request_rejects_invalid_evidence_source_cardinality(mode, sources):
    """Invalid evidence-mode source counts fail before provider invocation."""
    rubric = _rubric()
    blueprint = compile_item_blueprints(
        rubric,
        BlueprintPlan((DifficultyBand.MEDIUM,), (mode,), 1),
    )[0]
    with pytest.raises(ValueError, match="source cardinality"):
        build_generation_request(rubric, blueprint, sources)


def test_request_rejects_wrong_types_duplicates_budgets_and_rubric_replay():
    """Request construction validates all outer trust boundaries."""
    rubric = _rubric()
    blueprint = compile_item_blueprints(
        rubric,
        BlueprintPlan((DifficultyBand.MEDIUM,), (EvidenceMode.SINGLE_SOURCE,), 1),
    )[0]
    with pytest.raises(TypeError, match="rubric must be a RubricSpecification"):
        build_generation_request(object(), blueprint, (_source(),))
    with pytest.raises(TypeError, match="blueprint must be an ItemBlueprint"):
        build_generation_request(rubric, object(), (_source(),))
    with pytest.raises(ValueError, match=r"sources\[0\] must be a SourceDocument"):
        build_generation_request(rubric, blueprint, (object(),))

    multi = compile_item_blueprints(
        rubric,
        BlueprintPlan((DifficultyBand.MEDIUM,), (EvidenceMode.MULTI_SOURCE,), 1),
    )[0]
    with pytest.raises(ValueError, match="source_id values must be unique"):
        build_generation_request(rubric, multi, (_source(), _source()))
    with pytest.raises(ValueError, match="aggregate source content"):
        build_generation_request(
            rubric,
            multi,
            tuple(
                SourceDocument(f"source_document_{index}", "x" * 40000)
                for index in range(27)
            ),
        )
    with pytest.raises(ValueError, match="sources must contain at most 32"):
        build_generation_request(
            rubric,
            multi,
            tuple(
                SourceDocument(f"source_document_{index}", "x")
                for index in range(33)
            ),
        )

    different = replace(rubric, rubric_id="different_rubric")
    with pytest.raises(ValueError, match="rubric"):
        build_generation_request(different, blueprint, (_source(),))


def test_direct_request_constructor_rejects_forged_identity_and_seed():
    """Direct dataclass construction cannot bypass the content-addressed request."""
    request = _request()
    with pytest.raises(ValueError, match="request_id"):
        replace(request, request_id="generation_request_forged")
    with pytest.raises(ValueError, match="generation_seed"):
        replace(request, generation_seed=request.generation_seed + 1)
    with pytest.raises(ValueError, match="contract_id"):
        replace(request, contract_id="generation_contract_forged")
    forged = json.loads(request.contract_json)
    forged["blueprint"]["blueprint_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="blueprint_fingerprint"):
        replace(request, contract_json=json.dumps(forged))


@pytest.mark.parametrize(
    ("raw_json", "code"),
    [
        ("[]", "top_level_type"),
        ("not-json", "invalid_json"),
        ("", "invalid_json"),
        ("{}", "missing_field"),
    ],
)
def test_parser_rejects_invalid_json_envelopes(raw_json, code):
    """Malformed envelopes return stable codes without echoing non-empty input."""
    with pytest.raises(CandidateValidationError) as error:
        parse_generated_item_candidate(raw_json, _request())
    assert error.value.code == code
    if raw_json:
        assert raw_json not in str(error.value)


def test_parser_rejects_wrong_type_oversized_and_deep_json():
    """Provider output type, size, and nesting are checked before field parsing."""
    with pytest.raises(TypeError, match="raw_json must be a string"):
        parse_generated_item_candidate({}, _request())
    with pytest.raises(CandidateValidationError) as size_error:
        parse_generated_item_candidate("x" * 262145, _request())
    assert size_error.value.code == "raw_json_too_large"

    nested: object = "leaf"
    for _ in range(40):
        nested = [nested]
    with pytest.raises(CandidateValidationError) as depth_error:
        parse_generated_item_candidate(json.dumps(nested), _request())
    assert depth_error.value.code == "json_too_deep"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("blueprint_id", "item_blueprint_wrong"),
        ("blueprint_handle", "item_blueprint_wrong"),
        ("blueprint_fingerprint", "0" * 64),
        ("rubric_id", "different_rubric"),
        ("rubric_version", "9.9.9"),
        ("rubric_fingerprint", "f" * 64),
    ],
)
def test_provider_provenance_replay_fails_closed(field, bad_value):
    """A structurally valid item from another immutable contract is rejected."""
    request = _request()
    payload = _payload(request)
    payload[field] = bad_value
    with pytest.raises(CandidateValidationError) as error:
        _parse(payload, request)
    assert error.value.code == "provenance_mismatch"
    assert bad_value not in str(error.value)


@pytest.mark.parametrize(
    ("field", "mutation", "code"),
    [
        ("unknown_field", "add", "unknown_field"),
        ("stem", "delete", "missing_field"),
        ("item_id", 7, "invalid_type"),
        ("item_id", "bad", "invalid_identifier"),
        ("stem", "", "invalid_text"),
        ("stem", "x" * 8193, "text_too_large"),
        ("stimulus", "not-a-list", "invalid_type"),
        ("response_format", 7, "invalid_type"),
        ("response_format", "unknown_format", "invalid_response_format"),
        ("response_format", "selected_response", "response_format_mismatch"),
        ("options", "not-a-list", "invalid_type"),
        ("scoring_guide", "not-a-list", "invalid_type"),
        ("rubric_alignment", "not-a-list", "invalid_type"),
        ("source_attributions", "not-a-list", "invalid_type"),
        ("safety_notes", "not-a-list", "invalid_type"),
    ],
)
def test_top_level_candidate_fields_fail_closed(field, mutation, code):
    """Every top-level field is required, typed, bounded, and allowlisted."""
    request = _request()
    payload = _payload(request)
    if mutation == "add":
        payload[field] = "unexpected"
    elif mutation == "delete":
        del payload[field]
    else:
        payload[field] = mutation
    with pytest.raises(CandidateValidationError) as error:
        _parse(payload, request)
    assert error.value.code == code
    assert "unexpected" not in str(error.value)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("stimulus", [""], "invalid_text"),
        ("stimulus", [1], "invalid_type"),
        ("stimulus", ["x"] * 33, "collection_too_large"),
        ("safety_notes", [""], "invalid_text"),
        ("safety_notes", [1], "invalid_type"),
        ("safety_notes", ["same", "same"], "duplicate_value"),
    ],
)
def test_candidate_text_collections_are_bounded_unique_strings(field, value, code):
    """Stimulus and safety-note arrays enforce their text contracts."""
    request = _request()
    payload = _payload(request)
    payload[field] = value
    with pytest.raises(CandidateValidationError) as error:
        _parse(payload, request)
    assert error.value.code == code


@pytest.mark.parametrize(
    ("field", "entries", "code"),
    [
        (
            "scoring_guide",
            [
                {"score": 0, "evidence": "No.", "rationale": "No."},
                {"score": 0, "evidence": "Again.", "rationale": "Again."},
                {"score": 2, "evidence": "Full.", "rationale": "Full."},
            ],
            "duplicate_score",
        ),
        (
            "scoring_guide",
            [
                {"score": 0, "evidence": "No.", "rationale": "No."},
                {"score": 3, "evidence": "Other.", "rationale": "Other."},
                {"score": 2, "evidence": "Full.", "rationale": "Full."},
            ],
            "score_coverage",
        ),
        (
            "scoring_guide",
            [
                {"score": 1, "evidence": "Some.", "rationale": "Some."},
                {"score": 0, "evidence": "No.", "rationale": "No."},
                {"score": 2, "evidence": "Full.", "rationale": "Full."},
            ],
            "score_order",
        ),
        (
            "rubric_alignment",
            [
                {"score": 1, "observable_indicators": ["some"]},
                {"score": 0, "observable_indicators": ["none"]},
                {"score": 2, "observable_indicators": ["full"]},
            ],
            "score_order",
        ),
        (
            "rubric_alignment",
            [
                {"score": 0, "observable_indicators": []},
                {"score": 1, "observable_indicators": ["some"]},
                {"score": 2, "observable_indicators": ["full"]},
            ],
            "collection_too_small",
        ),
    ],
)
def test_score_level_arrays_require_exact_coverage_and_order(field, entries, code):
    """Score-level arrays cannot duplicate, omit, reorder, or empty their evidence."""
    request = _request()
    payload = _payload(request)
    payload[field] = entries
    with pytest.raises(CandidateValidationError) as error:
        _parse(payload, request)
    assert error.value.code == code


@pytest.mark.parametrize(
    ("attributions", "code"),
    [
        ([], "source_attribution_required"),
        ([{"source_id": "bad", "evidence_span": "evidence"}], "invalid_identifier"),
        (
            [{"source_id": "missing_source", "evidence_span": "evidence"}],
            "unknown_source",
        ),
        (
            [{"source_id": "policy_source", "evidence_span": "not present"}],
            "evidence_span_not_found",
        ),
        (
            [
                {"source_id": "policy_source", "evidence_span": "requires every"},
                {"source_id": "policy_source", "evidence_span": "requires every"},
            ],
            "duplicate_attribution",
        ),
    ],
)
def test_source_attributions_are_grounded_and_unique(attributions, code):
    """Attribution ids and evidence spans resolve to the supplied packet."""
    request = _request()
    payload = _payload(request)
    payload["source_attributions"] = attributions
    with pytest.raises(CandidateValidationError) as error:
        _parse(payload, request)
    assert error.value.code == code


def test_multi_source_candidate_must_use_two_distinct_sources():
    """A multi-source blueprint cannot be satisfied by citing only one source."""
    request = _request(mode=EvidenceMode.MULTI_SOURCE)
    payload = _payload(request)
    payload["source_attributions"] = payload["source_attributions"][:1]
    with pytest.raises(CandidateValidationError) as error:
        _parse(payload, request)
    assert error.value.code == "source_cardinality"


def test_closed_book_rejects_source_attribution():
    """Closed-book candidates cannot claim source-backed provenance."""
    request = _request(ResponseFormat.CONSTRUCTED_RESPONSE, EvidenceMode.CLOSED_BOOK)
    payload = _payload(request)
    payload["source_attributions"] = [
        {"source_id": "policy_source", "evidence_span": "fabricated"}
    ]
    with pytest.raises(CandidateValidationError) as error:
        _parse(payload, request)
    assert error.value.code == "closed_book_attribution"


@pytest.mark.parametrize(
    ("response_format", "options", "answer_key", "code"),
    [
        (
            ResponseFormat.CONSTRUCTED_RESPONSE,
            [{"option_id": "option_alpha", "text": "A"}],
            {
                "reference_response": "A",
                "accepted_variants": [],
                "rationale": "R",
            },
            "options_not_allowed",
        ),
        (ResponseFormat.CONSTRUCTED_RESPONSE, [], 7, "invalid_type"),
        (ResponseFormat.SELECTED_RESPONSE, [], _answer_key(ResponseFormat.SELECTED_RESPONSE), "option_count"),
        (
            ResponseFormat.SELECTED_RESPONSE,
            [
                {"option_id": "option_alpha", "text": "A"},
                {"option_id": "option_alpha", "text": "B"},
            ],
            _answer_key(ResponseFormat.SELECTED_RESPONSE),
            "duplicate_option_id",
        ),
        (
            ResponseFormat.SELECTED_RESPONSE,
            _options(ResponseFormat.SELECTED_RESPONSE),
            {"option_ids": ["missing_option"], "rationale": "R"},
            "invalid_answer_key",
        ),
        (
            ResponseFormat.BINARY_JUDGMENT,
            _options(ResponseFormat.SELECTED_RESPONSE),
            _answer_key(ResponseFormat.BINARY_JUDGMENT),
            "options_not_allowed",
        ),
        (
            ResponseFormat.BINARY_JUDGMENT,
            [],
            {"value": 1, "rationale": "R"},
            "invalid_type",
        ),
        (
            ResponseFormat.ORDINAL_RATING,
            [],
            {"score": True, "rationale": "R"},
            "invalid_type",
        ),
        (
            ResponseFormat.ORDINAL_RATING,
            [],
            {"score": 9, "rationale": "R"},
            "invalid_answer_key",
        ),
        (
            ResponseFormat.PAIRWISE_COMPARISON,
            [{"option_id": "response_alpha", "text": "A"}],
            _answer_key(ResponseFormat.PAIRWISE_COMPARISON),
            "option_count",
        ),
        (
            ResponseFormat.PAIRWISE_COMPARISON,
            _options(ResponseFormat.PAIRWISE_COMPARISON),
            {
                "outcome": "tie",
                "preferred_option_id": "response_alpha",
                "rationale": "R",
            },
            "invalid_answer_key",
        ),
        (
            ResponseFormat.PAIRWISE_COMPARISON,
            _options(ResponseFormat.PAIRWISE_COMPARISON),
            {
                "outcome": "right_option",
                "preferred_option_id": "response_alpha",
                "rationale": "R",
            },
            "invalid_answer_key",
        ),
    ],
)
def test_response_format_specific_structure_fails_closed(
    response_format,
    options,
    answer_key,
    code,
):
    """Response formats enforce their declared authoring invariants."""
    request = _request(response_format)
    payload = _payload(request)
    payload["options"] = options
    payload["answer_key"] = answer_key
    with pytest.raises(CandidateValidationError) as error:
        _parse(payload, request)
    assert error.value.code == code


def test_pairwise_tie_with_null_preference_is_valid():
    """Pairwise equivalence has an explicit outcome and no fabricated winner."""
    request = _request(ResponseFormat.PAIRWISE_COMPARISON)
    payload = _payload(request)
    payload["answer_key"] = {
        "outcome": "tie",
        "preferred_option_id": None,
        "rationale": "The responses are equivalent.",
    }
    candidate = _parse(payload, request)
    assert candidate.answer_key.to_dict()["preferred_option_id"] is None


def test_executor_rejects_non_protocol_invalid_metadata_and_wrong_request():
    """Execution requires the explicit provider protocol and stable metadata."""
    with pytest.raises(TypeError, match="provider must implement ItemGenerationProvider"):
        execute_generation(object(), _request())

    class BadMetadataProvider:
        provider_id = "bad"
        model_id = "valid_model"

        def generate(self, request):
            return json.dumps(_payload(request))

    with pytest.raises(ValueError, match="provider_id"):
        execute_generation(BadMetadataProvider(), _request())
    with pytest.raises(TypeError, match="request must be a GenerationRequest"):
        execute_generation(BadMetadataProvider(), object())
