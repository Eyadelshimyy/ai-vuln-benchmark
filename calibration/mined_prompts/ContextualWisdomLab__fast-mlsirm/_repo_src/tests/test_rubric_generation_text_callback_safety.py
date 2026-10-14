"""Callback-safety regressions for public rubric-generation text boundaries."""

from __future__ import annotations

import pytest

from fast_mlsirm.rubric import (
    BlueprintPlan,
    DifficultyBand,
    EvidenceMode,
    GenerationProviderError,
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


class _HostileText(str):
    """String subclass that records every caller-overridable text callback."""

    callbacks = 0

    @classmethod
    def reset(cls) -> None:
        """Reset callback accounting between public-boundary checks."""
        cls.callbacks = 0

    @classmethod
    def _called(cls, name: str):
        cls.callbacks += 1
        raise AssertionError(f"hostile {name} callback executed")

    def strip(self, *args, **kwargs):
        return type(self)._called("strip")

    def __len__(self):
        return type(self)._called("len")

    def __iter__(self):
        return type(self)._called("iter")

    def encode(self, *args, **kwargs):
        return type(self)._called("encode")


def _request():
    """Build one ordinary single-source request for parser/provider tests."""
    rubric = RubricSpecification(
        rubric_id="generation_callback_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which substantive claims are supported.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(0, "unsupported", "No support.", ("unsupported claim",)),
            RubricLevel(1, "supported", "Supported.", ("supported claim",)),
        ),
        task_families=("claim_verification",),
        evidence_requirements=("Cite the supplied source.",),
        prohibited_patterns=("Do not invent support.",),
        locale="en-US",
    )
    blueprint = compile_item_blueprints(
        rubric,
        BlueprintPlan(
            difficulty_bands=(DifficultyBand.MEDIUM,),
            evidence_modes=(EvidenceMode.SINGLE_SOURCE,),
            items_per_cell=1,
            seed=11,
        ),
    )[0]
    source = SourceDocument(
        "callback_source",
        "The policy requires evidence for every substantive claim.",
        "text/plain",
        "en-US",
    )
    return build_generation_request(rubric, blueprint, (source,))


def test_source_document_rejects_string_subclass_before_text_callbacks():
    """Source admission must not invoke caller-controlled text operations."""
    _HostileText.reset()
    with pytest.raises(ValueError, match="content must be a string"):
        SourceDocument(
            "callback_source",
            _HostileText("hostile source content"),
            "text/plain",
            "en-US",
        )
    assert _HostileText.callbacks == 0


def test_candidate_parser_rejects_string_subclass_before_length_or_iteration():
    """Direct parser use must reject hostile JSON text before inspecting it."""
    request = _request()
    _HostileText.reset()
    with pytest.raises(TypeError, match="raw_json must be a string"):
        parse_generated_item_candidate(_HostileText("{}"), request)
    assert _HostileText.callbacks == 0


def test_fixture_provider_rejects_string_subclass_before_storage():
    """Offline fixtures must preserve the same exact-text admission contract."""
    _HostileText.reset()
    with pytest.raises(ValueError, match="response_text must be a string"):
        StaticFixtureProvider(
            provider_id="fixture_provider",
            model_id="fixture_model",
            response_text=_HostileText("{}"),
        )
    assert _HostileText.callbacks == 0


def test_execute_generation_rejects_hostile_provider_text_before_parser_callbacks():
    """Provider-returned text subclasses must fail at the provider boundary."""
    request = _request()

    class HostileProvider:
        provider_id = "hostile_provider"
        model_id = "hostile_model"

        def generate(self, request):
            return _HostileText("{}")

    _HostileText.reset()
    with pytest.raises(GenerationProviderError) as error:
        execute_generation(HostileProvider(), request)
    assert error.value.code == "invalid_provider_output"
    assert _HostileText.callbacks == 0
