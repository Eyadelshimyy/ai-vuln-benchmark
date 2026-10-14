"""Branch-complete failure-path tests for rubric-centered item authoring."""

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
    canonical_generation_contract,
    compile_item_blueprints,
    render_generation_prompt,
)


def _levels() -> tuple[RubricLevel, ...]:
    """Return the minimal valid two-level scale."""
    return (
        RubricLevel(0, "unsupported", "No support.", ("no support",)),
        RubricLevel(1, "full_support", "Full support.", ("full support",)),
    )


def _rubric() -> RubricSpecification:
    """Return a minimal valid rubric for failure-path mutation."""
    return RubricSpecification(
        "faithfulness_rubric",
        "evidence_grounding",
        "Grounded response quality.",
        ResponseFormat.ORDINAL_RATING,
        _levels(),
        ("claim_verification",),
        ("Quote source evidence.",),
        ("Do not invent support.",),
        "en-US",
    )


@pytest.mark.parametrize("value", ["one string", b"bytes", object()])
def test_collection_fields_reject_scalar_and_non_iterable_values(value):
    """Collection validation covers string, bytes, and non-iterable callers."""
    with pytest.raises(
        ValueError,
        match="observable_indicators must be a collection",
    ):
        RubricLevel(0, "valid_label", "Valid descriptor.", value)


def test_enum_collections_reject_values_above_the_vocabulary_size():
    """A caller cannot bypass enum-cell bounds by repeating a fourth value."""
    with pytest.raises(ValueError, match="difficulty_bands must contain at most 3"):
        BlueprintPlan(difficulty_bands=("easy", "medium", "hard", "easy"))


def test_compiler_and_contract_type_boundaries_fail_closed():
    """Compiler and contract entrypoints reject unrelated object types."""
    rubric = _rubric()
    blueprint = compile_item_blueprints(rubric)[0]
    with pytest.raises(TypeError, match="rubric must be a RubricSpecification"):
        compile_item_blueprints(object())
    with pytest.raises(TypeError, match="plan must be a BlueprintPlan"):
        compile_item_blueprints(rubric, object())
    with pytest.raises(TypeError, match="rubric must be a RubricSpecification"):
        build_generation_contract(object(), blueprint)
    with pytest.raises(TypeError, match="blueprint must be an ItemBlueprint"):
        build_generation_contract(rubric, object())


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("rubric_id", "other_rubric", "rubric_id"),
        ("rubric_fingerprint", "0" * 64, "rubric_fingerprint"),
        ("response_format", ResponseFormat.BINARY_JUDGMENT, "response_format"),
        ("scoring_levels", (0, 1, 2), "scoring_levels"),
        ("evidence_requirements", ("Different requirement.",), "evidence_requirements"),
        ("prohibited_patterns", ("Different prohibition.",), "prohibited_patterns"),
    ],
)
def test_every_contract_compatibility_guard_rejects_replay(field, value, match):
    """Each exact-rubric compatibility field independently blocks replay."""
    rubric = _rubric()
    blueprint = compile_item_blueprints(rubric)[0]
    with pytest.raises(ValueError, match=match):
        build_generation_contract(rubric, replace(blueprint, **{field: value}))


def test_rubric_level_failure_paths_are_complete():
    """Level validation rejects invalid score, text, indicator, and size shapes."""
    with pytest.raises(ValueError, match="score must be between"):
        RubricLevel(-1, "valid_label", "Valid descriptor.", ("indicator",))
    with pytest.raises(ValueError, match="label must not be empty"):
        RubricLevel(0, "", "Valid descriptor.", ("indicator",))
    with pytest.raises(ValueError, match="descriptor must contain at most"):
        RubricLevel(0, "valid_label", "x" * 8193, ("indicator",))
    with pytest.raises(ValueError, match="at least 1 value"):
        RubricLevel(0, "valid_label", "Valid descriptor.", ())
    with pytest.raises(ValueError, match="must not contain duplicates"):
        RubricLevel(0, "valid_label", "Valid descriptor.", ("same", "same"))
    with pytest.raises(ValueError, match=r"observable_indicators\[0\] must be a string"):
        RubricLevel(0, "valid_label", "Valid descriptor.", (1,))
    with pytest.raises(ValueError, match="at most 32"):
        RubricLevel(
            0,
            "valid_label",
            "Valid descriptor.",
            tuple(str(index) for index in range(33)),
        )


def test_rubric_specification_failure_paths_are_complete():
    """Rubric validation rejects malformed identity, scale, collections, and metadata."""
    with pytest.raises(ValueError, match="two-or-more-token"):
        RubricSpecification(
            "bad",
            "evidence_grounding",
            "Grounded.",
            ResponseFormat.ORDINAL_RATING,
            _levels(),
            ("claim_verification",),
            ("Quote source evidence.",),
        )
    with pytest.raises(ValueError, match="response_format must be one of"):
        RubricSpecification(
            "faithfulness_rubric",
            "evidence_grounding",
            "Grounded.",
            "bad",
            _levels(),
            ("claim_verification",),
            ("Quote source evidence.",),
        )
    with pytest.raises(ValueError, match="levels must contain at least 2"):
        RubricSpecification(
            "faithfulness_rubric",
            "evidence_grounding",
            "Grounded.",
            ResponseFormat.ORDINAL_RATING,
            (_levels()[0],),
            ("claim_verification",),
            ("Quote source evidence.",),
        )
    with pytest.raises(ValueError, match=r"levels\[1\] must be a RubricLevel"):
        RubricSpecification(
            "faithfulness_rubric",
            "evidence_grounding",
            "Grounded.",
            ResponseFormat.ORDINAL_RATING,
            (_levels()[0], "bad"),
            ("claim_verification",),
            ("Quote source evidence.",),
        )
    with pytest.raises(ValueError, match="level scores must be contiguous"):
        RubricSpecification(
            "faithfulness_rubric",
            "evidence_grounding",
            "Grounded.",
            ResponseFormat.ORDINAL_RATING,
            (
                _levels()[0],
                RubricLevel(2, "other_level", "Other.", ("other",)),
            ),
            ("claim_verification",),
            ("Quote source evidence.",),
        )
    with pytest.raises(ValueError, match="level labels must be unique"):
        RubricSpecification(
            "faithfulness_rubric",
            "evidence_grounding",
            "Grounded.",
            ResponseFormat.ORDINAL_RATING,
            (
                _levels()[0],
                RubricLevel(1, "unsupported", "Other.", ("other",)),
            ),
            ("claim_verification",),
            ("Quote source evidence.",),
        )
    with pytest.raises(ValueError, match="task_families must not contain duplicates"):
        RubricSpecification(
            "faithfulness_rubric",
            "evidence_grounding",
            "Grounded.",
            ResponseFormat.ORDINAL_RATING,
            _levels(),
            ("claim_verification", "claim_verification"),
            ("Quote source evidence.",),
        )
    with pytest.raises(ValueError, match="evidence_requirements must not contain duplicates"):
        RubricSpecification(
            "faithfulness_rubric",
            "evidence_grounding",
            "Grounded.",
            ResponseFormat.ORDINAL_RATING,
            _levels(),
            ("claim_verification",),
            ("same", "same"),
        )
    with pytest.raises(ValueError, match="locale must be a BCP 47-style tag"):
        RubricSpecification(
            "faithfulness_rubric",
            "evidence_grounding",
            "Grounded.",
            ResponseFormat.ORDINAL_RATING,
            _levels(),
            ("claim_verification",),
            ("Quote source evidence.",),
            locale="bad_locale",
        )
    with pytest.raises(ValueError, match=r"schema_version must be '1\.0'"):
        RubricSpecification(
            "faithfulness_rubric",
            "evidence_grounding",
            "Grounded.",
            ResponseFormat.ORDINAL_RATING,
            _levels(),
            ("claim_verification",),
            ("Quote source evidence.",),
            schema_version="2.0",
        )


def test_blueprint_plan_failure_paths_are_complete():
    """Plan validation rejects empty, duplicate, unknown, non-integral, and bounded values."""
    with pytest.raises(ValueError, match="difficulty_bands must contain at least 1"):
        BlueprintPlan(())
    with pytest.raises(ValueError, match="difficulty_bands must not contain duplicates"):
        BlueprintPlan((DifficultyBand.EASY, DifficultyBand.EASY))
    with pytest.raises(ValueError, match="difficulty_bands must contain only"):
        BlueprintPlan(("bad",))
    with pytest.raises(ValueError, match="items_per_cell must be an integer"):
        BlueprintPlan(items_per_cell=True)
    with pytest.raises(ValueError, match="items_per_cell must be between"):
        BlueprintPlan(items_per_cell=0)
    with pytest.raises(ValueError, match="seed must be an integer"):
        BlueprintPlan(seed=1.5)
    with pytest.raises(ValueError, match="seed must fit an unsigned 64-bit integer"):
        BlueprintPlan(seed=-1)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("blueprint_id", "bad", "two-or-more-token"),
        ("rubric_fingerprint", "bad", "64-character"),
        ("task_family", "bad", "two-or-more-token"),
        ("difficulty_band", "bad", "difficulty_band must be one of"),
        ("evidence_mode", "bad", "evidence_mode must be one of"),
        ("replicate_index", True, "replicate_index must be an integer"),
        ("replicate_index", 100, "replicate_index must be between"),
        ("generation_seed", False, "generation_seed must be an integer"),
        ("generation_seed", -1, "generation_seed must fit an unsigned"),
        ("response_format", "bad", "response_format must be one of"),
        ("scoring_levels", (), "scoring_levels must contain at least 2"),
        ("scoring_levels", (0, 2), "scoring_levels must be contiguous"),
        ("evidence_requirements", (), "evidence_requirements must contain at least 1"),
        ("schema_version", "2.0", r"schema_version must be '1\.0'"),
    ],
)
def test_item_blueprint_failure_paths_are_complete(field, value, match):
    """Direct blueprint mutation cannot bypass any public constructor invariant."""
    blueprint = compile_item_blueprints(_rubric())[0]
    with pytest.raises(ValueError, match=match):
        replace(blueprint, **{field: value})


def test_default_budget_and_serialization_paths_are_exercised():
    """Default compilation, budget rejection, and every serializer execute."""
    rubric = _rubric()
    compiled = compile_item_blueprints(rubric)
    assert len(compiled) == 3
    large_rubric = RubricSpecification(
        "faithfulness_rubric",
        "evidence_grounding",
        "Grounded.",
        ResponseFormat.ORDINAL_RATING,
        _levels(),
        tuple(f"task_family_{index}" for index in range(8)),
        ("Quote source evidence.",),
    )
    with pytest.raises(ValueError, match="exceeds MAX_BLUEPRINTS"):
        compile_item_blueprints(
            large_rubric,
            BlueprintPlan(tuple(DifficultyBand), tuple(EvidenceMode), 100),
        )
    blueprint = compiled[0]
    assert blueprint.to_dict()["scoring_levels"] == [0, 1]
    assert rubric.to_dict()["levels"][0]["score"] == 0
    assert canonical_generation_contract(rubric, blueprint)
    assert render_generation_prompt(rubric, blueprint)
