"""Performance contracts for bounded rubric blueprint compilation."""

from __future__ import annotations

from fast_mlsirm.rubric import (
    BlueprintPlan,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    compile_item_blueprints,
)


def test_compiler_computes_rubric_fingerprint_once(monkeypatch):
    """A large blueprint matrix must not re-hash the same rubric per item."""
    rubric = RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which claims are supported.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(0, "unsupported", "No support.", ("no support",)),
            RubricLevel(1, "full_support", "Full support.", ("full support",)),
        ),
        task_families=("claim_verification",),
        evidence_requirements=("Quote supporting evidence.",),
    )
    original_getter = RubricSpecification.fingerprint.fget
    assert original_getter is not None
    calls = 0

    def counted_fingerprint(instance):
        nonlocal calls
        calls += 1
        return original_getter(instance)

    monkeypatch.setattr(
        RubricSpecification,
        "fingerprint",
        property(counted_fingerprint),
    )

    compiled = compile_item_blueprints(
        rubric,
        BlueprintPlan(items_per_cell=10),
    )

    assert len(compiled) == 30
    assert calls == 1
