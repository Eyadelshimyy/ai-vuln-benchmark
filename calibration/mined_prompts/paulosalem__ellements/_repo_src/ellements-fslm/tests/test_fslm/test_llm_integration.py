from __future__ import annotations

import os

import pytest
from ellements.core import LLMClient
from ellements.fslm import FSLMEvent, FSLMKernel, LLMDecisionEvaluator, machine, nl


def _live_evaluator() -> LLMDecisionEvaluator:
    if not (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("LITELLM_API_KEY")
    ):
        pytest.skip("live LLM credentials are not configured")
    model = os.environ.get("FSLM_TEST_MODEL", "gpt-4.1")
    return LLMDecisionEvaluator(LLMClient(model=model), model=model)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_guard_evaluator_allows_obvious_true_case() -> None:
    m = machine("llm-demo", initial="checking")
    with m.state("checking") as s:
        s.on("evidence").to("done", "accept").when(
            nl.guard(
                "evidence_says_passed",
                "Allow only if the event clearly says validation passed.",
            )
        )
    m.state("done")
    definition = m.build()
    evaluator = _live_evaluator()

    result = await FSLMKernel(
        definition,
        guard_evaluator=evaluator,
    ).step(
        definition.initial_snapshot(),
        FSLMEvent(type="evidence", payload={"summary": "Validation passed."}),
    )

    assert result.selected_transition == "accept"
    assert result.guard_results[0].allowed is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_guard_evaluator_rejects_obvious_false_case() -> None:
    m = machine("llm-demo-reject", initial="checking")
    with m.state("checking") as s:
        s.on("evidence").to("done", "accept").when(
            nl.guard(
                "evidence_says_passed",
                "Allow only if the event clearly says validation passed.",
            )
        )
    m.state("done")
    definition = m.build()

    result = await FSLMKernel(
        definition,
        guard_evaluator=_live_evaluator(),
    ).step(
        definition.initial_snapshot(),
        FSLMEvent(type="evidence", payload={"summary": "Validation failed."}),
    )

    assert result.status == "no_transition"
    assert result.selected_transition is None
    assert result.guard_results[0].allowed is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_invariant_checker_blocks_bad_state() -> None:
    m = machine("llm-invariant-demo", initial="validating")
    with m.state("validating") as s:
        s.on("finish").to("reporting", "finish_validation")
    with m.state("reporting") as s:
        s.invariant(
            nl.invariant(
                "has_validation_evidence",
                "The event or variables must include concrete validation evidence.",
            )
        )
    definition = m.build()

    result = await FSLMKernel(
        definition,
        invariant_checker=_live_evaluator(),
    ).step(
        definition.initial_snapshot(),
        FSLMEvent(type="finish", payload={"summary": "No tests were run."}),
    )

    assert result.selected_transition == "finish_validation"
    assert result.status == "blocked"
    assert result.invariant_results[0].allowed is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_invariant_checker_allows_good_state() -> None:
    m = machine("llm-invariant-pass-demo", initial="validating")
    with m.state("validating") as s:
        s.on("finish").to("reporting", "finish_validation")
    with m.state("reporting") as s:
        s.invariant(
            nl.invariant(
                "has_validation_evidence",
                "The event or variables must include concrete validation evidence.",
            )
        )
    definition = m.build()

    result = await FSLMKernel(
        definition,
        invariant_checker=_live_evaluator(),
    ).step(
        definition.initial_snapshot(),
        FSLMEvent(
            type="finish",
            payload={
                "validation": {
                    "command": "python -m pytest tests/test_feature.py -q",
                    "exit_code": 0,
                    "evidence": "3 passed",
                }
            },
        ),
    )

    assert result.selected_transition == "finish_validation"
    assert result.status == "transitioned"
    assert result.invariant_results[0].allowed is True
