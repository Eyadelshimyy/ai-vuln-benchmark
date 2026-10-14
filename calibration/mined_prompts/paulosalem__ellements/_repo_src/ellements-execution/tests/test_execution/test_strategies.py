"""Tests for execution strategies.

A lightweight ``FakeLLMClient`` is used in place of a real
:class:`ellements.core.LLMClient` so the suite runs without API keys.
The fake satisfies the duck-typed shape used by :class:`BaseStrategy`
(``complete``, ``complete_with_tools``, ``complete_structured``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from ellements.core import PromptKeyMissingError, ToolCallResponse
from ellements.execution import (
    BUILTIN_STRATEGIES,
    STRATEGY_FAMILIES,
    BaseStrategy,
    CollaborativeEditingConfig,
    CollaborativeEditingStrategy,
    CritiqueResult,
    Evaluation,
    PassthroughEditCallback,
    ReflectionConfig,
    ReflectionStrategy,
    SelfConsistencyConfig,
    SelfConsistencyStrategy,
    SingleCallConfig,
    SingleCallStrategy,
    StepRecord,
    Strategy,
    StrategyFamily,
    TreeOfThoughtConfig,
    TreeOfThoughtStrategy,
)
from pydantic import BaseModel

# ── Fake LLM client ───────────────────────────────────────────────


class FakeLLMClient:
    """Scripted async LLM client used in strategy tests."""

    def __init__(
        self,
        responses: list[str] | Callable[[list[dict[str, Any]]], str] | None = None,
        *,
        structured_responses: list[BaseModel] | None = None,
    ) -> None:
        self._responses = responses if responses is not None else []
        self._structured_responses = list(structured_responses or [])
        self._idx = 0
        self.calls: list[dict[str, Any]] = []
        self.model = "fake-model"

    async def complete(
        self,
        messages: Any,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        if isinstance(messages, str):
            payload = [{"role": "user", "content": messages}]
        else:
            payload = list(messages)
        self.calls.append(
            {
                "method": "complete",
                "messages": payload,
                "model": model,
                "temperature": temperature,
            }
        )
        return self._next_text(payload)

    async def complete_with_tools(
        self,
        messages: Any,
        *,
        tools: Any,
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ToolCallResponse:
        if isinstance(messages, str):
            payload = [{"role": "user", "content": messages}]
        else:
            payload = list(messages)
        self.calls.append(
            {
                "method": "complete_with_tools",
                "messages": payload,
                "tools": tools,
                "model": model,
                "temperature": temperature,
            }
        )
        return ToolCallResponse(content=self._next_text(payload), tool_calls=[])

    async def complete_structured(
        self,
        messages: Any,
        response_model: type[BaseModel],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> BaseModel:
        if not self._structured_responses:
            raise AssertionError("No structured response scripted")
        if isinstance(messages, str):
            payload = [{"role": "user", "content": messages}]
        else:
            payload = list(messages)
        self.calls.append(
            {
                "method": "complete_structured",
                "messages": payload,
                "model": model,
                "temperature": temperature,
                "response_model": response_model.__name__,
            }
        )
        value = self._structured_responses.pop(0)
        assert isinstance(value, response_model), (
            f"Scripted response type mismatch: expected {response_model.__name__}, "
            f"got {type(value).__name__}"
        )
        return value

    def _next_text(self, messages: list[dict[str, Any]]) -> str:
        if callable(self._responses):
            return self._responses(messages)
        if not self._responses:
            return "OK"
        idx = min(self._idx, len(self._responses) - 1)
        self._idx += 1
        return self._responses[idx]


# ── Strategy Protocol conformance ─────────────────────────────────


@pytest.mark.parametrize(
    "strategy_cls",
    [
        SingleCallStrategy,
        SelfConsistencyStrategy,
        ReflectionStrategy,
        TreeOfThoughtStrategy,
        CollaborativeEditingStrategy,
    ],
)
def test_built_in_strategy_conforms_to_protocol(strategy_cls):
    instance = strategy_cls()
    assert isinstance(instance, Strategy)


def test_strategy_catalog_lists_all_built_ins():
    assert set(BUILTIN_STRATEGIES) == {
        "single_call",
        "self_consistency",
        "tree_of_thought",
        "reflection",
        "collaborative_editing",
    }
    for spec in BUILTIN_STRATEGIES.values():
        strategy_cls = spec["strategy"]
        assert isinstance(strategy_cls(), Strategy)


def test_strategy_families_partition_catalog():
    seen: set[str] = set()
    for family in StrategyFamily:
        names = STRATEGY_FAMILIES[family]
        for name in names:
            assert name in BUILTIN_STRATEGIES
            assert name not in seen, f"{name} listed in multiple families"
            seen.add(name)
    assert seen == set(BUILTIN_STRATEGIES)


# ── BaseStrategy helpers ──────────────────────────────────────────


def test_get_prompt_raises_when_required_missing():
    with pytest.raises(PromptKeyMissingError):
        BaseStrategy._get_prompt({"a": "x"}, "missing")


def test_get_prompt_returns_empty_when_optional_missing():
    assert BaseStrategy._get_prompt({}, "missing", required=False) == ""


def test_render_template_mustache():
    rendered = BaseStrategy._render_template("hi {{name}}", {"name": "world"})
    assert rendered == "hi world"


def test_render_template_promptspec_placeholders():
    rendered = BaseStrategy._render_template("hi @{name}", {"name": "world"})
    assert rendered == "hi world"


# ── SingleCallStrategy ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_call_uses_default_prompt():
    client = FakeLLMClient(["the answer"])
    strategy = SingleCallStrategy()
    result = await strategy.execute(
        {"default": "What is 2+2?"},
        client,  # type: ignore[arg-type]
        config=SingleCallConfig(model="fake-model"),
    )
    assert result.output == "the answer"
    assert [s.name for s in result.steps] == ["call"]
    assert client.calls[0]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_single_call_passes_tools_through():
    client = FakeLLMClient(["used a tool"])
    strategy = SingleCallStrategy()
    result = await strategy.execute(
        {"default": "use the tool"},
        client,  # type: ignore[arg-type]
        tools=[
            {
                "type": "function",
                "function": {"name": "noop", "description": "x", "parameters": {}},
            }
        ],
    )
    assert result.output == "used a tool"
    assert client.calls[0]["method"] == "complete_with_tools"


@pytest.mark.asyncio
async def test_single_call_emits_on_step_callback():
    client = FakeLLMClient(["done"])
    seen: list[StepRecord] = []
    strategy = SingleCallStrategy()
    await strategy.execute(
        {"default": "go"},
        client,  # type: ignore[arg-type]
        config=SingleCallConfig(on_step=seen.append),
    )
    assert len(seen) == 1
    assert seen[0].name == "call"


@pytest.mark.asyncio
async def test_single_call_missing_prompt_raises():
    client = FakeLLMClient(["never"])
    with pytest.raises(PromptKeyMissingError):
        await SingleCallStrategy().execute(
            {},  # no default
            client,  # type: ignore[arg-type]
        )


# ── SelfConsistencyStrategy ───────────────────────────────────────


@pytest.mark.asyncio
async def test_self_consistency_majority_vote():
    client = FakeLLMClient(["A", "A", "B"])
    strategy = SelfConsistencyStrategy()
    result = await strategy.execute(
        {"default": "pick one"},
        client,  # type: ignore[arg-type]
        config=SelfConsistencyConfig(samples=3, aggregation="majority_vote"),
    )
    assert result.output == "A"
    assert result.metadata == {"samples": 3, "aggregation": "majority_vote"}


@pytest.mark.asyncio
async def test_self_consistency_majority_vote_uses_answer_lines():
    client = FakeLLMClient(
        [
            "Path one.\nANSWER: 42",
            "Different reasoning.\nANSWER: 42.",
            "Mistaken path.\nANSWER: 41",
        ]
    )
    strategy = SelfConsistencyStrategy()
    result = await strategy.execute(
        {"default": "solve"},
        client,  # type: ignore[arg-type]
        config=SelfConsistencyConfig(samples=3, aggregation="majority_vote"),
    )
    assert result.output == "ANSWER: 42"


@pytest.mark.asyncio
async def test_self_consistency_llm_judge_makes_extra_call():
    client = FakeLLMClient(["A", "B", "C", "synthesized"])
    strategy = SelfConsistencyStrategy()
    result = await strategy.execute(
        {"default": "pick one"},
        client,  # type: ignore[arg-type]
        config=SelfConsistencyConfig(samples=3, aggregation="llm_judge"),
    )
    assert result.output == "synthesized"
    assert sum(1 for c in client.calls if c["method"] == "complete") == 4


# ── ReflectionStrategy ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reflection_stops_when_satisfied_after_first_round():
    client = FakeLLMClient(
        responses=["draft", "should not be called"],
        structured_responses=[CritiqueResult(is_satisfied=True, issues=[])],
    )
    strategy = ReflectionStrategy()
    result = await strategy.execute(
        {
            "generate": "Write a sentence",
            "critique": "Review: {{response}}",
            "revise": "Improve {{response}} given {{issues}}",
        },
        client,  # type: ignore[arg-type]
        config=ReflectionConfig(max_rounds=3),
    )
    assert result.output == "draft"
    assert result.metadata["satisfied"] is True
    assert any(s.name == "stop" for s in result.steps)


@pytest.mark.asyncio
async def test_reflection_loops_until_satisfied():
    client = FakeLLMClient(
        responses=["draft", "revision-1", "revision-2"],
        structured_responses=[
            CritiqueResult(is_satisfied=False, issues=["wrong"]),
            CritiqueResult(is_satisfied=False, issues=["still wrong"]),
            CritiqueResult(is_satisfied=True, issues=[]),
        ],
    )
    strategy = ReflectionStrategy()
    result = await strategy.execute(
        {
            "generate": "Write a sentence",
            "critique": "Review: {{response}}",
            "revise": "Fix {{response}}; issues:\n{{issues}}",
        },
        client,  # type: ignore[arg-type]
        config=ReflectionConfig(max_rounds=5),
    )
    assert result.output == "revision-2"
    assert result.metadata["satisfied"] is True


@pytest.mark.asyncio
async def test_reflection_stops_at_max_rounds():
    client = FakeLLMClient(
        responses=["draft", "rev1", "rev2", "rev3"],
        structured_responses=[
            CritiqueResult(is_satisfied=False, issues=["x"]),
            CritiqueResult(is_satisfied=False, issues=["y"]),
            CritiqueResult(is_satisfied=False, issues=["z"]),
        ],
    )
    strategy = ReflectionStrategy()
    result = await strategy.execute(
        {
            "generate": "go",
            "critique": "review {{response}}",
            "revise": "improve {{response}} issues {{issues}}",
        },
        client,  # type: ignore[arg-type]
        config=ReflectionConfig(max_rounds=3),
    )
    assert result.metadata["satisfied"] is False


# ── TreeOfThoughtStrategy ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_tot_simple_mode_returns_synthesis():
    strategy = TreeOfThoughtStrategy()
    client = FakeLLMClient(
        responses=[
            "candidate-1",
            "candidate-2",
            "synthesis",
        ],
        structured_responses=[Evaluation(score=0.9, reasoning="best")],
    )
    result = await strategy.execute(
        {
            "generate": "Generate one approach.",
            "evaluate": "Rank: {{candidates}}",
            "synthesize": "Combine: {{candidates}}",
        },
        client,  # type: ignore[arg-type]
        config=TreeOfThoughtConfig(mode="simple", branching_factor=2),
    )
    assert result.output == "synthesis"
    assert result.metadata["mode"] == "simple"


@pytest.mark.asyncio
async def test_tot_beam_mode_synthesizes_best_state():
    strategy = TreeOfThoughtStrategy()
    client = FakeLLMClient(
        responses=[
            "t1",
            "t2",
            "synth",
        ],
        structured_responses=[
            Evaluation(score=0.9, reasoning="ok"),
            Evaluation(score=0.4, reasoning="meh"),
        ],
    )
    result = await strategy.execute(
        {
            "thought_step": "Next thought from {{state}}",
            "evaluate_step": "Rate {{state}}",
            "synthesize": "Synthesize {{best_path}}",
        },
        client,  # type: ignore[arg-type]
        config=TreeOfThoughtConfig(
            mode="beam",
            max_depth=1,
            branching_factor=2,
            beam_width=1,
        ),
    )
    assert result.metadata["mode"] == "beam"
    assert result.output == "synth"


@pytest.mark.asyncio
async def test_tot_dfs_mode_visits_branches_and_synthesizes():
    strategy = TreeOfThoughtStrategy()
    client = FakeLLMClient(
        responses=[
            "branch-1",
            "branch-2",
            "synth",
        ],
        structured_responses=[
            # Root: two branches scored, then one leaf eval each
            Evaluation(score=0.4, reasoning="ok"),
            Evaluation(score=0.9, reasoning="leaf-1"),
            Evaluation(score=0.7, reasoning="ok"),
            Evaluation(score=0.6, reasoning="leaf-2"),
        ],
    )
    result = await strategy.execute(
        {
            "thought_step": "next {{state}}",
            "evaluate_step": "score {{state}}",
            "synthesize": "render {{best_path}}",
        },
        client,  # type: ignore[arg-type]
        config=TreeOfThoughtConfig(
            mode="dfs",
            max_depth=1,
            branching_factor=2,
        ),
    )
    assert result.metadata["mode"] == "dfs"
    assert result.output == "synth"


# ── CollaborativeEditingStrategy ──────────────────────────────────


class _ScriptedCallback:
    def __init__(self, edits: list[str]) -> None:
        self.edits = list(edits)
        self.calls: list[tuple[str, str]] = []

    async def request_edit(self, content: str, context: str = "") -> str:
        self.calls.append((content, context))
        return self.edits.pop(0)


@pytest.mark.asyncio
async def test_collaborative_passthrough_unchanged_stops():
    client = FakeLLMClient(["draft"])
    strategy = CollaborativeEditingStrategy()
    result = await strategy.execute(
        {"generate": "go", "continue": "extend {{edited_content}}"},
        client,  # type: ignore[arg-type]
        config=CollaborativeEditingConfig(
            max_rounds=3,
            edit_callback=PassthroughEditCallback(),
        ),
    )
    assert result.output == "draft"
    assert result.metadata["rounds_completed"] == 1


@pytest.mark.asyncio
async def test_collaborative_continues_after_edit():
    callback = _ScriptedCallback(["edited", "still edited"])
    client = FakeLLMClient(["draft", "after-edit", "after-second-edit"])
    strategy = CollaborativeEditingStrategy()
    result = await strategy.execute(
        {"generate": "go", "continue": "given {{edited_content}}"},
        client,  # type: ignore[arg-type]
        config=CollaborativeEditingConfig(
            max_rounds=2,
            edit_callback=callback,
            continue_on_unchanged=True,
        ),
    )
    assert len(callback.calls) == 2
    assert result.output == "after-second-edit"


@pytest.mark.asyncio
async def test_collaborative_done_signal_exits():
    callback = _ScriptedCallback(["edited\nDONE"])
    client = FakeLLMClient(["draft"])
    strategy = CollaborativeEditingStrategy()
    result = await strategy.execute(
        {"generate": "go", "continue": "ignored"},
        client,  # type: ignore[arg-type]
        config=CollaborativeEditingConfig(
            max_rounds=4,
            edit_callback=callback,
        ),
    )
    assert result.output == "edited"
    assert any(s.metadata.get("done_signal") for s in result.steps)
