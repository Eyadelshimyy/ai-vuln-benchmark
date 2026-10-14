"""Tests for the agents package: AgentBackend Protocol, runner, stats."""

from __future__ import annotations

from typing import Any

import pytest
from ellements.agents import (
    AgentBackend,
    AgentBuilder,
    AgentEvent,
    AgentRunResult,
    AgentRunStats,
    ClaudeAgentsBackend,
    OpenAIAgentsBackend,
    run_agent_with_progress,
)

# ── Fake backend used by the runner tests ──────────────────────────


class _FakeStream:
    def __init__(self, events: list[AgentEvent], result: Any) -> None:
        self._events = events
        self._result = result

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self._events:
            yield event

    @property
    def result(self) -> Any:
        return self._result


class _FakeNativeResult:
    def __init__(self, final_output: str = "fake-out") -> None:
        self.final_output = final_output
        self.new_items: list[Any] = []


class _FakeBackend:
    name: str = "fake"

    def __init__(self, *, events: list[AgentEvent] | None = None):
        self._events = events or []
        self._native_result = _FakeNativeResult()
        self.created: list[dict[str, Any]] = []

    def create_agent(self, *, name, instructions, tools, model):
        agent = {
            "name": name,
            "instructions": instructions,
            "tools": tools,
            "model": model,
        }
        self.created.append(agent)
        return agent

    def create_session(self, session_id=None, **kwargs):
        return None

    async def run(self, agent, task, *, max_turns=10, session=None, **kwargs):
        return self._native_result

    def stream_run(self, agent, task, *, max_turns=10, session=None, **kwargs):
        return _FakeStream(self._events, self._native_result)


# ── AgentBackend Protocol conformance ─────────────────────────────


def test_fake_backend_conforms_to_protocol():
    assert isinstance(_FakeBackend(), AgentBackend)


def test_openai_backend_conforms_to_protocol():
    backend = OpenAIAgentsBackend()
    assert isinstance(backend, AgentBackend)
    assert backend.name == "openai_agents"


def test_claude_backend_class_implements_protocol_surface():
    assert all(
        hasattr(ClaudeAgentsBackend, attr)
        for attr in ("name", "create_agent", "create_session", "run", "stream_run")
    )


# ── AgentRunStats ──────────────────────────────────────────────────


def test_stats_increment_tool_call():
    stats = AgentRunStats()
    stats.increment_tool_call("search")
    stats.increment_tool_call("search")
    stats.increment_tool_call("fetch")
    assert stats.tool_calls == 3
    assert stats.tool_names == {"search": 2, "fetch": 1}


def test_stats_increment_metric_is_only_explicit_path():
    stats = AgentRunStats()
    assert not hasattr(stats, "searches")
    assert not hasattr(stats, "results_examined")
    stats.increment_metric("searches")
    stats.increment_metric("results_examined", 3)
    assert stats.metrics == {"searches": 1, "results_examined": 3}


def test_stats_merge():
    a = AgentRunStats(tool_calls=2, tool_names={"x": 2})
    b = AgentRunStats(tool_calls=3, tool_names={"x": 1, "y": 3})
    b.increment_metric("custom", 5)
    a.merge(b)
    assert a.tool_calls == 5
    assert a.tool_names == {"x": 3, "y": 3}
    assert a.metrics == {"custom": 5}


def test_stats_to_dict_round_trip():
    stats = AgentRunStats(tool_calls=1, tool_outputs=2, observed_items=4)
    stats.increment_metric("queries", 7)
    payload = stats.to_dict()
    assert payload["tool_calls"] == 1
    assert payload["tool_outputs"] == 2
    assert payload["observed_items"] == 4
    assert payload["metrics"]["queries"] == 7


# ── run_agent_with_progress (streaming) ───────────────────────────


@pytest.mark.asyncio
async def test_runner_streams_events_and_counts_tool_calls():
    events = [
        AgentEvent(type="agent_active", payload={"name": "Test"}),
        AgentEvent(type="tool_call", payload={"name": "search", "arguments": "{}"}),
        AgentEvent(type="tool_output", payload={"output": "ok"}),
        AgentEvent(type="message", payload={"text": "done", "final": True}),
    ]
    backend = _FakeBackend(events=events)
    agent = backend.create_agent(
        name="Test", instructions="i", tools=[], model="m"
    )

    progress: list[str] = []

    async def on_progress(msg: str) -> None:
        progress.append(msg)

    result = await run_agent_with_progress(
        backend, agent, "go", progress_callback=on_progress
    )

    assert isinstance(result, AgentRunResult)
    assert result.final_output == "fake-out"
    assert result.stats.tool_calls == 1
    assert result.stats.tool_outputs == 1
    assert result.stats.tool_names == {"search": 1}
    assert progress, "progress callback should have fired"


@pytest.mark.asyncio
async def test_runner_falls_back_to_run_when_no_activity_log():
    backend = _FakeBackend()
    agent = backend.create_agent(
        name="t", instructions="i", tools=[], model="m"
    )
    result = await run_agent_with_progress(
        backend, agent, "go", show_activity_log=False
    )
    assert isinstance(result, AgentRunResult)
    assert result.final_output == "fake-out"


@pytest.mark.asyncio
async def test_runner_unknown_event_types_are_ignored():
    events = [
        AgentEvent(type="openai.raw_response", payload={"raw": True}),
        AgentEvent(type="message", payload={"text": "ok", "final": True}),
    ]
    backend = _FakeBackend(events=events)
    agent = backend.create_agent(
        name="t", instructions="i", tools=[], model="m"
    )

    async def progress(msg: str) -> None: ...

    result = await run_agent_with_progress(
        backend, agent, "go", progress_callback=progress
    )
    assert result.stats.tool_calls == 0


# ── AgentRunResult exposes native result only via .result ─────────


def test_run_result_native_data_accessed_via_result_attribute():
    """No attribute pass-through: callers go through ``res.result`` explicitly."""
    native = _FakeNativeResult(final_output="hi")
    native.token_count = 42
    res = AgentRunResult(result=native, stats=AgentRunStats())
    assert res.final_output == "hi"
    assert res.result.token_count == 42
    assert not hasattr(res, "token_count")


# ── AgentBuilder ───────────────────────────────────────────────────


def test_builder_requires_model_before_build():
    backend = _FakeBackend()
    builder = AgentBuilder("Test", backend=backend).with_instructions("hi")
    with pytest.raises(ValueError, match="Model must be set"):
        builder.build()


def test_builder_builds_through_backend():
    backend = _FakeBackend()
    agent = (
        AgentBuilder("Test", backend=backend)
        .with_model("openai/gpt-4o-mini")
        .with_instructions("Do the thing")
        .build()
    )
    assert agent == backend.created[0]
    assert agent["model"] == "openai/gpt-4o-mini"


def test_builder_template_mode_requires_persona():
    backend = _FakeBackend()
    builder = (
        AgentBuilder("Test", backend=backend)
        .with_model("openai/gpt-4o-mini")
        .with_instructions_template("information-retrieval/youtube_search")
    )
    with pytest.raises(ValueError, match="active persona"):
        builder.build()


def test_builder_persona_id_requires_attached_library():
    backend = _FakeBackend()
    builder = AgentBuilder("Test", backend=backend)
    with pytest.raises(ValueError, match="library not attached"):
        builder.with_persona_id("nope")


def test_builder_guideline_id_requires_attached_library():
    backend = _FakeBackend()
    builder = AgentBuilder("Test", backend=backend)
    with pytest.raises(ValueError, match="library not attached"):
        builder.with_guideline_id("nope")
