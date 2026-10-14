"""Tests for LLMClient.complete_with_tools tool-calling loop and MaxToolIterationsError.

All tests mock ``litellm.acompletion`` so no real API calls happen.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ellements.core import (
    LLMClient,
    MaxToolIterationsError,
    SimpleTool,
    ToolCallResponse,
)


def _mock_response(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    usage: dict[str, Any] | None = None,
    response_cost: float | None = None,
) -> Any:
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.role = "assistant"
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    response.usage = usage
    response._hidden_params = (
        {"response_cost": response_cost} if response_cost is not None else {}
    )
    return response


def _mock_tool_call(*, call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _ping_tool() -> SimpleTool:
    def ping(message: str) -> str:
        """Echo a ping."""
        return f"pong:{message}"

    return SimpleTool(ping, name="ping", description="Echo a ping.")


# ── Happy path ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_with_tools_terminates_on_final_message():
    client = LLMClient(model="openai/gpt-4o-mini")
    tool = _ping_tool()

    tool_call = _mock_tool_call(
        call_id="call-1", name="ping", arguments={"message": "hi"}
    )

    responses = [
        _mock_response(content=None, tool_calls=[tool_call]),
        _mock_response(content="done"),
    ]
    mock = AsyncMock(side_effect=responses)

    with patch("ellements.core.llm.client.litellm.acompletion", mock):
        result = await client.complete_with_tools(
            "Please ping",
            tools=[tool],
            max_iterations=4,
        )

    assert isinstance(result, ToolCallResponse)
    assert result.content == "done"
    assert len(result.tool_calls) == 1
    record = result.tool_calls[0]
    assert record.name == "ping"
    assert record.arguments == {"message": "hi"}
    assert record.result == "pong:hi"


@pytest.mark.asyncio
async def test_complete_with_tools_records_every_call_in_order():
    client = LLMClient(model="openai/gpt-4o-mini")
    tool = _ping_tool()

    responses = [
        _mock_response(
            content=None,
            tool_calls=[
                _mock_tool_call(call_id="c1", name="ping", arguments={"message": "a"}),
                _mock_tool_call(call_id="c2", name="ping", arguments={"message": "b"}),
            ],
        ),
        _mock_response(content="all done"),
    ]
    mock = AsyncMock(side_effect=responses)

    with patch("ellements.core.llm.client.litellm.acompletion", mock):
        result = await client.complete_with_tools(
            "Please ping twice",
            tools=[tool],
            max_iterations=3,
        )

    assert [c.arguments for c in result.tool_calls] == [
        {"message": "a"},
        {"message": "b"},
    ]


@pytest.mark.asyncio
async def test_complete_with_tools_aggregates_usage_across_turns():
    client = LLMClient(model="openai/gpt-4o-mini")
    tool = _ping_tool()
    response_events: list[Any] = []

    class Observer:
        async def on_request(self, event: Any) -> None:
            pass

        async def on_response(self, event: Any) -> None:
            response_events.append(event)

        async def on_error(self, event: Any) -> None:
            pass

    client.add_observer(Observer())
    responses = [
        _mock_response(
            tool_calls=[
                _mock_tool_call(
                    call_id="c1",
                    name="ping",
                    arguments={"message": "a"},
                )
            ],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
            response_cost=0.001,
        ),
        _mock_response(
            content="done",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
            response_cost=0.002,
        ),
    ]

    with patch(
        "ellements.core.llm.client.litellm.acompletion",
        AsyncMock(side_effect=responses),
    ):
        await client.complete_with_tools(
            "Please ping",
            tools=[tool],
            max_iterations=3,
        )

    assert len(response_events) == 1
    assert response_events[0].usage == {
        "prompt_tokens": 30,
        "completion_tokens": 9,
        "total_tokens": 39,
        "response_cost": pytest.approx(0.003),
    }


# ── MaxToolIterationsError contract ────────────────────────────────


@pytest.mark.asyncio
async def test_max_tool_iterations_carries_partial_state():
    client = LLMClient(model="openai/gpt-4o-mini")
    tool = _ping_tool()

    def make_response(turn: int) -> Any:
        return _mock_response(
            content=None,
            tool_calls=[
                _mock_tool_call(
                    call_id=f"c{turn}",
                    name="ping",
                    arguments={"message": f"turn-{turn}"},
                )
            ],
        )

    mock = AsyncMock(side_effect=[make_response(i) for i in range(50)])

    with (
        patch("ellements.core.llm.client.litellm.acompletion", mock),
        pytest.raises(MaxToolIterationsError) as excinfo,
    ):
        await client.complete_with_tools(
            "Loop forever",
            tools=[tool],
            max_iterations=2,
        )

    err = excinfo.value
    assert err.max_iterations == 2
    assert len(err.tool_calls_made) == 2
    assert err.unresolved_tool_calls
    assert any(call["function"]["name"] == "ping" for call in err.unresolved_tool_calls)
    assert err.messages, "Partial conversation must be carried on the exception"


# ── Tool input validation ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_executor_required_for_schema_only_tools():
    client = LLMClient(model="openai/gpt-4o-mini")
    schema_only = {
        "type": "function",
        "function": {"name": "schema_only", "description": "x", "parameters": {}},
    }

    with pytest.raises(ValueError, match="tool_executor is required"):
        await client.complete_with_tools(
            "use schema",
            tools=[schema_only],
            max_iterations=1,
        )
