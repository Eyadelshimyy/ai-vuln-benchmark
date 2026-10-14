"""Result records for multi-turn tool-calling completions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    """One tool invocation made during a multi-turn completion."""

    name: str = Field(description="Tool/function name that was invoked")
    arguments: dict[str, Any] = Field(description="Arguments passed to the tool")
    result: str = Field(description="String result returned by the tool executor")


class ToolCallResponse(BaseModel):
    """Final response of a multi-turn completion with tool calling."""

    content: str = Field(description="Final text response from the model")
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="Ordered log of every tool invocation during the session",
    )


__all__ = ["ToolCallRecord", "ToolCallResponse"]
