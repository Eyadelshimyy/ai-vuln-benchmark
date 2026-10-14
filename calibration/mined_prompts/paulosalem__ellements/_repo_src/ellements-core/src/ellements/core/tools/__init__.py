"""Provider-neutral tool architecture.

This subpackage defines the canonical tool surface every LLM call and
every agent backend speaks to:

- :class:`Tool` — the duck-typed Protocol.
- :class:`ToolSpec` — the canonical in-memory record.
- :class:`SimpleTool` — convenience builder from a Python callable.
- :class:`ToolRegistry` — composable typed collection.
- :class:`ToolDialect` and built-in dialects — provider-specific wire
  formats (:class:`OpenAIChatDialect`, :class:`OpenAIResponsesDialect`,
  :class:`AnthropicDialect`, :class:`GeminiDialect`).
- :class:`ToolCallRecord` / :class:`ToolCallResponse` — result records.
"""

from __future__ import annotations

from .dialects import (
    AnthropicDialect,
    GeminiDialect,
    OpenAIChatDialect,
    OpenAIResponsesDialect,
    ToolDialect,
    default_dialect_for_model,
)
from .executor import ToolExecutor, stringify_tool_result
from .protocol import Tool
from .records import ToolCallRecord, ToolCallResponse
from .registry import ToolExecutorFn, ToolInput, ToolRegistry
from .simple import SimpleTool, bind_json_invoker
from .spec import ToolSpec

__all__ = [
    "AnthropicDialect",
    "GeminiDialect",
    "OpenAIChatDialect",
    "OpenAIResponsesDialect",
    "SimpleTool",
    "Tool",
    "ToolCallRecord",
    "ToolCallResponse",
    "ToolDialect",
    "ToolExecutor",
    "ToolExecutorFn",
    "ToolInput",
    "ToolRegistry",
    "ToolSpec",
    "bind_json_invoker",
    "default_dialect_for_model",
    "stringify_tool_result",
]
