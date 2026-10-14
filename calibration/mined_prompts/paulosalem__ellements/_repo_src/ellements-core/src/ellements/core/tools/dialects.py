"""Wire-format dialects for LLM-provider tool definitions.

A :class:`ToolDialect` translates a provider-neutral
:class:`~ellements.core.tools.ToolSpec` into the JSON shape a given
provider's API expects. Adding support for a new provider is one new
dialect class with no churn elsewhere.

The built-in dialects are:

- :class:`OpenAIChatDialect` — OpenAI Chat Completions / LiteLLM default.
- :class:`OpenAIResponsesDialect` — OpenAI Responses API (flatter shape).
- :class:`AnthropicDialect` — Anthropic Messages API direct.
- :class:`GeminiDialect` — Google Gemini direct API.

:func:`default_dialect_for_model` picks a sensible dialect from a model
name; callers can override explicitly when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from .spec import ToolSpec


@runtime_checkable
class ToolDialect(Protocol):
    """Emits a provider-specific tool definition from a :class:`ToolSpec`."""

    name: ClassVar[str]
    """Stable dialect identifier (e.g. ``"openai_chat"``)."""

    def emit(self, spec: ToolSpec) -> dict[str, Any]:
        """Return the provider-specific dictionary for *spec*."""


@dataclass(slots=True, frozen=True)
class OpenAIChatDialect:
    """OpenAI Chat Completions wire format (also used by LiteLLM)."""

    name: ClassVar[str] = "openai_chat"

    def emit(self, spec: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.params_json_schema,
            },
        }


@dataclass(slots=True, frozen=True)
class OpenAIResponsesDialect:
    """OpenAI Responses API wire format (flat, no ``function`` envelope)."""

    name: ClassVar[str] = "openai_responses"

    def emit(self, spec: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.params_json_schema,
        }


@dataclass(slots=True, frozen=True)
class AnthropicDialect:
    """Anthropic Messages API wire format (``input_schema`` key)."""

    name: ClassVar[str] = "anthropic"

    def emit(self, spec: ToolSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.params_json_schema,
        }


@dataclass(slots=True, frozen=True)
class GeminiDialect:
    """Google Gemini direct API wire format."""

    name: ClassVar[str] = "gemini"

    def emit(self, spec: ToolSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.params_json_schema,
        }


def default_dialect_for_model(
    model: str,
    *,
    use_responses_api: bool = False,
) -> ToolDialect:
    """Pick the appropriate :class:`ToolDialect` for *model*.

    LiteLLM normalizes every provider onto the OpenAI Chat Completions
    tool wire format, so the default for litellm-routed calls is
    :class:`OpenAIChatDialect`. For direct (non-litellm) clients,
    Anthropic/Gemini-specific dialects apply.

    Args:
        model: The model identifier, possibly prefixed with a provider
            slug (e.g. ``"anthropic/claude-3-5-sonnet"``).
        use_responses_api: If true, prefer
            :class:`OpenAIResponsesDialect` for OpenAI models.

    Returns:
        A :class:`ToolDialect` instance.
    """
    model_lc = model.lower()
    if model_lc.startswith("anthropic/") or model_lc.startswith("claude-"):
        return AnthropicDialect()
    if model_lc.startswith("gemini/") or model_lc.startswith("gemini-"):
        return GeminiDialect()
    if use_responses_api and (model_lc.startswith("openai/") or "/" not in model_lc):
        return OpenAIResponsesDialect()
    return OpenAIChatDialect()


__all__ = [
    "AnthropicDialect",
    "GeminiDialect",
    "OpenAIChatDialect",
    "OpenAIResponsesDialect",
    "ToolDialect",
    "default_dialect_for_model",
]
