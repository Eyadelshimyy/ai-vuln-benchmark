"""Strategy protocol and shared base utilities.

Strategies orchestrate multiple LLM calls in patterns like single-call,
self-consistency, tree-of-thought, reflection, and collaborative editing.

The :class:`Strategy` Protocol is the canonical public type. Any object
with a matching ``execute`` method satisfies it — no inheritance
required. :class:`BaseStrategy` is an optional helper that supplies
shared utilities (config normalization, prompt resolution, step
notification, runtime-template rendering, tool-aware LLM invocation).

Retry and backoff live exclusively on :class:`~ellements.core.LLMClient`.
Strategies must not retry on top.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

import chevron
from ellements.core import LLMClient, MessageInput, PromptKeyMissingError
from pydantic import BaseModel

from .config import (
    ExecutionStrategyConfig,
    StrategyConfigInput,
    normalize_strategy_config,
)

_logger = logging.getLogger(__name__)

ConfigModelT = TypeVar("ConfigModelT", bound=ExecutionStrategyConfig)
StructuredT = TypeVar("StructuredT", bound=BaseModel)

# Callback invoked after each :class:`StepRecord` is produced.
OnStepCallback = Callable[["StepRecord"], None]


@dataclass(slots=True)
class StepRecord:
    """One step in a strategy execution (single LLM call or aggregation)."""

    name: str
    prompt_key: str
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyResult:
    """Final result of a strategy execution."""

    output: str
    steps: list[StepRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Strategy(Protocol):
    """Protocol for execution strategies.

    Any object whose ``execute`` method matches this signature is a
    valid strategy — no inheritance required. This is the public type
    used in every signature that accepts a strategy.
    """

    async def execute(
        self,
        prompts: Mapping[str, str],
        client: LLMClient,
        tools: list[dict[str, Any]] | Mapping[str, Any] | None = None,
        config: StrategyConfigInput = None,
    ) -> StrategyResult: ...


class BaseStrategy:
    """Optional helper base class with utilities shared by built-in strategies."""

    @staticmethod
    def _notify_step(config: StrategyConfigInput, step: StepRecord) -> None:
        """Invoke the ``on_step`` callback if one is present in *config*."""
        if config is None:
            return
        callback: Any
        if isinstance(config, Mapping):
            callback = config.get("on_step")
        else:
            callback = getattr(config, "on_step", None)
        if callable(callback):
            callback(step)

    @staticmethod
    def _normalize_config(
        config: StrategyConfigInput,
        config_model: type[ConfigModelT],
    ) -> ConfigModelT:
        """Normalize *config* to the typed model expected by the strategy."""
        return normalize_strategy_config(config, config_model)

    @staticmethod
    def _get_prompt(
        prompts: Mapping[str, str],
        key: str,
        *,
        required: bool = True,
    ) -> str:
        """Return the prompt for *key*, with a strict required/optional contract.

        - ``required=True`` and the key is missing → raises
          :class:`PromptKeyMissingError`. There is no silent fallback to
          ``"default"``.
        - ``required=False`` and the key is missing → returns ``""``.
        """
        value = prompts.get(key)
        if value is not None:
            return value
        if not required:
            return ""
        raise PromptKeyMissingError(
            f"Strategy requires a prompt named {key!r}. "
            f"Available: {sorted(prompts.keys())}."
        )

    @staticmethod
    def _render_template(template: str, context: Mapping[str, Any]) -> str:
        """Render runtime placeholders against *context*.

        Strategies support regular Mustache placeholders (``{{name}}``) plus
        PromptSpec-style placeholders (``@{name}``) so executable PromptSpec
        files can use the same placeholder form as the rest of the DSL.
        """
        rendered: str = chevron.render(template, dict(context))

        def replace_promptspec_placeholder(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in context:
                return match.group(0)
            return str(context[key])

        return re.sub(
            r"@\{([A-Za-z_][\w.-]*)\}",
            replace_promptspec_placeholder,
            rendered,
        )

    @staticmethod
    def _stage_temperature(
        config: ExecutionStrategyConfig,
        stage: str,
        default: float,
    ) -> float:
        """Resolve the effective temperature for a named pipeline stage.

        ``stage_temperatures[stage]`` overrides the per-stage attribute
        on the typed config, which itself defaults to *default*.
        """
        if config.stage_temperatures and stage in config.stage_temperatures:
            return config.stage_temperatures[stage]
        attr = getattr(config, f"{stage}_temperature", None)
        if isinstance(attr, (int, float)):
            return float(attr)
        return default

    @classmethod
    async def _complete(
        cls,
        client: LLMClient,
        prompt: str,
        *,
        config: ExecutionStrategyConfig,
        temperature: float,
        system: str | None = None,
        tools: list[dict[str, Any]] | Mapping[str, Any] | None = None,
    ) -> str:
        """Make a single LLM call, with tools if any are provided.

        Retries are handled inside :class:`LLMClient`; strategies do not
        add a second retry layer.
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if tools:
            response = await client.complete_with_tools(
                messages,
                tools=tools,
                model=config.model,
                temperature=temperature,
            )
            return response.content
        return await client.complete(
            messages,
            model=config.model,
            temperature=temperature,
        )

    @staticmethod
    async def _complete_structured(
        client: LLMClient,
        prompt: str,
        response_model: type[StructuredT],
        *,
        config: ExecutionStrategyConfig,
        temperature: float,
        system: str | None = None,
    ) -> StructuredT:
        """Make a structured LLM call returning a Pydantic model instance."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await client.complete_structured(
            messages,
            response_model=response_model,
            model=config.model,
            temperature=temperature,
        )


__all__ = [
    "BaseStrategy",
    "OnStepCallback",
    "StepRecord",
    "Strategy",
    "StrategyResult",
]


# Allow type-checkers to see MessageInput is referenced even though we
# don't currently expose it through this module.
_ = MessageInput  # noqa: F841
