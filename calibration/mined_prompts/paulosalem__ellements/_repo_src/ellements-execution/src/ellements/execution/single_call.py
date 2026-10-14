"""Single-call strategy — the simplest execution pattern.

Makes one LLM call (optionally with tools) using the ``default`` prompt
and returns the result.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ellements.core import LLMClient

from .config import SingleCallConfig, StrategyConfigInput
from .strategies import BaseStrategy, StepRecord, StrategyResult


class SingleCallStrategy(BaseStrategy):
    """Execute a single LLM call with the ``default`` prompt."""

    REQUIRED_PROMPTS = ("default",)

    async def execute(
        self,
        prompts: Mapping[str, str],
        client: LLMClient,
        tools: list[dict[str, Any]] | Mapping[str, Any] | None = None,
        config: StrategyConfigInput = None,
    ) -> StrategyResult:
        typed_config = self._normalize_config(config, SingleCallConfig)
        prompt = self._get_prompt(prompts, "default")

        temperature = self._stage_temperature(typed_config, "single", default=0.7)
        if typed_config.temperature is not None:
            temperature = typed_config.temperature

        response = await self._complete(
            client,
            prompt,
            config=typed_config,
            temperature=temperature,
            system=prompts.get("system"),
            tools=tools,
        )

        step = StepRecord(name="call", prompt_key="default", response=response)
        self._notify_step(typed_config, step)
        return StrategyResult(output=response, steps=[step])
