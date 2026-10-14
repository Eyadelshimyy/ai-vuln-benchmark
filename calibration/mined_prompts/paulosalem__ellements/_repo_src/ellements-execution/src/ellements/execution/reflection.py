"""Reflection strategy — generate → critique → revise loop.

Produces an initial response, then asks the model to critique it via a
structured :class:`CritiqueResult`. When ``is_satisfied`` becomes true
(or ``max_rounds`` is exhausted), the most recent revision is returned.

Required prompts:
    - ``generate``: produce the initial response
    - ``critique``: receives ``response``; must return a CritiqueResult
    - ``revise``: receives ``response`` and ``issues``
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ellements.core import LLMClient
from pydantic import BaseModel, Field

from .config import ReflectionConfig, StrategyConfigInput
from .strategies import BaseStrategy, StepRecord, StrategyResult

_logger = logging.getLogger(__name__)


class CritiqueResult(BaseModel):
    """Structured critique output produced by the ``critique`` stage."""

    is_satisfied: bool = Field(
        description="True if the response is acceptable as-is (no further revision needed)"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific issues the next revision should address",
    )


class ReflectionStrategy(BaseStrategy):
    """Generate → critique → revise loop using a structured CritiqueResult."""

    REQUIRED_PROMPTS = ("generate", "critique", "revise")

    async def execute(
        self,
        prompts: Mapping[str, str],
        client: LLMClient,
        tools: list[dict[str, Any]] | Mapping[str, Any] | None = None,
        config: StrategyConfigInput = None,
    ) -> StrategyResult:
        typed_config = self._normalize_config(config, ReflectionConfig)
        system = prompts.get("system")
        steps: list[StepRecord] = []

        generate_temp = self._stage_temperature(
            typed_config, "generate", default=typed_config.generate_temperature
        )
        critique_temp = self._stage_temperature(
            typed_config, "critique", default=typed_config.critique_temperature
        )
        revise_temp = self._stage_temperature(
            typed_config, "revise", default=typed_config.revise_temperature
        )

        generate_prompt = self._get_prompt(prompts, "generate")
        current = await self._complete(
            client,
            generate_prompt,
            config=typed_config,
            temperature=generate_temp,
            system=system,
            tools=tools,
        )
        steps.append(
            StepRecord(name="generate", prompt_key="generate", response=current)
        )
        self._notify_step(typed_config, steps[-1])

        critique_template = self._get_prompt(prompts, "critique")
        revise_template = self._get_prompt(prompts, "revise")

        for round_num in range(typed_config.max_rounds):
            critique_prompt = self._render_template(
                critique_template, {"response": current}
            )
            critique = await self._complete_structured(
                client,
                critique_prompt,
                response_model=CritiqueResult,
                config=typed_config,
                temperature=critique_temp,
                system=system,
            )
            steps.append(
                StepRecord(
                    name=f"critique_{round_num}",
                    prompt_key="critique",
                    response=critique.model_dump_json(),
                    metadata={
                        "round": round_num,
                        "is_satisfied": critique.is_satisfied,
                        "issue_count": len(critique.issues),
                    },
                )
            )
            self._notify_step(typed_config, steps[-1])

            if critique.is_satisfied:
                stop_step = StepRecord(
                    name="stop",
                    prompt_key="critique",
                    response=f"Stopped at round {round_num}: critique satisfied.",
                    metadata={"round": round_num, "reason": "satisfied"},
                )
                steps.append(stop_step)
                self._notify_step(typed_config, stop_step)
                break

            revise_prompt = self._render_template(
                revise_template,
                {
                    "response": current,
                    "issues": "\n".join(f"- {issue}" for issue in critique.issues),
                },
            )
            current = await self._complete(
                client,
                revise_prompt,
                config=typed_config,
                temperature=revise_temp,
                system=system,
                tools=tools,
            )
            steps.append(
                StepRecord(
                    name=f"revise_{round_num}",
                    prompt_key="revise",
                    response=current,
                    metadata={"round": round_num},
                )
            )
            self._notify_step(typed_config, steps[-1])

        revise_steps = [s for s in steps if s.name.startswith("revise_")]
        return StrategyResult(
            output=current,
            steps=steps,
            metadata={
                "rounds_used": len(revise_steps) + 1,
                "satisfied": any(s.name == "stop" for s in steps),
            },
        )


__all__ = ["CritiqueResult", "ReflectionStrategy"]
