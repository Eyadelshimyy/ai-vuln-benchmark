"""Collaborative editing strategy — LLM generate → user edit → LLM continue.

The LLM produces an initial draft, the user edits it through any
:class:`EditCallback` implementation, and the LLM continues from the
edited version. The loop ends when the user signals completion
(via *done_signal*), leaves the content unchanged, returns empty
content, or ``max_rounds`` is reached.

Required prompts:
    - ``generate``: produce the initial draft
    - ``continue``: receives ``{{edited_content}}`` and ``{{original_content}}``

Optional prompts:
    - ``system``: persona/context forwarded to every LLM call
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ellements.core import LLMClient

from .callbacks import EditCallback
from .config import CollaborativeEditingConfig, StrategyConfigInput
from .strategies import BaseStrategy, StepRecord, StrategyResult

_logger = logging.getLogger(__name__)


class CollaborativeEditingStrategy(BaseStrategy):
    """LLM generate → user edit → LLM continue loop.

    Config keys:
        edit_callback (EditCallback): How to present content and receive edits.
        max_rounds (int): Maximum edit rounds after the initial generation.
        done_signal (str): If the user adds this string on its own line,
            the strategy finishes immediately.
        continue_on_unchanged (bool): If True, the LLM continues even
            when the user returns content identical to the LLM's
            previous output.
        generate_temperature (float): Temperature for initial generation.
        continue_temperature (float): Temperature for continuation.
    """

    REQUIRED_PROMPTS = ("generate", "continue")

    async def execute(
        self,
        prompts: Mapping[str, str],
        client: LLMClient,
        tools: list[dict[str, Any]] | Mapping[str, Any] | None = None,
        config: StrategyConfigInput = None,
    ) -> StrategyResult:
        typed_config = self._normalize_config(config, CollaborativeEditingConfig)
        max_rounds = typed_config.max_rounds
        done_signal = typed_config.done_signal
        continue_on_unchanged = typed_config.continue_on_unchanged
        callback: EditCallback = typed_config.edit_callback
        system = prompts.get("system")
        steps: list[StepRecord] = []

        generate_temp = self._stage_temperature(
            typed_config, "generate", default=typed_config.generate_temperature
        )
        continue_temp = self._stage_temperature(
            typed_config, "continue", default=typed_config.continue_temperature
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
            StepRecord(
                name="generate",
                prompt_key="generate",
                response=current,
                metadata={"round": 0, "source": "llm"},
            )
        )
        self._notify_step(typed_config, steps[-1])

        continue_template = self._get_prompt(prompts, "continue")

        for round_num in range(max_rounds):
            original = current
            context = f"Round {round_num + 1} of {max_rounds}"
            edited = await callback.request_edit(current, context)

            if not edited.strip():
                _logger.info(
                    "Collaborative editing aborted by user (empty content)"
                )
                steps.append(
                    StepRecord(
                        name=f"abort_{round_num}",
                        prompt_key="continue",
                        response="",
                        metadata={"round": round_num + 1, "reason": "user_abort"},
                    )
                )
                self._notify_step(typed_config, steps[-1])
                break

            if _has_done_signal(edited, done_signal):
                edited = _strip_done_signal(edited, done_signal)
                steps.append(
                    StepRecord(
                        name=f"user_edit_{round_num}",
                        prompt_key="continue",
                        response=edited,
                        metadata={
                            "round": round_num + 1,
                            "source": "user",
                            "done_signal": True,
                        },
                    )
                )
                self._notify_step(typed_config, steps[-1])
                current = edited
                break

            steps.append(
                StepRecord(
                    name=f"user_edit_{round_num}",
                    prompt_key="continue",
                    response=edited,
                    metadata={"round": round_num + 1, "source": "user"},
                )
            )
            self._notify_step(typed_config, steps[-1])

            if not continue_on_unchanged and edited.strip() == original.strip():
                _logger.info(
                    "Collaborative editing done: user approved without changes"
                )
                current = edited
                break

            continue_prompt = self._render_template(
                continue_template,
                {"edited_content": edited, "original_content": original},
            )
            current = await self._complete(
                client,
                continue_prompt,
                config=typed_config,
                temperature=continue_temp,
                system=system,
                tools=tools,
            )
            steps.append(
                StepRecord(
                    name=f"continue_{round_num}",
                    prompt_key="continue",
                    response=current,
                    metadata={"round": round_num + 1, "source": "llm"},
                )
            )
            self._notify_step(typed_config, steps[-1])

        rounds_completed = len(
            [s for s in steps if s.name.startswith("user_edit_")]
        )
        return StrategyResult(
            output=current,
            steps=steps,
            metadata={
                "rounds_completed": rounds_completed,
                "max_rounds": max_rounds,
            },
        )


def _has_done_signal(content: str, signal: str) -> bool:
    return any(line.strip() == signal for line in content.splitlines())


def _strip_done_signal(content: str, signal: str) -> str:
    lines = [line for line in content.splitlines() if line.strip() != signal]
    return "\n".join(lines).strip()


__all__ = ["CollaborativeEditingStrategy"]
